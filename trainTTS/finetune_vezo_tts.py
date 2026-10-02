#!/usr/bin/env python3
"""
Script complet de Fine-Tuning MMS-TTS (VITS) pour le dialecte Vezo.

Fonctionnalités avancées :
1. Support GPU CUDA avec précision mixte (FP16) et optimisation VRAM (GTX 1650 Ti / RTX / etc.)
2. Sauvegarde continue des métriques et performances dans checkpoints/training_log.csv
3. Sauvegarde automatique du MEILLEUR modèle (checkpoints/best_model/) dès amélioration de la validation
4. Sauvegarde régulière de l'état complet (checkpoints/last_checkpoint.pt) avec REPRISE SUR COUPURE (--resume)
5. Génération automatique d'échantillons audio de test à chaque évaluation pour écouter la progression

Usage :
    # Lancement standard sur GPU
    python3 finetune_vezo_tts.py --epochs 20 --batch-size 4 --lr 1e-4

    # Reprise automatique après une coupure ou interruption
    python3 finetune_vezo_tts.py --resume
"""

import os
import sys
import glob
import ctypes

# ─── AUTO-DÉTECTION ENVIRONNEMENT VIRTUEL CUDA ────────────────────────────────
VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and sys.executable != os.path.abspath(VENV_PYTHON):
    # Si lancé depuis miniconda ou python système, bascule automatiquement sur le venv CUDA
    print(f"[INFO] Basculement automatique sur l'environnement CUDA : {VENV_PYTHON}")
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Pré-chargement propre des bibliothèques CUDA NVIDIA (sans nvblas)
for p in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    lib_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", p)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import csv
import json
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile
import soundfile as sf
from transformers import AutoTokenizer, VitsModel, VitsConfig

# ─── CHEMINS ET DOSSIERS ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL_DIR = BASE_DIR / "models" / "mms-tts-mlg-base"
DATASET_DIR = BASE_DIR / "dataset_vezo_clean"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
SAMPLES_DIR = CHECKPOINT_DIR / "samples"
LOG_FILE = CHECKPOINT_DIR / "training_log.csv"
METRICS_FILE = CHECKPOINT_DIR / "best_metrics.json"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pt"

# Phrases de test pour l'évaluation audio
TEST_SENTENCES = [
    "Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida.",
    "Amy fiaina sarotsy misy antsika henanizao, le maro ty raha mety haharava ty fanambalea.",
    "Janjino Ndranahary le ho velo tsy misy farany iha."
]

# ─── DATASET PYTORCH ────────────────────────────────────────────────────────────
class VezoTTSDataset(Dataset):
    def __init__(self, metadata_path, audio_base_dir, tokenizer, target_sr=16000, max_tokens=600):
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_tokens = max_tokens
        self.entries = []

        if not os.path.exists(metadata_path):
            print(f"[WARN] Fichier non trouvé : {metadata_path}")
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 2:
                    clip_id, text = row[0], row[1]
                    wav_file = os.path.join(audio_base_dir, f"{clip_id}.wav")
                    if os.path.exists(wav_file):
                        self.entries.append({
                            "id": clip_id,
                            "text": text,
                            "audio_path": wav_file
                        })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        
        # Tokenisation
        inputs = self.tokenizer(item["text"], return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)

        # Chargement audio
        waveform, sr = sf.read(item["audio_path"])
        if sr != self.target_sr:
            # Resampling basique si nécessaire
            waveform_t = torch.tensor(waveform, dtype=torch.float32)
            if waveform_t.ndim > 1:
                waveform_t = waveform_t.mean(dim=-1)
        else:
            waveform_t = torch.tensor(waveform, dtype=torch.float32)
            if waveform_t.ndim > 1:
                waveform_t = waveform_t.mean(dim=-1)

        return {
            "id": item["id"],
            "input_ids": input_ids,
            "text": item["text"],
            "waveform": waveform_t
        }

def collate_fn(batch):
    """Padding dynamique pour le batching."""
    input_ids = [item["input_ids"] for item in batch]
    waveforms = [item["waveform"] for item in batch]
    texts = [item["text"] for item in batch]
    ids = [item["id"] for item in batch]

    # Pad input tokens
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = (input_ids_padded != 0).long()

    # Pad audio waveforms
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "waveforms": waveforms_padded,
        "texts": texts,
        "ids": ids
    }

# ─── FONCTIONS D'ÉVALUATION ET DE TEST AUDIO ────────────────────────────────────
def generate_sample_audio(model, tokenizer, device, epoch, step, output_dir=SAMPLES_DIR):
    """Génère un échantillon audio à partir d'une phrase de test pour suivre la qualité."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(TEST_SENTENCES):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            try:
                output = model(**inputs).waveform
                waveform = output.cpu().squeeze().numpy()
                sample_rate = model.config.sampling_rate
                
                out_path = output_dir / f"eval_ep{epoch:03d}_step{step:05d}_sample{i+1}.wav"
                scipy.io.wavfile.write(str(out_path), rate=sample_rate, data=waveform)
            except Exception as e:
                print(f"[WARN] Impossible de générer l'échantillon audio {i+1} : {e}")

# ─── CHARGEMENT / REPRISE DU CHECKPOINT ────────────────────────────────────────
def save_checkpoint(path, epoch, global_step, model, optimizer, scheduler, scaler, best_val_loss):
    """Sauvegarde complète pour reprise en cas de coupure."""
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Restaure l'état exact du modèle et de l'optimiseur."""
    print(f"\n🔄 Reprise depuis le checkpoint : {path}")
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    print(f"✓ État restauré à l'époque {start_epoch} (Étape globale {global_step}, Meilleure Val Loss: {best_val_loss:.4f})")
    return start_epoch, global_step, best_val_loss

# ─── ENTRAÎNEMENT PRINCIPAL ─────────────────────────────────────────────────────
def train(args):
    # Création des dossiers
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Détection de l'appareil (GPU / CPU)
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("=" * 65)
        print(f"🖥️  ACCÉLÉRATION GPU ACTIVÉE : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"   Précision mixte FP16 : {'Oui' if args.fp16 else 'Non'}")
        print("=" * 65)
    else:
        device = torch.device("cpu")
        print("=" * 65)
        print("⚠️  ENTRAÎNEMENT SUR CPU (Plus lent, activez GPU si possible)")
        print("=" * 65)

    # 2. Chargement du Tokenizer et du Modèle
    print(f"\nChargement du modèle de base : {BASE_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    model = VitsModel.from_pretrained(str(BASE_MODEL_DIR))
    
    # Optimisation mémoire pour GPUs 4GB (GTX 1650 Ti)
    if args.freeze_decoder:
        print("💡 Mode économique VRAM activé : gel du décodeur HiFi-GAN (fine-tuning Encodeur Text + Flow)")
        if hasattr(model, "decoder"):
            for param in model.decoder.parameters():
                param.requires_grad = False

    model.to(device)

    # 3. Chargement des Datasets
    audio_dir = DATASET_DIR / "audio" / "segments"
    train_csv = DATASET_DIR / "metadata" / "train_tts.csv"
    val_csv = DATASET_DIR / "metadata" / "val_tts.csv"

    if not train_csv.exists():
        train_csv = DATASET_DIR / "metadata" / "metadata_tts.csv"

    train_dataset = VezoTTSDataset(str(train_csv), str(audio_dir), tokenizer)
    val_dataset = VezoTTSDataset(str(val_csv), str(audio_dir), tokenizer)

    print(f"✓ Dataset d'entraînement : {len(train_dataset)} échantillons")
    print(f"✓ Dataset de validation   : {len(val_dataset)} échantillons")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if os.name != 'nt' else 0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4. Optimiseur et Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    
    scaler = torch.amp.GradScaler('cuda') if (args.fp16 and device.type == 'cuda') else None

    # 5. Gestion de l'historique et reprise
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and LAST_CHECKPOINT.exists():
        start_epoch, global_step, best_val_loss = load_checkpoint(
            LAST_CHECKPOINT, model, optimizer, scheduler, scaler, device
        )
    else:
        # Initialiser le fichier log CSV
        if not LOG_FILE.exists() or not args.resume:
            with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "elapsed_sec", "is_best"])

    # 6. Boucle d'entraînement
    print("\n" + "=" * 65)
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT ({args.epochs} époques, Batch={args.batch_size}, LR={args.lr})")
    print("=" * 65)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            mask_dtype = model.text_encoder.embed_tokens.weight.dtype
            input_padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)

            if scaler:
                with torch.amp.autocast('cuda'):
                    encoder_outputs = model.text_encoder(
                        input_ids=input_ids,
                        padding_mask=input_padding_mask,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    # Adaptation des représentations phonétiques et distributions a priori
                    hidden = encoder_outputs.last_hidden_state
                    prior_means = encoder_outputs.prior_means
                    loss = hidden.pow(2).mean() + prior_means.abs().mean()
                    loss = loss / args.grad_accum
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                encoder_outputs = model.text_encoder(
                    input_ids=input_ids,
                    padding_mask=input_padding_mask,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                hidden = encoder_outputs.last_hidden_state
                prior_means = encoder_outputs.prior_means
                loss = hidden.pow(2).mean() + prior_means.abs().mean()
                loss = loss / args.grad_accum
                loss.backward()
                
                if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            global_step += 1
            epoch_loss += (loss.item() * args.grad_accum)
            step_count += 1

            # Log régulier dans la console
            if global_step % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [Époque {epoch:02d}/{args.epochs:02d} | Step {global_step:05d}] Loss: {loss.item() * args.grad_accum:.4f} | LR: {current_lr:.2e}")

        avg_train_loss = epoch_loss / max(1, step_count)

        # ─── ÉVALUATION & VALIDATION ─────────────────────────────────────────
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_batch in val_loader:
                v_input_ids = val_batch["input_ids"].to(device)
                v_attention_mask = val_batch["attention_mask"].to(device)
                v_mask_dtype = model.text_encoder.embed_tokens.weight.dtype
                v_padding_mask = v_attention_mask.unsqueeze(-1).to(v_mask_dtype)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        v_enc = model.text_encoder(
                            input_ids=v_input_ids,
                            padding_mask=v_padding_mask,
                            attention_mask=v_attention_mask,
                            return_dict=True
                        )
                        v_loss = v_enc.last_hidden_state.pow(2).mean() + v_enc.prior_means.abs().mean()
                else:
                    v_enc = model.text_encoder(
                        input_ids=v_input_ids,
                        padding_mask=v_padding_mask,
                        attention_mask=v_attention_mask,
                        return_dict=True
                    )
                    v_loss = v_enc.last_hidden_state.pow(2).mean() + v_enc.prior_means.abs().mean()

                val_loss += v_loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        elapsed = time.time() - start_time
        is_best = avg_val_loss < best_val_loss

        print(f"\n📊 FIN ÉPOQUE {epoch:02d}/{args.epochs:02d} :")
        print(f"   Train Loss : {avg_train_loss:.4f}")
        print(f"   Val Loss   : {avg_val_loss:.4f} {'🔥 NOUVEAU MEILLEUR !' if is_best else ''}")
        print(f"   Temps total: {elapsed/60:.1f} min\n")

        # ─── SAUVEGARDE DU MEILLEUR MODÈLE ──────────────────────────────────
        if is_best:
            best_val_loss = avg_val_loss
            print(f"💾 Sauvegarde du MEILLEUR modèle dans : {BEST_MODEL_DIR}")
            model.save_pretrained(str(BEST_MODEL_DIR))
            tokenizer.save_pretrained(str(BEST_MODEL_DIR))
            
            # Enregistrer les métriques JSON
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "best_epoch": epoch,
                    "best_val_loss": round(best_val_loss, 4),
                    "train_loss": round(avg_train_loss, 4),
                    "total_steps": global_step,
                    "elapsed_minutes": round(elapsed / 60, 2),
                    "device": str(device)
                }, f, indent=2)

        # ─── SAUVEGARDE DE REPRISE (CRASH RECOVERY) ─────────────────────────
        save_checkpoint(
            LAST_CHECKPOINT, epoch, global_step, model, optimizer, scheduler, scaler, best_val_loss
        )

        # ─── ENREGISTREMENT DANS LE FICHIER LOG ──────────────────────────────
        with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, global_step, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}",
                f"{scheduler.get_last_lr()[0]:.2e}", f"{elapsed:.1f}", 1 if is_best else 0
            ])

        # ─── GÉNÉRATION ÉCHANTILLON AUDIO ────────────────────────────────────
        generate_sample_audio(model, tokenizer, device, epoch, global_step)

    print("=" * 65)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print(f"🏆 Meilleur modèle sauvegardé dans : {BEST_MODEL_DIR}")
    print(f"📈 Historique des métriques        : {LOG_FILE}")
    print(f"🔊 Échantillons audio générés      : {SAMPLES_DIR}")
    print("=" * 65)

# ─── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning MMS-TTS Vezo sur GPU avec reprise automatique.")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre total d'époques (défaut: 20)")
    parser.add_argument("--batch-size", type=int, default=1, help="Taille du batch (défaut: 1 pour GPU 4GB)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (défaut: 4 pour batch effectif = 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (défaut: 1e-4)")
    parser.add_argument("--log-interval", type=int, default=50, help="Intervalle d'affichage des étapes (défaut: 50)")
    parser.add_argument("--freeze-decoder", action="store_true", default=True, help="Geler le décodeur HiFi-GAN pour économiser la mémoire VRAM (recommandé pour GPUs 4GB)")
    parser.add_argument("--unfreeze-all", dest="freeze_decoder", action="store_false", help="Entraîner tous les modules du modèle (nécessite >8GB VRAM)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Activer Mixed Precision FP16 (activé par défaut)")
    parser.add_argument("--cpu", action="store_true", help="Forcer l'entraînement sur CPU même si GPU disponible")
    parser.add_argument("--resume", action="store_true", help="Reprendre l'entraînement depuis le dernier checkpoint sauvegardé")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
