#!/usr/bin/env python3
"""
ÉTAPE 3 : Génère un dataset HuggingFace local (format Arrow) depuis le CSV normalisé.

Le script run_vits_finetuning.py de ylacombe/finetune-hf-vits attend un dataset
au format HuggingFace (datasets.DatasetDict) avec des colonnes :
  - audio : {"array": np.ndarray, "sampling_rate": int, "path": str}
  - text  : str

Ce script charge le dataset_vezo_clean et le convertit dans ce format,
puis le sauvegarde localement pour que run_vits_finetuning.py puisse le charger
via dataset_name=<chemin local>.
"""

import os, sys, ctypes

# ─── Préchargement CUDA ───────────────────────────────────────────────────────
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

from pathlib import Path
import numpy as np
import soundfile as sf
import datasets
from datasets import Dataset, DatasetDict, Audio, Value, Features

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DIR  = SCRIPT_DIR / "dataset_vezo_clean" / "audio" / "segments"
META_DIR   = SCRIPT_DIR / "dataset_vezo_clean" / "metadata"
TRAIN_CSV  = META_DIR / "train_tts.csv" if (META_DIR / "train_tts.csv").exists() else (META_DIR / "train_tts_normalized.csv")
VAL_CSV    = META_DIR / "val_tts.csv" if (META_DIR / "val_tts.csv").exists() else (META_DIR / "val_tts_normalized.csv")
OUTPUT_DIR = SCRIPT_DIR / "dataset_vezo_hf"

SAMPLING_RATE = 16000


def load_split(csv_path: Path, audio_dir: Path, split_name: str):
    """Charge un split depuis le CSV normalisé."""
    rows = []
    missing = 0

    with open(csv_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            clip_id, text = parts[0], parts[1]

            # Chercher le fichier audio
            audio_path = audio_dir / f"{clip_id}.wav"
            if not audio_path.exists():
                missing += 1
                continue

            rows.append({
                "audio": str(audio_path),
                "text":  text.strip(),
                "clip_id": clip_id,
            })

    print(f"  [{split_name}] {len(rows)} clips chargés ({missing} manquants ignorés)")
    return rows


def build_dataset():
    print("=" * 65)
    print("🔨 CONSTRUCTION DU DATASET HUGGINGFACE VEZO")
    print("=" * 65)

    # Vérification des fichiers normalisés
    if not TRAIN_CSV.exists():
        print(f"❌ Fichier manquant : {TRAIN_CSV}")
        print("   → Lancez d'abord : python prepare_dataset_for_vits.py")
        sys.exit(1)

    print(f"\n📂 Source CSV   : {META_DIR}")
    print(f"📂 Source audio : {AUDIO_DIR}")

    train_rows = load_split(TRAIN_CSV, AUDIO_DIR, "train")
    val_rows   = load_split(VAL_CSV,   AUDIO_DIR, "validation")

    print(f"\n📊 Total : {len(train_rows)} train + {len(val_rows)} val")

    print("\n⏳ Création des datasets HuggingFace...")

    train_data = {
        "audio": [r["audio"] for r in train_rows],
        "text": [r["text"] for r in train_rows],
        "clip_id": [r["clip_id"] for r in train_rows],
    }
    val_data = {
        "audio": [r["audio"] for r in val_rows],
        "text": [r["text"] for r in val_rows],
        "clip_id": [r["clip_id"] for r in val_rows],
    }

    train_dataset = Dataset.from_dict(train_data).cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    val_dataset   = Dataset.from_dict(val_data).cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    dataset_dict = DatasetDict({
        "train":      train_dataset,
        "validation": val_dataset,
    })

    # Sauvegarder
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n💾 Sauvegarde dans : {OUTPUT_DIR}")
    dataset_dict.save_to_disk(str(OUTPUT_DIR))

    print(f"✅ Dataset HuggingFace sauvegardé !")
    print(f"   Train       : {len(train_rows)} clips")
    print(f"   Validation  : {len(val_rows)} clips")
    print(f"   Emplacement : {OUTPUT_DIR}")
    print(f"\n🎯 Utilisez ce chemin dans la config : \"dataset_name\": \"{OUTPUT_DIR}\"")

    return str(OUTPUT_DIR)


if __name__ == "__main__":
    build_dataset()
