#!/usr/bin/env python3
"""
Prépare le modèle actuel (Checkpoint 9600 / ~40 époques) comme nouvelle base de départ
et crée des sauvegardes propres.

Génère :
1. `models/mms-tts-vezo-v2-base-with-disc` : Nouveau modèle de base complet (Générateur + Discriminateur Vezo)
2. `models/mms-tts-vezo-epoch40-inference` : Version légère autonome pour l'inférence / test TTS
3. `models/mms-tts-vezo-checkpoint9600-backup` : Sauvegarde complète de sécurité
"""

import os, sys, shutil, ctypes
from pathlib import Path

# Preload CUDA libraries (MUST BE BEFORE TORCH / SAFETENSORS)
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except Exception: pass

import torch
from safetensors.torch import load_file, save_file

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CKPT_DIR = MODELS_DIR / "mms-tts-vezo-finetuned" / "checkpoint-9600"
CONFIG_BASE = MODELS_DIR / "mms-tts-mlg-with-disc"

NEW_BASE_DIR = MODELS_DIR / "mms-tts-vezo-v2-base-with-disc"
INFERENCE_BACKUP_DIR = MODELS_DIR / "mms-tts-vezo-epoch40-inference"
RAW_BACKUP_DIR = MODELS_DIR / "mms-tts-vezo-checkpoint9600-backup"


def main():
    print("=" * 65)
    print("📦 CRÉATION DU NOUVEAU MODÈLE DE BASE (VEZO V2 - 40 ÉPOQUES)")
    print("=" * 65)

    if not CKPT_DIR.exists():
        print(f"❌ Checkpoint introuvable : {CKPT_DIR}")
        sys.exit(1)

    # 1. Sauvegarde brute complète du checkpoint 9600
    print(f"\n1️⃣ Sauvegarde brute de sécurité dans : {RAW_BACKUP_DIR.name}/")
    if RAW_BACKUP_DIR.exists():
        shutil.rmtree(RAW_BACKUP_DIR)
    shutil.copytree(CKPT_DIR, RAW_BACKUP_DIR)
    print("   ✅ Sauvegarde brute effectuée")

    # 2. Création de la nouvelle base complète avec Discriminateur (332 Mo)
    print(f"\n2️⃣ Assemblage du nouveau modèle de base dans : {NEW_BASE_DIR.name}/")
    NEW_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Copier les fichiers de configuration
    for cfg_name in ["config.json", "tokenizer_config.json", "vocab.json", "added_tokens.json", "preprocessor_config.json"]:
        src = CONFIG_BASE / cfg_name
        if src.exists():
            shutil.copy2(src, NEW_BASE_DIR / cfg_name)

    # Combiner model.safetensors (générateur) et model_1.safetensors (discriminateur)
    gen_sd = load_file(str(CKPT_DIR / "model.safetensors"))
    disc_sd = load_file(str(CKPT_DIR / "model_1.safetensors"))

    combined_sd = {}
    for k, v in gen_sd.items():
        combined_sd[k] = v
    for k, v in disc_sd.items():
        combined_sd[k] = v

    save_file(combined_sd, str(NEW_BASE_DIR / "model.safetensors"))
    print(f"   ✅ Modèle complet (Générateur + Discriminateur) sauvegardé ({len(combined_sd)} tenseurs, ~332 Mo)")

    # 3. Création du modèle d'inférence autonome propre (145 Mo)
    print(f"\n3️⃣ Préparation du modèle d'inférence autonome dans : {INFERENCE_BACKUP_DIR.name}/")
    INFERENCE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Copier les fichiers de config
    for cfg_name in ["config.json", "tokenizer_config.json", "vocab.json", "added_tokens.json", "preprocessor_config.json"]:
        src = CONFIG_BASE / cfg_name
        if src.exists():
            shutil.copy2(src, INFERENCE_BACKUP_DIR / cfg_name)

    # Copier le modèle converti prêt à l'emploi
    current_model = MODELS_DIR / "mms-tts-vezo-finetuned" / "model.safetensors"
    if current_model.exists():
        shutil.copy2(str(current_model), str(INFERENCE_BACKUP_DIR / "model.safetensors"))

    print("   ✅ Modèle d'inférence autonome prêt")

    print("\n" + "=" * 65)
    print("🎉 TOUT EST PRÊT !")
    print(f"📂 Nouveau modèle de base : {NEW_BASE_DIR}")
    print(f"📂 Modèle de test (Epoch 40) : {INFERENCE_BACKUP_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
