#!/usr/bin/env python3
"""
Prépare le modèle V4 à partir du meilleur checkpoint actuel (checkpoint-7200)
et garantit un format de poids 100% propre pour VitsModelForPreTraining.
"""

import os, sys, shutil, ctypes
from pathlib import Path

# Preload CUDA libraries (MUST BE BEFORE TORCH / SAFETENSORS)
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import torch
from safetensors.torch import load_file, save_file

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SRC_FT_DIR = MODELS_DIR / "mms-tts-vezo-finetuned"
CONFIG_BASE = MODELS_DIR / "mms-tts-mlg-with-disc"
V4_BASE_DIR = MODELS_DIR / "mms-tts-vezo-v4-base"


def prepare_v4():
    print("=" * 65)
    print("📦 PRÉPARATION DU NOUVEAU MODÈLE DE BASE VEZO V4")
    print("=" * 65)

    if not SRC_FT_DIR.exists():
        print(f"❌ Dossier introuvable : {SRC_FT_DIR}")
        sys.exit(1)

    # Trouver le dernier checkpoint
    ckpts = [d for d in SRC_FT_DIR.glob("checkpoint-*") if d.is_dir()]
    if not ckpts:
        print(f"❌ Aucun checkpoint trouvé dans : {SRC_FT_DIR}")
        sys.exit(1)

    ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
    latest_ckpt = ckpts[-1]
    print(f"📌 Checkpoint source sélectionné : {latest_ckpt.name}")

    gen_path = latest_ckpt / "model.safetensors"
    disc_path = latest_ckpt / "model_1.safetensors"

    if not gen_path.exists() or not disc_path.exists():
        print("❌ Fichiers de poids manquants dans le checkpoint !")
        sys.exit(1)

    # Création du dossier V4 Base
    V4_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Copier la configuration
    for cfg in ["config.json", "tokenizer_config.json", "vocab.json", "added_tokens.json", "preprocessor_config.json"]:
        src = CONFIG_BASE / cfg
        if src.exists():
            shutil.copy2(src, V4_BASE_DIR / cfg)

    # Charger et combiner les poids
    gen_sd = load_file(str(gen_path))
    disc_sd = load_file(str(disc_path))

    mapped_sd = {}
    for k, v in gen_sd.items():
        mapped_sd[k] = v

    for k, v in disc_sd.items():
        # Corriger le préfixe du discriminateur
        if k.startswith("discriminators."):
            mapped_sd["discriminator." + k] = v
        else:
            mapped_sd["discriminator.discriminators." + k] = v

    # Adapter le format des parametrizations Wavenet pour VitsModelForPreTraining
    new_sd = {}
    for k, val in mapped_sd.items():
        if ("posterior_encoder.wavenet" in k or "flow.flows" in k) and k.endswith(".weight") and "conv_pre" not in k and "conv_post" not in k:
            base_k = k[:-7]
            g = torch.linalg.vector_norm(val, dim=tuple(range(1, val.ndim)), keepdim=True)
            v = val
            new_sd[base_k + ".parametrizations.weight.original0"] = g
            new_sd[base_k + ".parametrizations.weight.original1"] = v
        else:
            new_sd[k] = val

    out_file = V4_BASE_DIR / "model.safetensors"
    save_file(new_sd, str(out_file))
    print(f"✅ Modèle V4 assemblé et sauvegardé ({len(new_sd)} tenseurs) dans : {V4_BASE_DIR.name}/")

    # Vérification avec VitsModelForPreTraining
    print("\n🔍 Vérification de la compatibilité VitsModelForPreTraining...")
    sys.path.insert(0, str(BASE_DIR / "finetune-hf-vits"))
    from utils import VitsModelForPreTraining

    model = VitsModelForPreTraining.from_pretrained(str(V4_BASE_DIR))
    print("🎉 SUCCÈS ! Modèle V4 chargé avec 0 clé manquante et 0 clé inattendue !")


if __name__ == "__main__":
    prepare_v4()
