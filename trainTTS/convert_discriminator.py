#!/usr/bin/env python3
"""
ÉTAPE 2 : Convertit le checkpoint facebook/mms-tts-mlg pour ajouter le discriminateur.

Le modèle de base (145 Mo) ne contient que le GÉNÉRATEUR.
Pour un fine-tuning VITS correct, il faut le DISCRIMINATEUR (= 332 Mo au total).
Le discriminateur original est dans le repo facebook/mms-tts (fichier D_100000.pth).

Ce script est une VERSION ADAPTÉE de finetune-hf-vits/convert_original_discriminator_checkpoint.py
avec préchargement CUDA intégré et gestion du chemin relatif correct.
"""

import os, sys, ctypes

# ─── Préchargement CUDA (OBLIGATOIRE avant tout import torch) ────────────────
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

# ─── Ajouter finetune-hf-vits au path ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VITS_REPO = os.path.join(SCRIPT_DIR, "finetune-hf-vits")
sys.path.insert(0, VITS_REPO)

# Aussi ajouter monotonic_align au path
MONO_DIR = os.path.join(VITS_REPO, "monotonic_align")
sys.path.insert(0, MONO_DIR)

import torch
from transformers.models.vits.modeling_vits import VitsModel
from huggingface_hub import hf_hub_download

from utils.feature_extraction_vits import VitsFeatureExtractor
from utils.configuration_vits import VitsConfig, logging
from utils.modeling_vits_training import VitsDiscriminator, VitsModelForPreTraining

logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.vits")

MAPPING = {"conv_post": "final_conv"}
LANGUAGE = "mlg"
GENERATOR_PATH = os.path.join(SCRIPT_DIR, "models", "mms-tts-mlg-base")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "models", "mms-tts-mlg-with-disc")


@torch.no_grad()
def convert_checkpoint():
    print("=" * 65)
    print("🔧 CONVERSION DU CHECKPOINT MMS-TTS + AJOUT DU DISCRIMINATEUR")
    print("=" * 65)

    # Télécharger le checkpoint discriminateur depuis HuggingFace
    print(f"\n📥 Téléchargement de D_100000.pth depuis facebook/mms-tts (langue: {LANGUAGE})...")
    checkpoint_path = hf_hub_download(
        repo_id="facebook/mms-tts",
        subfolder=f"full_models/{LANGUAGE}",
        filename="D_100000.pth",
        local_dir=os.path.join(SCRIPT_DIR, "models", "disc_raw")
    )
    print(f"   ✅ Sauvegardé : {checkpoint_path}")

    # Charger la config et le générateur
    print(f"\n📂 Chargement du générateur depuis : {GENERATOR_PATH}")
    config = VitsConfig.from_pretrained(GENERATOR_PATH)
    generator = VitsModel.from_pretrained(GENERATOR_PATH)
    print(f"   ✅ Config : {config.hidden_size}d, sample_rate={config.sampling_rate}")

    # Créer le discriminateur
    print("\n🏗️  Initialisation du discriminateur VITS...")
    discriminator = VitsDiscriminator(config)
    for disc in discriminator.discriminators:
        disc.apply_weight_norm()

    # Charger les poids du discriminateur
    print(f"\n📦 Chargement des poids du discriminateur...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # Mapper les clés
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for old_name, new_name in MAPPING.items():
            new_k = new_k.replace(old_name, new_name)
        new_state_dict[new_k] = v

    extra_keys = set(new_state_dict.keys()) - set(discriminator.state_dict().keys())
    missing_keys = set(discriminator.state_dict().keys()) - set(new_state_dict.keys())

    if extra_keys:
        print(f"   ⚠️  Clés supplémentaires ignorées : {extra_keys}")
    if missing_keys:
        print(f"   ⚠️  Clés manquantes : {missing_keys}")

    discriminator.load_state_dict(new_state_dict, strict=False)
    print("   ✅ Poids du discriminateur chargés")

    for disc in discriminator.discriminators:
        disc.remove_weight_norm()

    # Assembler le modèle complet pour le fine-tuning
    print("\n🔗 Assemblage du modèle complet (générateur + discriminateur)...")
    model_for_training = VitsModelForPreTraining(config)
    model_for_training.text_encoder = generator.text_encoder
    model_for_training.flow = generator.flow
    model_for_training.decoder = generator.decoder
    model_for_training.duration_predictor = generator.duration_predictor
    model_for_training.posterior_encoder = generator.posterior_encoder
    if config.num_speakers > 1:
        model_for_training.embed_speaker = generator.embed_speaker
    model_for_training.discriminator = discriminator

    # Sauvegarder
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"\n💾 Sauvegarde dans : {OUTPUT_PATH}")
    model_for_training.save_pretrained(OUTPUT_PATH)

    # Copier le tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Feature extractor
    feature_extractor = VitsFeatureExtractor()
    feature_extractor.save_pretrained(OUTPUT_PATH)

    size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in os.listdir(OUTPUT_PATH)
        if os.path.isfile(os.path.join(OUTPUT_PATH, f))
    ) / 1e6
    print(f"\n✅ Modèle sauvegardé ({size_mb:.0f} Mo) dans : {OUTPUT_PATH}")
    print("   → Taille attendue ~332 Mo (avec discriminateur)")
    print("\n🎯 Prochaine étape : lancer le fine-tuning avec run_vits_finetuning.py")


if __name__ == "__main__":
    convert_checkpoint()
