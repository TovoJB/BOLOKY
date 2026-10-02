#!/usr/bin/env python3
"""
Test rapide pour le modèle fine-tuné importé depuis Google Colab.
Emplacement du modèle : colab_vezo_tts/models/mms-tts-vezo-finetuned/models/mms-tts-vezo-finetuned
"""

import os, sys, ctypes, argparse
from pathlib import Path

# ─── AUTO-DÉTECTION VENV PYTHON ──────────────────────────────────────────────
VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and sys.executable != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

# Preload CUDA libraries
for p in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    lib_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", p)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

from test_model import generate_voice, interactive_mode, compare_models, DEFAULT_BASE_MODEL

BASE_DIR = Path(__file__).resolve().parent
COLAB_MODEL = BASE_DIR / "colab_vezo_tts" / "models" / "mms-tts-vezo-finetuned" / "models" / "mms-tts-vezo-finetuned"

if not COLAB_MODEL.exists():
    # Fallback si déplacé
    COLAB_MODEL = BASE_DIR / "colab_vezo_tts" / "models" / "mms-tts-vezo-finetuned"

def main():
    parser = argparse.ArgumentParser(description="Tester rapidement le modèle entraîné sur Google Colab.")
    parser.add_argument("--text", "-t", type=str, help="Texte en Vezo à synthétiser")
    parser.add_argument("--sr", "--sample-rate", type=int, default=22050, help="Fréquence d'échantillonnage de sortie (défaut: 22050 Hz)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Vitesse d'élocution (défaut: 1.0)")
    parser.add_argument("--noise-scale", type=float, default=0.35, help="Clarté / réduction du bruit (défaut: 0.35)")
    parser.add_argument("--compare", "-c", action="store_true", help="Comparer avec le modèle de base")
    parser.add_argument("--no-play", action="store_true", help="Ne pas jouer automatiquement le son")

    args = parser.parse_args()

    print("=" * 65)
    print("🚀 TEST DU MODÈLE FINE-TUNÉ GOOGLE COLAB")
    print(f"📂 Emplacement : {COLAB_MODEL}")
    print("=" * 65)

    if args.compare:
        test_text = args.text or "Amy fiaina sarotsy misy antsika henanizao, maro ty raha mety haharava ty fanambalea."
        print(f"\n[1/2] Modèle de BASE :")
        generate_voice(test_text, DEFAULT_BASE_MODEL, "compare_base.wav", play_audio=False, speed=args.speed, target_sr=args.sr)
        print(f"\n[2/2] Modèle COLAB Fine-tuné :")
        generate_voice(test_text, COLAB_MODEL, "compare_colab.wav", play_audio=not args.no_play, speed=args.speed, noise_scale=args.noise_scale, target_sr=args.sr)
    elif args.text:
        generate_voice(args.text, COLAB_MODEL, "colab_sample.wav", play_audio=not args.no_play, speed=args.speed, noise_scale=args.noise_scale, target_sr=args.sr)
    else:
        interactive_mode(COLAB_MODEL, speed=args.speed, noise_scale=args.noise_scale)


if __name__ == "__main__":
    main()
