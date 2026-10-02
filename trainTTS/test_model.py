#!/usr/bin/env python3
"""
Script de test interactif et comparaison pour le modèle TTS Vezo / Malagasy.

Usage :
    # 1. Tester une phrase Vezo directement en ligne de commande :
    ./test_model.py --text "Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida: 25 taona."

    # 2. Comparer côte-à-côte le modèle de BASE et le modèle FINE-TUNÉ :
    ./test_model.py --compare --text "Amy fiaina sarotsy misy antsika henanizao, maro ty raha mety haharava ty fanambalea."

    # 3. Mode interactif (taper des phrases au clavier) :
    ./test_model.py
"""

import os, sys, glob, ctypes, argparse, re
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

import torch
import numpy as np
import scipy.io.wavfile
import subprocess
from transformers import AutoTokenizer, AutoModelForTextToWaveform

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_BASE_MODEL  = MODELS_DIR / "mms-tts-mlg-with-disc"
if not DEFAULT_BASE_MODEL.exists():
    DEFAULT_BASE_MODEL = MODELS_DIR / "mms-tts-mlg-base"

OUTPUT_DIR = BASE_DIR / "tts_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_best_model_path() -> Path:
    """Trouve dynamiquement le modèle le plus avancé disponible."""
    candidates = [
        MODELS_DIR / "mms-tts-vezo-finetuned-v4" / "best_model",
        MODELS_DIR / "mms-tts-vezo-finetuned-v4",
        MODELS_DIR / "mms-tts-vezo-v4-base",
        MODELS_DIR / "mms-tts-vezo-finetuned" / "best_model",
        MODELS_DIR / "mms-tts-vezo-finetuned",
        DEFAULT_BASE_MODEL,
    ]
    for c in candidates:
        if c.exists() and (c / "model.safetensors").exists():
            return c
    return DEFAULT_BASE_MODEL

VALID_CHARS = set("abcdefghijklmnoprstyvz àìòôỳ'-|")

def normalize_text(text: str) -> str:
    """Normalise le texte pour le tokenizer MMS-TTS."""
    text = text.lower()
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('\u02bc', "'")
    text = text.replace('"', '').replace('\u201c', '').replace('\u201d', '').replace('«', '').replace('»', '')
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('é', 'e').replace('ê', 'e').replace('ë', 'e').replace('è', 'e')
    text = text.replace('â', 'a').replace('á', 'a').replace('ä', 'a')
    text = text.replace('î', 'i').replace('í', 'i').replace('ï', 'i')
    text = text.replace('û', 'u').replace('ú', 'u').replace('ü', 'u')
    text = text.replace('ç', 's').replace('ñ', 'n')

    num_map = {
        '0': 'folo', '1': 'iray', '2': 'roa', '3': 'telo',
        '4': 'efatra', '5': 'dimy', '6': 'enina', '7': 'fito',
        '8': 'valo', '9': 'sivy',
    }
    # Convertir ponctuation en pause naturelle MMS-TTS (|)
    text = re.sub(r'[,;:]+', ' | ', text)
    text = re.sub(r'[.!?]+', ' | ', text)
    text = re.sub(r'[()«»\[\]{}]', '', text)

    text = ''.join(c for c in text if c in VALID_CHARS)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'(\| )+', '| ', text)
    return re.sub(r' +', ' ', text).strip().strip('|').strip()


def post_process_audio(waveform, sr=16000):
    """Nettoyage acoustique propre sans distorsion métallique."""
    import scipy.signal
    # 1. Filtre passe-haut doux à 50Hz (supprime uniquement les infra-basses et le continu)
    sos = scipy.signal.butter(2, 50.0, btype='highpass', fs=sr, output='sos')
    clean = scipy.signal.sosfilt(sos, waveform)

    # 2. Normalisation crête douce (0.95) pour un volume propre sans saturation
    max_val = np.max(np.abs(clean))
    if max_val > 1e-5:
        clean = clean * (0.95 / max_val)
    return clean.astype(np.float32)


def generate_voice(text, model_path, output_name, play_audio=True, speed=1.0, noise_scale=0.55, noise_scale_dur=0.667, target_sr=16000):
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Erreur : Dossier modèle introuvable : {model_path}")
        return None

    clean_text = normalize_text(text)
    print(f"\n⏳ Modèle       : {model_path.name}")
    print(f"   Texte brut   : \"{text}\"")
    print(f"   Texte propre : \"{clean_text}\"")
    print(f"   Paramètres   : Sample Rate={target_sr}Hz | Vitesse={speed}x | noise_scale={noise_scale}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTextToWaveform.from_pretrained(str(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.speaking_rate = speed

    inputs = tokenizer(clean_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs, noise_scale=noise_scale, noise_scale_duration=noise_scale_dur).waveform

    waveform = output.cpu().squeeze().numpy()

    # 1. Post-traitement audio (Clarté + Anti-basses sourdes + Pré-accentuation)
    enhanced_wav = post_process_audio(waveform, target_sr)

    # 2. Sauvegarde du fichier corrigé
    out_file = OUTPUT_DIR / output_name
    scipy.io.wavfile.write(str(out_file), rate=target_sr, data=enhanced_wav)

    # 3. Sauvegarde de la version brute 16kHz pour référence
    raw_name = output_name.replace(".wav", "_raw16k.wav")
    scipy.io.wavfile.write(str(OUTPUT_DIR / raw_name), rate=16000, data=waveform)

    print(f"✅ Audio généré (Corrigé {target_sr}Hz) : {out_file}")
    print(f"   (Version brute 16kHz disponible dans  : {OUTPUT_DIR / raw_name})")

    if play_audio:
        try:
            subprocess.run(["aplay", str(out_file)], capture_output=True)
        except Exception:
            try:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", str(out_file)], capture_output=True)
            except Exception:
                pass

    return out_file


def compare_models(text):
    print("=" * 65)
    print(f"🔬 COMPARAISON DES MODÈLES :")
    print(f"   \"{text}\"")
    print("=" * 65)

    print("\n[1/2] Synthèse avec le MODÈLE DE BASE (facebook/mms-tts-mlg)...")
    base_wav = generate_voice(text, DEFAULT_BASE_MODEL, "compare_1_base.wav", play_audio=False)

    if DEFAULT_FT_MODEL.exists():
        print("\n[2/2] Synthèse avec le MODÈLE FINE-TUNÉ (Vezo)...")
        ft_wav = generate_voice(text, DEFAULT_FT_MODEL, "compare_2_finetuned.wav", play_audio=False)
    else:
        print(f"\n[INFO] Modèle fine-tuné en cours d'entraînement dans {DEFAULT_FT_MODEL}.")
        ft_wav = None

    print("\n" + "=" * 65)
    print("🎧 Fichiers audio générés dans tts_samples/ :")
    if base_wav:
        print(f"  - Base       : {base_wav}")
    if ft_wav:
        print(f"  - Fine-tuné  : {ft_wav}")
    print("=" * 65)


def interactive_mode(model_path, speed=1.15, noise_scale=0.4):
    current_model = model_path
    print("=" * 65)
    print("🎙️  MODE INTERACTIF TTS VEZO")
    print(f"Modèle actif : {current_model.name}")
    print(f"Paramètres   : Vitesse={speed}x | noise_scale={noise_scale}")
    print("Commandes    : ':base' (basculer base), ':ft' (basculer finetuné), 'q' (quitter)")
    print("Tapez votre phrase en Vezo / Malagasy et appuyez sur Entrée :")
    print("=" * 65)

    idx = 1
    while True:
        try:
            phrase = input(f"\n📝 [{current_model.name}] Texte > ").strip()
            if not phrase:
                continue
            if phrase.lower() in ['q', 'quit', 'exit']:
                print("Sortie.")
                break
            if phrase.lower() in [':base', 'base']:
                current_model = DEFAULT_BASE_MODEL
                print(f"🔄 Basculement sur le modèle DE BASE : {current_model.name}")
                continue
            if phrase.lower() in [':colab', 'colab']:
                current_model = DEFAULT_COLAB_MODEL
                print(f"🔄 Basculement sur le modèle COLAB : {current_model}")
                continue
            if phrase.lower() in [':ft', 'ft', ':finetuned']:
                current_model = DEFAULT_FT_MODEL if DEFAULT_FT_MODEL.exists() else DEFAULT_BASE_MODEL
                print(f"🔄 Basculement sur le modèle FINE-TUNÉ : {current_model.name}")
                continue

            out_name = f"interactive_{current_model.name}_{idx:03d}.wav"
            generate_voice(phrase, current_model, out_name, play_audio=True, speed=speed, noise_scale=noise_scale)
            idx += 1
        except (KeyboardInterrupt, EOFError):
            print("\nSortie.")
            break


def main():
    parser = argparse.ArgumentParser(description="Tester et écouter le modèle TTS Vezo.")
    parser.add_argument("--text", type=str, help="Texte à synthétiser")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modèle spécifique")
    parser.add_argument("--colab", action="store_true", help="Tester le modèle fine-tuné importé depuis Google Colab")
    parser.add_argument("--base", action="store_true", help="Tester spécifiquement le modèle de BASE (facebook/mms-tts-mlg)")
    parser.add_argument("--sr", "--sample-rate", type=int, default=16000, help="Fréquence d'échantillonnage de sortie (défaut: 16000 Hz natif du modèle, ou 22050, 24000)")
    parser.add_argument("--speed", type=float, default=1.0, help="Vitesse d'élocution (ex: 1.0 ou 1.15, défaut: 1.0)")
    parser.add_argument("--noise-scale", type=float, default=0.55, help="Échelle de bruit (défaut naturel: 0.55, plus bas = plus doux, plus haut = plus dynamique)")
    parser.add_argument("--compare", action="store_true", help="Comparer le modèle de base et le modèle fine-tuné côte-à-côte")
    parser.add_argument("--no-play", action="store_true", help="Ne pas jouer automatiquement le son")

    args = parser.parse_args()

    if args.base:
        selected_model = DEFAULT_BASE_MODEL
    elif args.model:
        selected_model = Path(args.model)
    else:
        selected_model = find_best_model_path()

    if args.compare:
        test_text = args.text or "Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida."
        compare_models(test_text)
    elif args.text:
        generate_voice(args.text, selected_model, "output_test.wav", play_audio=not args.no_play, speed=args.speed, noise_scale=args.noise_scale, target_sr=args.sr)
    else:
        interactive_mode(selected_model, speed=args.speed, noise_scale=args.noise_scale)


if __name__ == "__main__":
    main()
