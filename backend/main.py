from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import sentencepiece as spm
import torch
import json
import uuid
import datetime
import glob
import subprocess
import shutil
import numpy as np
import scipy.io.wavfile
import scipy.signal
import re
import librosa
import tempfile

# Add utils to path for MMS transcription
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))
import mms_transcribeCPU as mms_stt

app = FastAPI(title="Merina to Betsileo Multi-Model Translator API")

# Allow Next.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "transcriptions")
TTS_OUTPUT_DIR = os.path.join(BASE_DIR, "tts_output")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SUBTITLES_DIR = os.path.join(BASE_DIR, "subtitles")
TTS_CACHE_DIR = os.path.join(BASE_DIR, "../utils/modele")
TTS_MODEL_ID = "facebook/mms-tts-mlg"

for d in [TRANSCRIPTIONS_DIR, TTS_OUTPUT_DIR, UPLOADS_DIR, SUBTITLES_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve generated audio and subtitle files statically
app.mount("/tts_audio", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts_audio")
app.mount("/subtitles", StaticFiles(directory=SUBTITLES_DIR), name="subtitles")

# ── Translation models ────────────────────────────────────────────────────────
class TranslationRequest(BaseModel):
    text: str
    model: str = "model1"  # "model1", "model1_2", "model2" or "model3"
    direction: str = "merina-to-betsileo"

models = {}
tokenizers = {}

def load_model(name, path):
    print(f"Loading {name} from {path}...")
    try:
        if os.path.exists(path):
            tokenizers[name] = AutoTokenizer.from_pretrained(path)
            models[name] = AutoModelForSeq2SeqLM.from_pretrained(path)
            print(f"{name} loaded successfully.")
            return True
        else:
            print(f"Warning: {name} path {path} does not exist.")
            return False
    except Exception as e:
        print(f"Warning: Could not load {name} from {path}: {e}")
        return False

# Load all models on startup
load_model("model1", "../merina_to_betsileo")
load_model("model1_2", "../byt5-betsileo-malagasy")
load_model("model2", "../merina_to_betsileo2")
load_model("model3", "../merina-betsileo-new-tokenizer-final")
load_model("vezo", "../merina_to_vezo_final" if os.path.exists("../merina_to_vezo_final") else "../merina_to_vezo")
load_model("fr_en", "../opus_mt_fr_en")
load_model("en_mg", "../opus_mt_en_mg")
load_model("mg_en", "../opus_mt_mg_en")

# ── Translation rules ─────────────────────────────────────────────────────────
rules_path = "dataset/regles_verifiees.json"
rules = {}
inverted_rules = {}
if os.path.exists(rules_path):
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        inverted_rules = {v.lower(): k.lower() for k, v in rules.items() if isinstance(v, str)}
        print(f"Loaded {len(rules)} verified translation rules (and {len(inverted_rules)} inverted).")
    except Exception as e:
        print(f"Error loading translation rules: {e}")

# ── TTS lazy-loaded model ─────────────────────────────────────────────────────
_tts_model = None
_tts_tokenizer = None

# ---- TTS2 (kokoro) ----
_tts2_model = None
_tts2_tokenizer = None

def get_tts_model():
    """Charge le modèle MMS pour la synthèse vocale Malagasy (target_lang='mlg') depuis le dossier local mms_model."""
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        local_mms_path = "/home/tovo/Bureau/crappingSianaka/mms_model"
        if not os.path.exists(local_mms_path):
            local_mms_path = os.path.abspath(os.path.join(BASE_DIR, "../mms_model"))

        if os.path.exists(local_mms_path):
            print(f"[INFO] Chargement du modèle TTS Malgache MMS depuis l'emplacement local : {local_mms_path} (target_lang='mlg')...")
            try:
                _tts_tokenizer = AutoTokenizer.from_pretrained(local_mms_path, target_lang="mlg")
                _tts_model = VitsModel.from_pretrained(local_mms_path, target_lang="mlg")
            except Exception as e_local:
                print(f"[WARNING] Utilisation du fallback TTS mms-tts-mlg suite à : {e_local}")
                tts_fallback = "/home/tovo/Bureau/crappingSianaka/utils/modele/models--facebook--mms-tts-mlg/snapshots/315fb13a7db845580c6fa7f0e3b55d617542a2ef"
                if os.path.exists(tts_fallback):
                    _tts_tokenizer = AutoTokenizer.from_pretrained(tts_fallback)
                    _tts_model = VitsModel.from_pretrained(tts_fallback)
                else:
                    _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID, cache_dir=TTS_CACHE_DIR)
                    _tts_model = VitsModel.from_pretrained(TTS_MODEL_ID, cache_dir=TTS_CACHE_DIR)
        else:
            print(f"[INFO] Chargement du modèle TTS {TTS_MODEL_ID} depuis le cache...")
            _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID, cache_dir=TTS_CACHE_DIR)
            _tts_model = VitsModel.from_pretrained(TTS_MODEL_ID, cache_dir=TTS_CACHE_DIR)

        _tts_model.eval()
        print("[INFO] Modèle TTS Malgache MMS (mlg) chargé avec succès.")
    return _tts_model, _tts_tokenizer

# backend/main.py – fonction get_tts2_model (à remplacer)

def get_tts2_model():
    """Lazy‑load the Kokoro ONNX TTS model."""
    global _tts2_model
    if _tts2_model is None:
        # Chemin correct : ../MalagsyTTS (pas .../kokoro)
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "MalagsyTTS")
        )
        # Recherche des fichiers .onnx et .bin
        onnx_path = next(
            (os.path.join(base_dir, n) for n in ("kokoro-v1.0.onnx",
                                                 "kokoro-v0_19.onnx",
                                                 "kokoro.onnx")
             if os.path.isfile(os.path.join(base_dir, n))),
            None,
        )
        bin_path = next(
            (os.path.join(base_dir, n) for n in ("voices-v1.0.bin",
                                                 "voices-v0_19.bin",
                                                 "voices.bin")
             if os.path.isfile(os.path.join(base_dir, n))),
            None,
        )
        if not onnx_path or not bin_path:
            raise FileNotFoundError(
                f"kokoro TTS model not found at {base_dir}. "
                "Expected .onnx and .bin files."
            )
        from kokoro_onnx import Kokoro
        print(f"[INFO] Loading Kokoro model: {onnx_path} + {bin_path}")
        _tts2_model = Kokoro(onnx_path, bin_path)
    return _tts2_model, None

# ---- Vezo MMS-TTS (mms-tts-vezo-finetuned) ----
_vezo_tts_model = None
_vezo_tts_tokenizer = None
VEZO_TTS_PATH = "/home/tovo/Bureau/scrap/models/mms-tts-vezo-finetuned-v4"

VALID_VEZO_CHARS = set("abcdefghijklmnopqrstuvwxyzàâéèêëìîïòôùûü '|-")

def get_vezo_tts_model():
    """Lazy-load the fine-tuned Vezo MMS-TTS model."""
    global _vezo_tts_model, _vezo_tts_tokenizer
    if _vezo_tts_model is None:
        if not os.path.exists(VEZO_TTS_PATH):
            raise RuntimeError(f"Vezo TTS model not found at {VEZO_TTS_PATH}")
        print(f"[INFO] Loading fine-tuned Vezo TTS model from {VEZO_TTS_PATH}...")
        from transformers import AutoTokenizer, AutoModelForTextToWaveform
        _vezo_tts_tokenizer = AutoTokenizer.from_pretrained(VEZO_TTS_PATH)
        _vezo_tts_model = AutoModelForTextToWaveform.from_pretrained(VEZO_TTS_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _vezo_tts_model.to(device)
        _vezo_tts_model.eval()
        print(f"[INFO] Vezo TTS model loaded successfully on {device}.")
    return _vezo_tts_model, _vezo_tts_tokenizer

def normalize_vezo_text(text: str) -> str:
    """Normaliser le texte pour la synthèse vocale MMS-TTS Vezo."""
    text = text.lower()
    num_map = {
        '0': 'folo', '1': 'iray', '2': 'roa', '3': 'telo',
        '4': 'efatra', '5': 'dimy', '6': 'enina', '7': 'fito',
        '8': 'valo', '9': 'sivy',
    }
    for n, w in num_map.items():
        text = text.replace(n, f" {w} ")
    text = re.sub(r'[,;:]+', ' | ', text)
    text = re.sub(r'[.!?]+', ' | ', text)
    text = re.sub(r'[()«»\[\]{}]', '', text)
    text = ''.join(c for c in text if c in VALID_VEZO_CHARS)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'(\| )+', '| ', text)
    return re.sub(r' +', ' ', text).strip().strip('|').strip()

def synthesize_vezo_tts(
    text: str,
    speed: float = 1.15,
    target_sr: int = 22050,
    noise_scale: float = 0.35,
    noise_scale_duration: float = 0.6,
    highpass_cutoff: int = 60,
    noise_gate_threshold: float = 0.01,
    peak_norm: float = 0.95,
    warmth_level: float = 0.5
) -> dict:
    """Générer un fichier audio WAV en dialecte Vezo avec le modèle fine-tuné et filtrage anti-bruit & anti-son métallique."""
    if not text or not text.strip():
        return None

    clean_text = normalize_vezo_text(text)
    if not clean_text:
        clean_text = text.lower()

    tts_id = str(uuid.uuid4())
    wav_filename = f"{tts_id}_vezo.wav"
    wav_path = os.path.join(TTS_OUTPUT_DIR, wav_filename)

    model, tokenizer = get_vezo_tts_model()
    device = next(model.parameters()).device

    # Ajuster la vitesse d'élocution
    model.speaking_rate = speed

    inputs = tokenizer(clean_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(
            **inputs,
            noise_scale=noise_scale,
            noise_scale_duration=noise_scale_duration
        ).waveform

    waveform = output.squeeze().cpu().numpy()

    # 1. Filtre passe-haut (High-pass filter) pour supprimer les ronflements et bruits très basses fréquences
    if highpass_cutoff > 0:
        try:
            sos = scipy.signal.butter(4, float(highpass_cutoff), btype='highpass', fs=target_sr, output='sos')
            waveform = scipy.signal.sosfilt(sos, waveform)
        except Exception as e_hp:
            print(f"[WARNING] Highpass filter error: {e_hp}")

    # 2. Noise Gate (Atténuation du souffle et bruit de fond dans les silences)
    if noise_gate_threshold > 0:
        try:
            import scipy.ndimage
            env = scipy.ndimage.gaussian_filter1d(np.abs(waveform), sigma=max(1, int(target_sr * 0.01)))
            gate = np.clip(env / float(noise_gate_threshold), 0.0, 1.0)
            waveform = waveform * gate
        except Exception as e_ng:
            print(f"[WARNING] Noise gate error: {e_ng}")

    # 3. Adoucisseur & Chaleur Vocale (Elimine le son métallique et rend la voix plus naturelle)
    if warmth_level > 0.0:
        try:
            cutoff_lp = max(4000.0, min(float(target_sr) / 2.0 - 500.0, 11000.0 - warmth_level * 3000.0))
            sos_lp = scipy.signal.butter(2, cutoff_lp, btype='lowpass', fs=target_sr, output='sos')
            waveform = scipy.signal.sosfilt(sos_lp, waveform)

            w_norm = min(0.45, 350.0 / (float(target_sr) / 2.0))
            b_w, a_w = scipy.signal.iirpeak(w_norm, Q=1.5)
            body = scipy.signal.lfilter(b_w, a_w, waveform) * (0.12 * warmth_level)
            waveform = waveform + body
        except Exception as e_warm:
            print(f"[WARNING] Warmth filter error: {e_warm}")

    # 4. Normalisation crête (Peak Normalization)
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-5:
        waveform = waveform * (peak_norm / max_val)

    audio_int16 = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(wav_path, rate=target_sr, data=audio_int16)

    return {
        "id": tts_id,
        "audio_url": f"/tts_audio/{wav_filename}",
        "text": text,
        "clean_text": clean_text,
        "sr": target_sr,
        "speed": speed,
        "noise_scale": noise_scale,
        "noise_scale_duration": noise_scale_duration,
        "highpass_cutoff": highpass_cutoff,
        "noise_gate_threshold": noise_gate_threshold,
        "peak_norm": peak_norm,
        "warmth_level": warmth_level
    }

def synthesize_merina_tts(
    text: str,
    speed: float = 1.0,
    target_sr: int = 22050,
    highpass_cutoff: int = 60
) -> dict:
    """Générer un fichier audio WAV en Malagasy Officiel (Merina) via MMS-TTS-MLG."""
    if not text or not text.strip():
        return None
    tts_id = str(uuid.uuid4())
    wav_filename = f"{tts_id}_merina.wav"
    wav_path = os.path.join(TTS_OUTPUT_DIR, wav_filename)

    model, tokenizer = get_tts_model()
    original_rate = model.config.speaking_rate
    model.config.speaking_rate = original_rate * speed

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    audio_np = output.squeeze().cpu().numpy()

    if highpass_cutoff > 0:
        try:
            sos = scipy.signal.butter(4, float(highpass_cutoff), btype='highpass', fs=target_sr, output='sos')
            audio_np = scipy.signal.sosfilt(sos, audio_np)
        except Exception:
            pass

    max_val = np.max(np.abs(audio_np))
    if max_val > 1e-5:
        audio_np = audio_np * (0.95 / max_val)

    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(wav_path, target_sr, audio_int16)
    model.config.speaking_rate = original_rate

    return {
        "id": tts_id,
        "audio_url": f"/tts_audio/{wav_filename}",
        "text": text,
        "sr": target_sr,
        "speed": speed,
        "highpass_cutoff": highpass_cutoff
    }

def synthesize_english_tts(text: str, speed: float = 1.0) -> dict:
    """Générer un fichier audio WAV en Anglais avec Kokoro TTS."""
    if not text or not text.strip():
        return None
    tts_id = str(uuid.uuid4())
    wav_filename = f"{tts_id}_english.wav"
    wav_path = os.path.join(TTS_OUTPUT_DIR, wav_filename)
    try:
        kokoro, _ = get_tts2_model()
        samples, sample_rate = kokoro.create(text, voice='af_sarah', speed=speed, lang="en-us")
        audio_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(wav_path, sample_rate, audio_int16)
        return {
            "id": tts_id,
            "audio_url": f"/tts_audio/{wav_filename}",
            "text": text,
            "speed": speed
        }
    except Exception as e:
        print(f"[WARNING] English TTS error: {e}")
        return None

def load_nllb_model():
    """Lazy-load Meta NLLB-200 (600M) for high-accuracy Malagasy <-> English translation."""
    global models, tokenizers
    if "nllb" not in models:
        nllb_id = "facebook/nllb-200-distilled-600M"
        local_nllb_path = os.path.abspath(os.path.join(BASE_DIR, "../nllb-200-distilled-600M"))
        path_to_load = local_nllb_path if os.path.exists(local_nllb_path) else nllb_id
        try:
            print(f"[INFO] Loading NLLB-200 model from {path_to_load}...")
            tokenizers["nllb"] = AutoTokenizer.from_pretrained(path_to_load, src_lang="plt_Latn")
            models["nllb"] = AutoModelForSeq2SeqLM.from_pretrained(path_to_load)
            print("[INFO] NLLB-200 model loaded successfully for Malagasy <-> English.")
            return True
        except Exception as e:
            print(f"[WARNING] Could not load NLLB-200: {e}")
            return False
    return True

# ── Helper: run dialect translation ──────────────────────────────────────────
def run_translation(text: str, model_key: str = "model1", direction: str = "merina-to-betsileo") -> str:
    """Run dialect translation using loaded models."""
    if direction in ["merina-to-english", "malagasy-to-english", "vezo-to-english"]:
        merina_text = text
        if direction == "vezo-to-english":
            merina_text = run_translation(text, model_key="vezo", direction="vezo-to-merina")

        # 1. Utiliser Meta NLLB-200 (Modèle Open-Source SOTA pour le Malgache vers l'Anglais)
        try:
            if load_nllb_model():
                tokenizer = tokenizers["nllb"]
                model = models["nllb"]
                inputs_mg = tokenizer(merina_text, return_tensors="pt", padding=True, truncation=True)
                try:
                    eng_code_id = tokenizer.convert_tokens_to_ids("eng_Latn")
                except Exception:
                    eng_code_id = None

                with torch.no_grad():
                    if eng_code_id is not None and isinstance(eng_code_id, int) and eng_code_id > 0:
                        out_en = model.generate(**inputs_mg, forced_bos_token_id=eng_code_id, max_length=128)
                    else:
                        out_en = model.generate(**inputs_mg, max_length=128)
                res_nllb = tokenizer.decode(out_en[0], skip_special_tokens=True).strip()
                if res_nllb:
                    return res_nllb
        except Exception as e_nllb:
            print(f"[WARNING] NLLB-200 translation failed, falling back to opus_mt_mg_en: {e_nllb}")

        # 2. Fallback opus_mt_mg_en
        if "mg_en" in models:
            inputs_mg = tokenizers["mg_en"](merina_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out_en = models["mg_en"].generate(**inputs_mg, max_length=128)
            return tokenizers["mg_en"].decode(out_en[0], skip_special_tokens=True)
        return merina_text

    if direction in ["english-to-vezo", "english-to-merina", "english-to-betsileo"]:
        # Step 1: EN -> Merina (Malagasy)
        if "en_mg" in models:
            inputs_en = tokenizers["en_mg"](text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out_mg = models["en_mg"].generate(**inputs_en, max_length=128)
            merina_text = tokenizers["en_mg"].decode(out_mg[0], skip_special_tokens=True)
        else:
            merina_text = text

        if direction == "english-to-merina":
            return merina_text
        elif direction == "english-to-betsileo":
            return run_translation(merina_text, model_key="model3", direction="merina-to-betsileo")
        else: # english-to-vezo
            return run_translation(merina_text, model_key="vezo", direction="merina-to-vezo")

    if direction == "merina-to-vezo" or model_key == "vezo":
        fallback_order = [model_key, "vezo", "model3", "model2", "model1"]
    else:
        fallback_order = [model_key, "model1_2", "model1", "model2", "model3", "vezo"]
    selected = next((m for m in fallback_order if m in models), None)
    if not selected:
        return text  # No model available, return as-is

    tokenizer = tokenizers[selected]
    model = models[selected]
    if selected == "model1_2":
        dir_prompt = "translate Malagasy to Betsileo" if direction == "merina-to-betsileo" else "translate Betsileo to Malagasy"
        input_text = f"{dir_prompt}: {text}"
    else:
        input_text = text.lower() if selected in ["model3", "vezo"] else text

    # Single-word dictionary fast-path (only for Betsileo)
    if direction in ["merina-to-betsileo", "betsileo-to-merina"]:
        clean_text = input_text.strip(".,!?;:()\"' ")
        words = clean_text.split()
        active_rules = rules if direction == "merina-to-betsileo" else inverted_rules

        if len(words) == 1:
            single_word = words[0].lower()
            if single_word in active_rules:
                return active_rules[single_word]

    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── Helper: Whisper STT lazy loader ──────────────────────────────────────────
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[INFO] Loading Whisper small model for Speech-to-Text...")
        _whisper_model = whisper.load_model("small")
        print("[INFO] Whisper model loaded successfully.")
    return _whisper_model

# ── Helper: extract audio from video ─────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

def extract_audio_if_video(input_path: str, output_wav: str) -> str:
    """If input is a video, extract audio to output_wav. Otherwise copy as-is."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", output_wav],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        return output_wav
    else:
        # For audio files, still convert to ensure 16kHz mono WAV
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_wav],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # If ffmpeg fails (e.g. already wav), just copy
            shutil.copy(input_path, output_wav)
        return output_wav

# ════════════════════════════════════════════════════════════════════════════
#  EXISTING ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "fastapi_vezo_backend",
        "models_available": list(models.keys()),
        "endpoints": ["/translate", "/tts-vezo", "/tts", "/speech-to-vezo"],
        "model1_loaded": "model1" in models,
        "model1_2_loaded": "model1_2" in models,
        "model2_loaded": "model2" in models,
        "model3_loaded": "model3" in models,
        "vezo_loaded": "vezo" in models,
    }

@app.post("/translate")
async def translate(req: TranslationRequest):
    if req.direction in ["english-to-vezo", "english-to-merina", "english-to-betsileo"]:
        translated = run_translation(req.text, model_key=req.model, direction=req.direction)
        return {
            "translation": translated,
            "model_used": "english_pivot_pipeline",
            "tokens": req.text.split()
        }

    if req.direction == "merina-to-vezo" or req.model == "vezo":
        fallback_order = [req.model, "vezo", "model3", "model2", "model1"]
    else:
        fallback_order = [req.model, "model1_2", "model1", "model2", "model3", "vezo"]
    selected_model = next((m for m in fallback_order if m in models), None)

    if not selected_model:
        return {"translation": f"[Aucun modèle disponible pour: {req.text}]"}

    tokenizer = tokenizers[selected_model]
    model = models[selected_model]

    if selected_model == "model1_2":
        dir_prompt = "translate Malagasy to Betsileo" if req.direction == "merina-to-betsileo" else "translate Betsileo to Malagasy"
        text = f"{dir_prompt}: {req.text}"
    else:
        # Le modèle fine-tuné requiert un texte en minuscules
        text = req.text.lower() if selected_model in ["model3", "vezo"] else req.text

    # 1. Optimisation pour mot unique (Bypass Dictionnaire direct pour Betsileo)
    if req.direction in ["merina-to-betsileo", "betsileo-to-merina"]:
        clean_text = text.strip(".,!?;:()\"' ")
        words = clean_text.split()
        active_rules = rules if req.direction == "merina-to-betsileo" else inverted_rules

        if len(words) == 1:
            single_word = words[0].lower()
            if single_word in active_rules:
                translated_word = active_rules[single_word]
                # Conserver la capitalisation (ex: Asia -> Azia)
                if words[0][0].isupper():
                    translated_word = translated_word.capitalize()
                # Conserver la ponctuation environnante
                start_idx = text.lower().find(words[0].lower())
                if start_idx != -1:
                    prefix = text[:start_idx]
                    suffix = text[start_idx + len(words[0]):]
                    final_translation = prefix + translated_word + suffix
                else:
                    final_translation = translated_word

                return {
                    "translation": final_translation,
                    "model_used": "dictionnaire_verifie",
                    "tokens": [words[0]],
                    "generation_details": [{
                        "token": translated_word,
                        "prob": 100.0,
                        "alternatives": [{"token": translated_word, "prob": 100.0}]
                    }],
                    "constraints_applied": True,
                    "forced_words": [translated_word]
                }

    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

    # Extraire les tokens d'entrée
    input_ids = inputs["input_ids"][0].tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Détection des contraintes lexicales (mots forcés du dictionnaire pour Betsileo)
    words = text.lower().split()
    forced_words = []
    if req.direction in ["merina-to-betsileo", "betsileo-to-merina"]:
        active_rules = rules if req.direction == "merina-to-betsileo" else inverted_rules
        for w in words:
            clean_w = w.strip(".,!?;:()\"'")
            if clean_w in active_rules:
                target_word = active_rules[clean_w]
                forced_words.append(target_word)

    # Éliminer les doublons tout en gardant l'ordre
    unique_forced_words = []
    for word in forced_words:
        if word not in unique_forced_words:
            unique_forced_words.append(word)

    force_words_ids = []
    for word in unique_forced_words[:5]: # limite de 5 contraintes max par précaution
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if ids:
            force_words_ids.append(ids)

    # Inférence
    if force_words_ids:
        print(f"Applying lexical constraints: forcing target words {unique_forced_words} (IDs: {force_words_ids})")
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            force_words_ids=[force_words_ids],
            return_dict_in_generate=True,
            output_scores=True
        )
    else:
        # Génération greedy avec scores pour récupérer les probabilités exactes
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    translated_ids = outputs.sequences[0].tolist()
    translated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Calculer les probabilités des tokens choisis et de leurs alternatives
    generation_details = []

    # On ne calcule les probabilités pas-à-pas que si les dimensions s'alignent parfaitement (généralement en recherche gloutonne num_beams=1)
    if hasattr(outputs, "scores") and outputs.scores and len(outputs.scores) == len(translated_ids) - 1:
        for i, logits in enumerate(outputs.scores):
            step_logits = logits[0]
            probs = torch.softmax(step_logits, dim=-1)

            chosen_token_id = translated_ids[i + 1]
            chosen_token = tokenizer.convert_ids_to_tokens(chosen_token_id)

            # On ignore les tokens spéciaux de fin ou de padding pour l'affichage propre
            if chosen_token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                continue

            chosen_prob = probs[chosen_token_id].item()

            # Obtenir les alternatives top 3 pour cette étape
            top_probs, top_indices = torch.topk(probs, k=3)
            alternatives = []
            for prob_val, idx_val in zip(top_probs.tolist(), top_indices.tolist()):
                alt_token = tokenizer.convert_ids_to_tokens(idx_val)
                alternatives.append({
                    "token": alt_token,
                    "prob": round(prob_val * 100, 1)
                })

            generation_details.append({
                "token": chosen_token,
                "prob": round(chosen_prob * 100, 1),
                "alternatives": alternatives
            })

    return {
        "translation": translated_text,
        "model_used": selected_model,
        "tokens": input_tokens,
        "generation_details": generation_details,
        "constraints_applied": bool(force_words_ids),
        "forced_words": unique_forced_words
    }

# --- TOKENIZER ENDPOINT ---
class TokenizeRequest(BaseModel):
    text: str

sp = spm.SentencePieceProcessor()
sp_model_path = "../marian_malagasy.model"

@app.post("/tokenize")
async def tokenize_text(req: TokenizeRequest):
    try:
        # Load dynamically in case it was just created/updated
        sp.load(sp_model_path)
        tokens = sp.encode_as_pieces(req.text)
        return {"tokens": tokens}
    except Exception as e:
        return {"tokens": [f"[Erreur de chargement du modèle: {e}]"]}


# ════════════════════════════════════════════════════════════════════════════
#  STT — TRANSCRIPTION (Audio / Video → Texte + Traduction dialectique)
# ════════════════════════════════════════════════════════════════════════════

def dedup_consecutive_words(text: str) -> str:
    """Supprime les mots répétés consécutifs dans une phrase."""
    words = text.split()
    if not words:
        return text
    result = [words[0]]
    for word in words[1:]:
        if word.lower() != result[-1].lower():
            result.append(word)
    return " ".join(result)

def merge_short_segments(segments: list, min_words: int = 4) -> list:
    """Fusionne les segments consécutifs dont le texte est très court (< min_words mots).
    Les deux segments fusionnés sont séparés par une virgule, et le résultat se termine par un point-virgule.
    Ex: ["Eny", "tsara ihany"] -> ["Eny, tsara ihany;"]
    """
    merged = []
    i = 0
    while i < len(segments):
        seg = dict(segments[i])
        # Si le segment actuel est court ET qu'il y a un suivant
        if len(seg["text"].split()) < min_words and i + 1 < len(segments):
            next_seg = segments[i + 1]
            # Fusionner : prendre le start du premier, le end du second
            combined_text = seg["text"].rstrip(".;,") + ", " + next_seg["text"].rstrip(".;,") + ";"
            merged.append({
                "start": seg["start"],
                "end": next_seg["end"],
                "text": combined_text
            })
            i += 2  # Sauter le segment suivant car il est déjà fusionné
        else:
            merged.append(seg)
            i += 1
    return merged

@app.post("/transcribe")
async def transcribe_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("model1"),
    direction: str = Form("merina-to-betsileo"),
    dubbing: bool = Form(False),
):
    """
    Upload un fichier audio (.mp3, .wav) ou vidéo (.mp4, .mkv, .avi, .mov).
    1. Si vidéo → extraire l'audio avec ffmpeg
    2. Transcrire avec MMS (Malgache)
    3. Appliquer la conversion dialectique
    4. Sauvegarder et retourner le résultat
    """
    job_id = str(uuid.uuid4())
    original_filename = file.filename or "upload"
    ext = os.path.splitext(original_filename)[1].lower()

    # Sauvegarder le fichier uploadé
    upload_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")
    wav_path = os.path.join(UPLOADS_DIR, f"{job_id}.wav")

    # Helper pour le format VTT
    def format_vtt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extraction / conversion audio
        extract_audio_if_video(upload_path, wav_path)

        # Transcription MMS avec timestamps
        print(f"[STT] Transcription de {original_filename}...")
        segments = mms_stt.transcribe_with_timestamps(wav_path, lang="mlg")
        if not segments:
            raise HTTPException(status_code=500, detail="La transcription MMS a échoué.")

        # Traduction dialectique segment par segment et génération VTT
        print(f"[STT] Traduction dialectique ({direction})...")
        segments = merge_short_segments(segments, min_words=4)
        vtt_content = ["WEBVTT\n"]
        full_transcription = []
        full_translation = []
        dubbed_segments = []
        
        for i, seg in enumerate(segments):
            start_str = format_vtt_time(seg["start"])
            end_str = format_vtt_time(seg["end"])
            
            # Traduire le texte du segment et dédupliquer
            translated_text = run_translation(seg["text"], model, direction=direction)
            translated_text = dedup_consecutive_words(translated_text)
            
            full_transcription.append(seg["text"])
            full_translation.append(translated_text)
            
            vtt_content.append(f"{start_str} --> {end_str}")
            vtt_content.append(translated_text)
            vtt_content.append("") # Ligne vide pour séparer
            
        transcription_final = " ".join(full_transcription)
        translation_final = " ".join(full_translation)

        # Sauvegarder le fichier VTT
        vtt_filename = f"{job_id}.vtt"
        vtt_path = os.path.join(SUBTITLES_DIR, vtt_filename)
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vtt_content))
            
        vtt_url = f"/subtitles/{vtt_filename}"

        # Sauvegarder
        record = {
            "id": job_id,
            "filename": original_filename,
            "transcription": transcription_final,
            "translation": translation_final,
            "model_used": model,
            "direction": direction,
            "vtt_url": vtt_url,
            "dubbed_url": None,
            "original_dubbed_url": None,
            "created_at": datetime.datetime.now().isoformat(),
        }
        json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            
        # Nettoyage programmé plus tard car le background task pourrait en avoir besoin
        if not dubbing:
            for p in [upload_path, wav_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        
        if dubbing:
            print(f"[STT] Lancement de la tâche de doublage en arrière-plan pour {job_id}...")
            # Copie des données nécessaires pour la tâche asynchrone
            segments_data = []
            for seg, trans, orig in zip(segments, full_translation, full_transcription):
                segments_data.append({
                    "start": seg["start"], 
                    "end": seg["end"],
                    "translated_text": trans,
                    "original_text": orig
                })
            
            background_tasks.add_task(
                generate_dubbing_task,
                job_id=job_id,
                segments_data=segments_data,
                wav_path=wav_path,
                upload_path=upload_path,
                ext=ext
            )
            
        return {
            "id": job_id,
            "transcription": transcription_final,
            "translation": translation_final,
            "model_used": model,
            "direction": direction,
            "vtt_url": vtt_url,
            "dubbed_url": None,
            "original_dubbed_url": None,
            "is_dubbing": dubbing
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_dubbing_task(job_id: str, segments_data: list, wav_path: str, upload_path: str, ext: str):
    """Tâche d'arrière-plan pour générer l'audio TTS (traduction + original) et les muxer."""
    print(f"[DUBBING] Démarrage de la tâche de doublage pour {job_id}...")
    
    dubbed_translated_segs = []  # TTS depuis la traduction
    dubbed_original_segs = []    # TTS depuis la transcription originale
    
    tts_model, tts_tokenizer = get_tts_model()
    target_sr = tts_model.config.sampling_rate
    
    for seg in segments_data:
        # --- Doublage TRADUCTION ---
        translated_text = seg.get("translated_text", "").strip()
        if translated_text:
            if not translated_text.rstrip().endswith((".", "!", "?", ";")):
                translated_text = translated_text.rstrip() + ";"
            try:
                inputs = tts_tokenizer(translated_text, return_tensors="pt")
                with torch.no_grad():
                    audio_np = tts_model(**inputs).waveform.squeeze().cpu().numpy()
                dubbed_translated_segs.append({"start": seg["start"], "audio": audio_np, "sr": target_sr})
            except Exception as e:
                print(f"[DUBBING] Erreur TTS traduction: {e}")

        # --- Doublage ORIGINAL ---
        original_text = seg.get("original_text", "").strip()
        if original_text:
            if not original_text.rstrip().endswith((".", "!", "?", ";")):
                original_text = original_text.rstrip() + ";"
            try:
                inputs = tts_tokenizer(original_text, return_tensors="pt")
                with torch.no_grad():
                    audio_np = tts_model(**inputs).waveform.squeeze().cpu().numpy()
                dubbed_original_segs.append({"start": seg["start"], "audio": audio_np, "sr": target_sr})
            except Exception as e:
                print(f"[DUBBING] Erreur TTS original: {e}")
    
    total_duration = librosa.get_duration(path=wav_path)

    def build_dubbed_video(segs, suffix, label):
        if not segs:
            return None
        try:
            full_audio = np.zeros(int(total_duration * target_sr) + 1, dtype=np.float32)
            for seg_data in segs:
                start_sample = int(seg_data["start"] * target_sr)
                audio_data = seg_data["audio"]
                end_sample = start_sample + len(audio_data)
                if end_sample > len(full_audio):
                    audio_data = audio_data[:len(full_audio)-start_sample]
                    end_sample = len(full_audio)
                full_audio[start_sample:end_sample] = audio_data

            dubbed_audio_path = os.path.join(TTS_OUTPUT_DIR, f"{job_id}_{suffix}.wav")
            scipy.io.wavfile.write(dubbed_audio_path, target_sr, full_audio)

            if ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
                dubbed_video_path = os.path.join(TTS_OUTPUT_DIR, f"{job_id}_{suffix}{ext}")
                cmd = [
                    "ffmpeg", "-y", "-i", upload_path, "-i", dubbed_audio_path,
                    "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", dubbed_video_path
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    url = f"/tts_audio/{job_id}_{suffix}{ext}"
                else:
                    print(f"[DUBBING] Erreur ffmpeg {label}: {result.stderr}")
                    url = f"/tts_audio/{job_id}_{suffix}.wav"
            else:
                url = f"/tts_audio/{job_id}_{suffix}.wav"
            print(f"[DUBBING] {label} terminé : {url}")
            return url
        except Exception as e:
            print(f"[DUBBING] Erreur assemblage {label}: {e}")
            return None

    dubbed_url = build_dubbed_video(dubbed_translated_segs, "dubbed", "Doublage traduction")
    original_dubbed_url = build_dubbed_video(dubbed_original_segs, "orig_dubbed", "Doublage original")

    # Mettre à jour le fichier JSON
    json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{job_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["dubbed_url"] = dubbed_url
        record["original_dubbed_url"] = original_dubbed_url
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # Nettoyage
    for p in [upload_path, wav_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


@app.get("/transcriptions")
async def list_transcriptions():
    """Liste toutes les transcriptions sauvegardées."""
    records = []
    for json_file in sorted(glob.glob(os.path.join(TRANSCRIPTIONS_DIR, "*.json")), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            pass
    return records


@app.delete("/transcriptions/{id}")
async def delete_transcription(id: str):
    """Supprime une transcription par ID (incluant son fichier VTT éventuel)."""
    json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{id}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
        
    os.remove(json_path)
    
    vtt_path = os.path.join(SUBTITLES_DIR, f"{id}.vtt")
    if os.path.exists(vtt_path):
        os.remove(vtt_path)
        
    return {"success": True, "id": id}


# ════════════════════════════════════════════════════════════════════════════
#  TTS — SYNTHÈSE VOCALE (Texte → Audio WAV)
# ════════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0  # speaking rate multiplier
    sr: int = 22050
    highpass_cutoff: int = 60

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convertit un texte Malgache Officiel en audio WAV avec facebook/mms-tts-mlg.
    Retourne l'URL du fichier audio généré.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte est vide.")

    try:
        res = synthesize_merina_tts(
            req.text,
            speed=req.speed,
            target_sr=req.sr,
            highpass_cutoff=req.highpass_cutoff
        )
        return res
    except Exception as e:
        print(f"Error in /tts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        # Sauvegarder métadonnées
        record = {
            "id": tts_id,
            "text": req.text,
            "speed": req.speed,
            "audio_file": wav_filename,
            "created_at": datetime.datetime.now().isoformat(),
        }
        meta_path = os.path.join(TTS_OUTPUT_DIR, f"{tts_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return {
            "id": tts_id,
            "audio_url": f"/tts_audio/{wav_filename}",
            "text": req.text,
        }

    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=500, detail=str(e))

# ---- New TTS2 endpoint using kokoro model ----
@app.post("/tts2")
async def text_to_speech2(req: TTSRequest):
    """Convertit du texte en audio avec le modèle kokoro (TTS2).
    Le modèle doit être présent dans ../MalagsyTTS/kokoro.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte est vide.")

    tts_id = str(uuid.uuid4())
    wav_filename = f"{tts_id}.wav"
    wav_path = os.path.join(TTS_OUTPUT_DIR, wav_filename)

    try:
        kokoro, _ = get_tts2_model()
        audio, sr = kokoro.create(req.text, voice="af_heart", speed=float(req.speed))

        import soundfile as sf
        sf.write(wav_path, audio, sr)

        record = {
            "id": tts_id,
            "model": "kokoro",
            "text": req.text,
            "speed": req.speed,
            "audio_file": wav_filename,
            "created_at": datetime.datetime.now().isoformat(),
        }
        meta_path = os.path.join(TTS_OUTPUT_DIR, f"{tts_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return {
            "id": tts_id,
            "audio_url": f"/tts_audio/{wav_filename}",
            "text": req.text,
            "model": "kokoro",
        }

    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts_list")
async def list_tts():
    """Liste tous les audios TTS générés."""
    records = []
    for meta_file in sorted(glob.glob(os.path.join(TTS_OUTPUT_DIR, "*.json")), reverse=True):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["audio_url"] = f"/tts_audio/{data['audio_file']}"
                records.append(data)
        except Exception:
            pass
    return records


@app.delete("/tts/{id}")
async def delete_tts(id: str):
    """Supprime un audio TTS par ID."""
    meta_path = os.path.join(TTS_OUTPUT_DIR, f"{id}.json")
    wav_path_pattern = os.path.join(TTS_OUTPUT_DIR, f"{id}.wav")

    deleted = False
    for p in [meta_path, wav_path_pattern]:
        if os.path.exists(p):
            os.remove(p)
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Audio TTS introuvable.")
    return {"success": True, "id": id}




# ════════════════════════════════════════════════════════════════════════════
#  KOKORO DUBBING — Doublage vidéo avec Kokoro TTS (phonèmes éditables)
# ════════════════════════════════════════════════════════════════════════════

class KokoroDubRequest(BaseModel):
    job_id: str

class KokoroRegenRequest(BaseModel):
    job_id: str
    phonemes: list  # list of { segment_idx, text, phonemes, start, end }

class KokoroManualRequest(BaseModel):
    job_id: str
    manual_phoneme: str  # Texte phonétique saisi manuellement par l'utilisateur


def _build_kokoro_video(job_id: str, segments_data: list, wav_path: str, upload_path: str, ext: str):
    """Génère l'audio Kokoro, assemble la vidéo et retourne les URLs."""
    import soundfile as sf
    kokoro, _ = get_tts2_model()
    target_sr = 24000  # Kokoro native sample rate

    # Durée totale depuis l'audio source
    if os.path.exists(wav_path):
        total_duration = librosa.get_duration(path=wav_path)
    else:
        total_duration = max((s["end"] for s in segments_data), default=10.0) + 1.0

    full_audio = np.zeros(int(total_duration * target_sr) + target_sr, dtype=np.float32)

    phoneme_segments = []
    for seg in segments_data:
        text = (seg.get("phonemes") or seg.get("translated_text") or seg.get("text", "")).strip()
        if not text:
            phoneme_segments.append({
                "segment_idx": seg.get("segment_idx", 0),
                "text": seg.get("text", ""),
                "phonemes": "",
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })
            continue

        try:
            audio_seg, _ = kokoro.create(text, voice="af_heart", speed=1.0)
            start_sample = int(seg["start"] * target_sr)
            end_sample = start_sample + len(audio_seg)
            if end_sample > len(full_audio):
                audio_seg = audio_seg[:len(full_audio) - start_sample]
                end_sample = len(full_audio)
            full_audio[start_sample:end_sample] = audio_seg
        except Exception as e:
            print(f"[KOKORO] Erreur TTS pour segment: {e}")

        phoneme_segments.append({
            "segment_idx": seg.get("segment_idx", 0),
            "text": seg.get("text", ""),
            "phonemes": text,
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
        })

    # Écrire le WAV Kokoro
    kokoro_wav = os.path.join(TTS_OUTPUT_DIR, f"{job_id}_kokoro.wav")
    sf.write(kokoro_wav, full_audio, target_sr)

    # Assembler avec la vidéo si le fichier d'origine est une vidéo
    kokoro_url = f"/tts_audio/{job_id}_kokoro.wav"
    if os.path.exists(upload_path) and ext in VIDEO_EXTENSIONS:
        kokoro_video = os.path.join(TTS_OUTPUT_DIR, f"{job_id}_kokoro{ext}")
        cmd = [
            "ffmpeg", "-y", "-i", upload_path, "-i", kokoro_wav,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", kokoro_video
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            kokoro_url = f"/tts_audio/{job_id}_kokoro{ext}"
        else:
            print(f"[KOKORO] ffmpeg error: {result.stderr.decode()}")

    return kokoro_url, phoneme_segments


@app.post("/kokoro_dub")
async def kokoro_dub(req: KokoroDubRequest):
    """Génère un doublage Kokoro pour une transcription existante.
    Retourne l'URL du fichier doublé + les segments de texte éditables.
    """
    json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{req.job_id}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")

    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # Reconstruire les segments depuis le VTT
    vtt_url = record.get("vtt_url", "")
    vtt_path = os.path.join(SUBTITLES_DIR, os.path.basename(vtt_url)) if vtt_url else ""

    segments_data = []
    if os.path.exists(vtt_path):
        # Parser le VTT pour retrouver les timestamps + textes de traduction
        with open(vtt_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        i = 0
        seg_idx = 0
        while i < len(lines):
            line = lines[i].strip()
            if "-->" in line:
                parts = line.split("-->")
                def vtt_to_sec(t):
                    t = t.strip()
                    h, m, s = t.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(s)
                start = vtt_to_sec(parts[0])
                end = vtt_to_sec(parts[1])
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                segments_data.append({
                    "segment_idx": seg_idx,
                    "text": " ".join(text_lines),
                    "phonemes": " ".join(text_lines),
                    "translated_text": " ".join(text_lines),
                    "start": start,
                    "end": end,
                })
                seg_idx += 1
            i += 1
    else:
        # Fallback : utiliser la traduction complète comme un seul segment
        segments_data = [{
            "segment_idx": 0,
            "text": record.get("translation", ""),
            "phonemes": record.get("translation", ""),
            "translated_text": record.get("translation", ""),
            "start": 0.0,
            "end": 5.0,
        }]

    # Trouver le fichier vidéo uploadé
    upload_path = ""
    for ext_try in VIDEO_EXTENSIONS:
        p = os.path.join(UPLOADS_DIR, f"{req.job_id}{ext_try}")
        if os.path.exists(p):
            upload_path = p
            break
    ext = os.path.splitext(upload_path)[1].lower() if upload_path else ".mp4"
    wav_path = os.path.join(UPLOADS_DIR, f"{req.job_id}.wav")

    try:
        kokoro_url, phoneme_segments = _build_kokoro_video(
            req.job_id, segments_data, wav_path, upload_path, ext
        )

        # Mettre à jour le JSON
        record["kokoro_dubbed_url"] = kokoro_url
        record["kokoro_phonemes"] = phoneme_segments
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return {
            "job_id": req.job_id,
            "kokoro_dubbed_url": kokoro_url,
            "phonemes": phoneme_segments,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kokoro_dub_regen")
async def kokoro_dub_regen(req: KokoroRegenRequest):
    """Regénère le doublage Kokoro avec des phonèmes modifiés par l'utilisateur."""
    json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{req.job_id}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")

    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    upload_path = ""
    for ext_try in VIDEO_EXTENSIONS:
        p = os.path.join(UPLOADS_DIR, f"{req.job_id}{ext_try}")
        if os.path.exists(p):
            upload_path = p
            break
    ext = os.path.splitext(upload_path)[1].lower() if upload_path else ".mp4"
    wav_path = os.path.join(UPLOADS_DIR, f"{req.job_id}.wav")

    # Supprimer l'ancien WAV/vidéo Kokoro pour forcer la régénération
    for suffix in [f"{req.job_id}_kokoro.wav", f"{req.job_id}_kokoro{ext}"]:
        p = os.path.join(TTS_OUTPUT_DIR, suffix)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    try:
        # Utiliser les phonèmes fournis par l'utilisateur
        segments_data = [dict(s) for s in req.phonemes]
        kokoro_url, phoneme_segments = _build_kokoro_video(
            req.job_id, segments_data, wav_path, upload_path, ext
        )

        record["kokoro_dubbed_url"] = kokoro_url
        record["kokoro_phonemes"] = phoneme_segments
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return {
            "job_id": req.job_id,
            "kokoro_dubbed_url": kokoro_url,
            "phonemes": phoneme_segments,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kokoro_dub_manual")
async def kokoro_dub_manual(req: KokoroManualRequest):
    """Génère un doublage Kokoro à partir d'un phonème saisi manuellement.
    Le phonème est utilisé comme un seul segment couvrant la durée totale de la vidéo.
    """
    json_path = os.path.join(TRANSCRIPTIONS_DIR, f"{req.job_id}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")

    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if not req.manual_phoneme.strip():
        raise HTTPException(status_code=400, detail="Le phonème saisi est vide.")

    upload_path = ""
    for ext_try in VIDEO_EXTENSIONS:
        p = os.path.join(UPLOADS_DIR, f"{req.job_id}{ext_try}")
        if os.path.exists(p):
            upload_path = p
            break
    ext = os.path.splitext(upload_path)[1].lower() if upload_path else ".mp4"
    wav_path = os.path.join(UPLOADS_DIR, f"{req.job_id}.wav")

    # Déterminer la durée totale
    if os.path.exists(wav_path):
        total_duration = librosa.get_duration(path=wav_path)
    elif upload_path and os.path.exists(upload_path):
        total_duration = librosa.get_duration(path=upload_path)
    else:
        total_duration = 5.0

    # Construire un unique segment avec le phonème manuel
    segments_data = [{
        "segment_idx": 0,
        "text": req.manual_phoneme,
        "phonemes": req.manual_phoneme,
        "translated_text": req.manual_phoneme,
        "start": 0.0,
        "end": total_duration,
    }]

    # Supprimer les anciens fichiers Kokoro pour forcer la régénération
    for suffix in [f"{req.job_id}_kokoro.wav", f"{req.job_id}_kokoro{ext}"]:
        p = os.path.join(TTS_OUTPUT_DIR, suffix)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    try:
        kokoro_url, phoneme_segments = _build_kokoro_video(
            req.job_id, segments_data, wav_path, upload_path, ext
        )

        record["kokoro_dubbed_url"] = kokoro_url
        record["kokoro_phonemes"] = phoneme_segments
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return {
            "job_id": req.job_id,
            "kokoro_dubbed_url": kokoro_url,
            "phonemes": phoneme_segments,
            "manual_phoneme": req.manual_phoneme,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-to-vezo")
async def speech_to_vezo(
    file: UploadFile = File(...),
    direction: str = Form("english-to-vezo"),
    language: str = Form("en"),
    speed: float = Form(1.15),
    sr: int = Form(22050),
    noise_scale: float = Form(0.35),
    noise_scale_duration: float = Form(0.6),
    highpass_cutoff: int = Form(60),
    noise_gate_threshold: float = Form(0.01),
    peak_norm: float = Form(0.95),
    warmth_level: float = Form(0.5)
):
    """Speech-to-Text en direct/enregistré (Anglais) -> Traduction Vezo/Merina avec Synthèse Vocale."""
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".wav"
    upload_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")
    wav_path = os.path.join(UPLOADS_DIR, f"{job_id}_speech.wav")

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        extract_audio_if_video(upload_path, wav_path)

        # Déterminer le transcripteur STT (MMS STT fixe sur Malgache 'mlg', Whisper fixe sur 'mg')
        if direction in ["merina-to-english", "malagasy-to-english", "vezo-to-english"]:
            try:
                print(f"[STT] Langue fixe : Malgache (mlg). Inférence via Meta MMS STT...")
                transcription_text = mms_stt.transcribe(wav_path, lang="mlg") or ""
            except Exception as e_mms:
                print(f"[WARNING] Échec de MMS STT, basculement vers Whisper avec langue fixe 'mg': {e_mms}")
                w_model = get_whisper_model()
                result = w_model.transcribe(wav_path, language="mg", task="transcribe")
                transcription_text = result.get("text", "").strip()
        else:
            w_model = get_whisper_model()
            result = w_model.transcribe(wav_path, language="en", task="transcribe")
            transcription_text = result.get("text", "").strip()

        if not transcription_text:
            return {
                "transcription": "",
                "translation": "",
                "direction": direction,
                "error": "Aucune parole détectée."
            }

        translated_text = run_translation(transcription_text, model_key="vezo", direction=direction)

        # Générer automatiquement la voix synthèse
        audio_url = None
        if direction in ["merina-to-english", "malagasy-to-english", "vezo-to-english"]:
            try:
                tts_res = synthesize_english_tts(translated_text, speed=speed)
                if tts_res:
                    audio_url = tts_res["audio_url"]
            except Exception as e_tts:
                print(f"[WARNING] English TTS failed in /speech-to-vezo: {e_tts}")
        elif direction == "english-to-merina" or direction == "french-to-merina":
            try:
                tts_res = synthesize_merina_tts(
                    translated_text,
                    speed=speed,
                    target_sr=sr,
                    highpass_cutoff=highpass_cutoff
                )
                if tts_res:
                    audio_url = tts_res["audio_url"]
            except Exception as e_tts:
                print(f"[WARNING] Merina TTS failed in /speech-to-vezo: {e_tts}")
        else:
            try:
                tts_res = synthesize_vezo_tts(
                    translated_text,
                    speed=speed,
                    target_sr=sr,
                    noise_scale=noise_scale,
                    noise_scale_duration=noise_scale_duration,
                    highpass_cutoff=highpass_cutoff,
                    noise_gate_threshold=noise_gate_threshold,
                    peak_norm=peak_norm,
                    warmth_level=warmth_level
                )
                if tts_res:
                    audio_url = tts_res["audio_url"]
            except Exception as e_tts:
                print(f"[WARNING] Vezo TTS failed in /speech-to-vezo: {e_tts}")

        return {
            "transcription": transcription_text,
            "translation": translated_text,
            "direction": direction,
            "language": language,
            "vezo_audio_url": audio_url,
            "audio_url": audio_url
        }
    except Exception as e:
        print(f"Error in /speech-to-vezo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [upload_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

class VezoTTSRequest(BaseModel):
    text: str
    speed: float = 1.15
    sr: int = 22050
    noise_scale: float = 0.35
    noise_scale_duration: float = 0.6
    highpass_cutoff: int = 60
    noise_gate_threshold: float = 0.01
    peak_norm: float = 0.95
    warmth_level: float = 0.5

@app.post("/tts-vezo")
async def tts_vezo(req: VezoTTSRequest):
    """Convertit un texte Vezo en audio avec le modèle fine-tuné et paramètres ajustables."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte est vide.")
    try:
        res = synthesize_vezo_tts(
            req.text,
            speed=req.speed,
            target_sr=req.sr,
            noise_scale=req.noise_scale,
            noise_scale_duration=req.noise_scale_duration,
            highpass_cutoff=req.highpass_cutoff,
            noise_gate_threshold=req.noise_gate_threshold,
            peak_norm=req.peak_norm,
            warmth_level=req.warmth_level
        )
        return res
    except Exception as e:
        print(f"Error in /tts-vezo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class EnglishTTSRequest(BaseModel):
    text: str
    speed: float = 1.0

@app.post("/tts-english")
async def tts_english(req: EnglishTTSRequest):
    """Convertit un texte Anglais en audio avec Kokoro TTS."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte est vide.")
    try:
        res = synthesize_english_tts(req.text, speed=req.speed)
        return res
    except Exception as e:
        print(f"Error in /tts-english: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
