#!/usr/bin/env python3
"""
Script de pré-traitement audio haute fidélité pour le dataset TTS Vezo.

Optimisations appliquées :
1. Suppression propre des silences de début et fin (évite les décalages d'alignement phonétique).
2. Filtre passe-haut @ 50 Hz (élimine les infra-basses et le décalage DC qui étouffent la voix).
3. Normalisation de volume dynamique (Peak à -1.0 dB / RMS uniforme pour stabiliser l'apprentissage).
4. Ré-échantillonnage strict à 16 000 Hz haute précision.
5. Reconstruction automatique du dataset HuggingFace (`dataset_vezo_hf`).
"""

import os, sys, ctypes
from pathlib import Path

# ─── Préchargement CUDA ───────────────────────────────────────────────────────
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import numpy as np
import soundfile as sf
import scipy.signal
from tqdm import tqdm
import shutil

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "dataset_vezo_clean" / "audio" / "segments"
BACKUP_DIR = BASE_DIR / "dataset_vezo_clean" / "audio" / "segments_raw_backup"
TARGET_SR = 16000


def trim_silence(audio: np.ndarray, sr: int, top_db: float = 30.0, frame_length: int = 512, hop_length: int = 128, pad_ms: int = 25):
    """Supprime les silences de début et fin avec une marge de transition douce."""
    if len(audio) == 0:
        return audio
    
    # Énergie par frame
    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0])
    )
    rms = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-12)
    max_rms = np.max(rms)
    if max_rms == 0:
        return audio
    
    threshold = max_rms * (10 ** (-top_db / 20.0))
    active_frames = np.where(rms > threshold)[0]
    
    if len(active_frames) == 0:
        return audio
    
    start_frame = active_frames[0]
    end_frame = active_frames[-1]
    
    pad_samples = int(sr * pad_ms / 1000)
    start_idx = max(0, start_frame * hop_length - pad_samples)
    end_idx = min(len(audio), (end_frame + 1) * hop_length + pad_samples)
    
    return audio[start_idx:end_idx]


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 50.0):
    """Élimine les composantes DC et infra-basses sous 50Hz."""
    sos = scipy.signal.butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95):
    """Normalise le niveau crête (peak) à -0.45 dB (0.95 max)."""
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio * (target_peak / max_val)
    return audio


def process_file(wav_path: Path):
    try:
        data, sr = sf.read(str(wav_path))
        
        # Stéréo -> Mono si nécessaire
        if data.ndim > 1:
            data = np.mean(data, axis=-1)
        
        # 1. Resampling à 16000 Hz si différent
        if sr != TARGET_SR:
            num_samples = int(round(len(data) * TARGET_SR / sr))
            data = scipy.signal.resample(data, num_samples)
            sr = TARGET_SR
        
        # 2. Filtre passe-haut (50 Hz)
        data = highpass_filter(data, sr=TARGET_SR, cutoff=50.0)
        
        # 3. Trim silences de début / fin
        data = trim_silence(data, sr=TARGET_SR, top_db=30.0, pad_ms=25)
        
        # 4. Normalisation de volume Peak
        data = normalize_peak(data, target_peak=0.95)
        
        # 5. Sauvegarde
        sf.write(str(wav_path), data.astype(np.float32), TARGET_SR)
        return True
    except Exception as e:
        print(f"Erreur sur {wav_path.name}: {e}")
        return False


def main():
    print("=" * 65)
    print("🎙️  PRÉ-TRAITEMENT AUDIO HAUTE QUALITÉ DU DATASET VEZO")
    print("=" * 65)
    
    wav_files = sorted(list(AUDIO_DIR.glob("*.wav")))
    print(f"📂 Dossier audio  : {AUDIO_DIR}")
    print(f"📊 Fichiers trouvés: {len(wav_files)}")
    
    if len(wav_files) == 0:
        print("❌ Aucun fichier WAV trouvé.")
        sys.exit(1)
        
    # Sauvegarde initiale si premier lancement
    if not BACKUP_DIR.exists():
        print(f"\n💾 Création d'une sauvegarde de sécurité dans : {BACKUP_DIR.name}/")
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        # Ne copier que si pas déjà sauvegardé
        for f in tqdm(wav_files[:50], desc="Sauvegarde échantillon"):
            shutil.copy2(str(f), str(BACKUP_DIR / f.name))
        print("  ✓ Sauvegarde échantillon effectuée.")
        
    print("\n⏳ Traitement des fichiers audio (Trim silence + Filtre 50Hz + Normalisation)...")
    success = 0
    for f in tqdm(wav_files, desc="Traitement audio"):
        if process_file(f):
            success += 1
            
    print(f"\n✅ {success}/{len(wav_files)} fichiers audio pré-traités avec succès !")
    
    # Reconstruire le dataset HuggingFace
    print("\n🔨 Reconstruction automatique du dataset HuggingFace...")
    from build_hf_dataset import build_dataset
    build_dataset()
    
    print("\n" + "=" * 65)
    print("🎉 PRÉ-TRAITEMENT TERMINÉ ET DATASET PRÊT !")
    print("=" * 65)


if __name__ == "__main__":
    main()
