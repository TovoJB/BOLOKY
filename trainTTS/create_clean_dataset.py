#!/usr/bin/env python3
"""
Création d'un dataset propre et filtré (dataset_vezo_clean)
prêt pour le fine-tuning TTS et ASR.

Critères de sélection :
- 1.0s <= Durée <= 15.0s
- 5 <= Caractères <= 300
- Tokens (VitsTokenizer) <= 600
- Fichier audio WAV existant et valide

Organisation générée :
dataset_vezo_clean/
├── audio/
│   └── segments/          (fichiers WAV propres prêts à l'entraînement)
├── metadata/
│   ├── metadata_asr.csv   (format standard ASR avec audio_path, duration, transcript)
│   ├── metadata_tts.csv   (format LJSpeech id|text|text)
│   ├── train_tts.csv      (~95% pour l'entraînement)
│   └── val_tts.csv        (~5% pour la validation)
└── summary.json           (statistiques complètes du dataset propre)
"""

import os
import csv
import json
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DATASET = BASE_DIR / "dataset_vezo"
DEST_DATASET = BASE_DIR / "dataset_vezo_clean"

MAX_CHARS = 300
MIN_CHARS = 5
MAX_DURATION = 15.0
MIN_DURATION = 1.0
MAX_TOKENS = 600

def main():
    print("=" * 65)
    print("🧹 CRÉATION DU DATASET PROPRE (dataset_vezo_clean)")
    print("=" * 65)

    src_asr = SRC_DATASET / "metadata" / "metadata_asr_bible.csv"
    src_wav_dir = SRC_DATASET / "audio" / "segments"

    if not src_asr.exists():
        print(f"Erreur : {src_asr} introuvable.")
        return

    # Préparer les répertoires de destination
    dest_audio_dir = DEST_DATASET / "audio" / "segments"
    dest_meta_dir = DEST_DATASET / "metadata"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_meta_dir.mkdir(parents=True, exist_ok=True)

    clean_records = []
    excluded_records = []
    total_duration_sec = 0.0

    with open(src_asr, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            text = row["transcript"].strip()
            nchars = len(text)
            est_tokens = nchars * 2
            clip_id = Path(row["audio_path"]).stem
            src_wav_file = src_wav_dir / f"{clip_id}.wav"

            # Vérification des critères
            reasons = []
            if not src_wav_file.exists():
                reasons.append("fichier WAV manquant")
            if dur > MAX_DURATION:
                reasons.append(f"durée > {MAX_DURATION}s ({dur:.1f}s)")
            elif dur < MIN_DURATION:
                reasons.append(f"durée < {MIN_DURATION}s ({dur:.1f}s)")
            if nchars > MAX_CHARS:
                reasons.append(f"caractères > {MAX_CHARS} ({nchars} ch)")
            elif nchars < MIN_CHARS:
                reasons.append(f"caractères < {MIN_CHARS} ({nchars} ch)")
            if est_tokens > MAX_TOKENS:
                reasons.append(f"tokens > {MAX_TOKENS} (~{est_tokens})")

            if reasons:
                excluded_records.append({
                    "id": clip_id,
                    "reasons": reasons,
                    "duration": dur,
                    "text": text
                })
            else:
                dest_wav_rel = f"audio/segments/{clip_id}.wav"
                dest_wav_file = dest_audio_dir / f"{clip_id}.wav"

                # Création d'un lien physique (ou copie si non supporté) pour économiser l'espace disque
                if not dest_wav_file.exists():
                    try:
                        os.link(src_wav_file, dest_wav_file)
                    except OSError:
                        shutil.copy2(src_wav_file, dest_wav_file)

                clean_records.append({
                    "id": clip_id,
                    "audio_path": dest_wav_rel,
                    "duration": f"{dur:.3f}",
                    "transcript": text,
                    "language": row.get("language", "skg-x-vz"),
                    "source": row.get("source", "")
                })
                total_duration_sec += dur

    # 1. Sauvegarde metadata ASR propre
    meta_asr_clean = dest_meta_dir / "metadata_asr.csv"
    with open(meta_asr_clean, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "duration", "transcript", "language", "source"])
        writer.writeheader()
        for r in clean_records:
            writer.writerow({
                "audio_path": r["audio_path"],
                "duration": r["duration"],
                "transcript": r["transcript"],
                "language": r["language"],
                "source": r["source"]
            })

    # 2. Sauvegarde metadata TTS propre (format LJSpeech : id|text|text)
    meta_tts_clean = dest_meta_dir / "metadata_tts.csv"
    with open(meta_tts_clean, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(f"{r['id']}|{r['transcript']}|{r['transcript']}\n")

    # 3. Train / Val split (95% train / 5% val) avec seed déterministe
    random.seed(42)
    shuffled = list(clean_records)
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * 0.05)
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]

    train_tts_file = dest_meta_dir / "train_tts.csv"
    with open(train_tts_file, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(f"{r['id']}|{r['transcript']}|{r['transcript']}\n")

    val_tts_file = dest_meta_dir / "val_tts.csv"
    with open(val_tts_file, "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(f"{r['id']}|{r['transcript']}|{r['transcript']}\n")

    # 4. Résumé statistique JSON
    total_hours = total_duration_sec / 3600.0
    summary = {
        "total_clean_samples": len(clean_records),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "excluded_samples": len(excluded_records),
        "total_duration_hours": round(total_hours, 2),
        "total_duration_seconds": round(total_duration_sec, 2),
        "average_sample_duration_sec": round(total_duration_sec / len(clean_records), 2) if clean_records else 0,
        "criteria": {
            "min_duration_sec": MIN_DURATION,
            "max_duration_sec": MAX_DURATION,
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
            "max_tokens": MAX_TOKENS
        }
    }

    with open(DEST_DATASET / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Échantillons conservés : {len(clean_records)} (soit {len(clean_records)/(len(clean_records)+len(excluded_records))*100:.1f}%)")
    print(f"   - Entraînement (Train) : {len(train_set)} échantillons")
    print(f"   - Validation   (Val)   : {len(val_set)} échantillons")
    print(f"✓ Échantillons exclus    : {len(excluded_records)}")
    print(f"✓ Durée totale audio net : {total_hours:.2f} heures ({total_duration_sec/60:.1f} minutes)")
    print(f"✓ Durée moyenne par clip : {total_duration_sec/len(clean_records):.2f} secondes")
    print(f"\n📂 Dataset prêt dans : {DEST_DATASET}")
    print("=" * 65)

if __name__ == "__main__":
    main()
