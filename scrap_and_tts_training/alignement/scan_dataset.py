#!/usr/bin/env python3
"""
Rapport d'analyse et de marquage du dataset Vezo TTS/ASR.
Identifie les fichiers audio et textes qui nécessitent un découpage ou un nettoyage.
"""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_ASR = BASE_DIR / "dataset_vezo" / "metadata" / "metadata_asr_bible.csv"

MAX_CHARS = 300
MAX_DURATION = 15.0
MIN_DURATION = 1.0
MIN_CHARS = 5
MAX_TOKENS = 600

def main():
    if not METADATA_ASR.exists():
        print(f"Erreur : fichier {METADATA_ASR} introuvable.")
        sys.exit(1)

    longs = []
    courts = []
    oks = []

    with open(METADATA_ASR, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            text = row["transcript"]
            nchars = len(text)
            est_tokens = nchars * 2  # approximation VitsTokenizer (ratio ~1.9 - 2.0)
            
            entry = {
                "id": Path(row["audio_path"]).stem,
                "path": row["audio_path"],
                "duration": dur,
                "chars": nchars,
                "tokens": est_tokens,
                "text": text
            }

            if dur > MAX_DURATION or nchars > MAX_CHARS or est_tokens > MAX_TOKENS:
                entry["reason"] = []
                if dur > MAX_DURATION:
                    entry["reason"].append(f"durée > {MAX_DURATION}s ({dur:.1f}s)")
                if nchars > MAX_CHARS:
                    entry["reason"].append(f"caractères > {MAX_CHARS} ({nchars} ch)")
                if est_tokens > MAX_TOKENS:
                    entry["reason"].append(f"tokens > {MAX_TOKENS} (~{est_tokens} tok)")
                longs.append(entry)
            elif dur < MIN_DURATION or nchars < MIN_CHARS:
                entry["reason"] = []
                if dur < MIN_DURATION:
                    entry["reason"].append(f"durée < {MIN_DURATION}s ({dur:.1f}s)")
                if nchars < MIN_CHARS:
                    entry["reason"].append(f"caractères < {MIN_CHARS} ({nchars} ch)")
                courts.append(entry)
            else:
                oks.append(entry)

    total = len(longs) + len(courts) + len(oks)
    print("=" * 65)
    print("📊 BILAN QUALITÉ ET MARQUAGE DU DATASET VEZO")
    print("=" * 65)
    print(f"Total des entrées : {total}")
    print(f"✅ Prêts pour le TTS/ASR (OK)      : {len(oks):>5} ({len(oks)/total*100:.1f}%)")
    print(f"🔴 Trop longs (à découper)         : {len(longs):>5} ({len(longs)/total*100:.1f}%)")
    print(f"🟡 Trop courts (à vérifier/ignorer): {len(courts):>5} ({len(courts)/total*100:.1f}%)")
    print("=" * 65)

    if longs:
        print("\n🔍 TOP 10 DES ENTRÉES LES PLUS LONGUES (À DÉCOUPER) :")
        longs_sorted = sorted(longs, key=lambda x: x["duration"], reverse=True)
        for i, it in enumerate(longs_sorted[:10], 1):
            reasons = ", ".join(it["reason"])
            print(f"\n{i}. [{it['id']}] — {it['duration']:.1f}s | {it['chars']} ch | ~{it['tokens']} tok")
            print(f"   Motif: {reasons}")
            print(f"   Texte: \"{it['text'][:120]}...\"")

    print("\n💡 Pour découper visuellement ces entrées avec l'interface graphique :")
    print("   python3 alignement/interface_decoupage.py")

if __name__ == "__main__":
    main()
