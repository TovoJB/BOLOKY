#!/usr/bin/env python3
"""
ÉTAPE 1 : Nettoyage et normalisation du texte du dataset pour le tokenizer MMS-TTS.

Le tokenizer VitsTokenizer n'accepte que 31 symboles :
a-z, àìòôỳ, ' - espace, |
Tout le reste est SILENCIEUSEMENT supprimé → dataset corrompu !

Ce script :
1. Diagnostique les caractères invalides dans le dataset
2. Les supprime ou les convertit proprement (ponctuation, chiffres, majuscules)
3. Génère un nouveau dataset_vezo_clean/metadata/metadata_tts_normalized.csv

Usage :
    ./venv/bin/python prepare_dataset_for_vits.py --dry-run  # voir les problèmes sans modifier
    ./venv/bin/python prepare_dataset_for_vits.py             # normaliser et sauvegarder
"""

import os, sys, re, csv, argparse
from pathlib import Path
from collections import Counter

# Vocabulaire exact du tokenizer facebook/mms-tts-mlg
VALID_CHARS = set("abcdefghijklmnoprstyvz àìòôỳ'-|")

DATASET_DIR = Path(__file__).resolve().parent / "dataset_vezo_clean"
META_TTS = DATASET_DIR / "metadata" / "metadata_tts.csv"
META_TRAIN = DATASET_DIR / "metadata" / "train_tts.csv"
META_VAL = DATASET_DIR / "metadata" / "val_tts.csv"

# ─── RÈGLES DE NORMALISATION ────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Normalise un texte pour qu'il soit 100% compatible avec le tokenizer MMS-TTS.
    """
    # 1. Minuscules
    text = text.lower()

    # 2. Apostrophes typographiques → apostrophe ASCII
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('\u02bc', "'")

    # 3. Guillemets → supprimer
    text = text.replace('"', '').replace('\u201c', '').replace('\u201d', '')
    text = text.replace('«', '').replace('»', '')

    # 4. Tiret cadratin → tiret simple
    text = text.replace('—', '-').replace('–', '-')

    # 5. Accents non supportés → équivalent supporté ou suppression
    text = text.replace('é', 'e').replace('ê', 'e').replace('ë', 'e')
    text = text.replace('è', 'e')   # è n'est pas dans le vocab mais à, ì, ò, ô, ỳ le sont
    text = text.replace('â', 'a').replace('á', 'a').replace('ä', 'a')
    text = text.replace('î', 'i').replace('í', 'i').replace('ï', 'i')
    text = text.replace('û', 'u').replace('ú', 'u').replace('ü', 'u')
    text = text.replace('ç', 's')
    text = text.replace('ñ', 'n')

    # 6. Chiffres → version textuelle malagasy (simplification)
    num_map = {
        '0': 'folo', '1': 'iray', '2': 'roa', '3': 'telo',
        '4': 'efatra', '5': 'dimy', '6': 'enina', '7': 'fito',
        '8': 'valo', '9': 'sivy',
    }
    for digit, word in num_map.items():
        text = text.replace(digit, f' {word} ')

    # 7. Ponctuation → supprimer (le tokenizer ne la gère pas)
    text = re.sub(r'[.,;:!?()«»\[\]{}]', '', text)

    # 8. Espaces multiples
    text = re.sub(r' +', ' ', text).strip()

    # 9. Supprimer tout caractère encore invalide
    text = ''.join(c for c in text if c.lower() in VALID_CHARS or c in VALID_CHARS)

    # 10. Dernier nettoyage
    text = re.sub(r' +', ' ', text).strip()

    return text


def check_dataset(meta_file: Path, dry_run: bool = False):
    """Analyse et normalise toutes les entrées du dataset."""
    changed = 0
    dropped = 0
    total = 0
    unknown_counter = Counter()

    normalized_rows = []

    with open(meta_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue

            clip_id, orig_text = parts[0], parts[1]
            total += 1

            # Détecter caractères invalides
            for c in orig_text:
                if c.lower() not in VALID_CHARS and c not in VALID_CHARS:
                    unknown_counter[c] += 1

            norm = normalize_text(orig_text)

            if norm != orig_text.lower().strip():
                changed += 1

            if len(norm) < 5:
                dropped += 1
                continue

            normalized_rows.append((clip_id, norm))

    print("=" * 65)
    print("🔍 DIAGNOSTIC DU DATASET")
    print("=" * 65)
    print(f"Total entrées      : {total}")
    print(f"Entrées modifiées  : {changed} ({changed/total*100:.1f}%)")
    print(f"Entrées supprimées : {dropped} (trop courtes après normalisation)")
    print(f"Entrées conservées : {len(normalized_rows)}")
    print("\nTop 15 caractères invalides (supprimés silencieusement) :")
    for c, count in unknown_counter.most_common(15):
        print(f"  {repr(c)} (U+{ord(c):04X}) → {count} apparitions")

    # Exemple de transformation
    sample = "Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida: 25 taona."
    print(f"\n📋 Exemple de normalisation :")
    print(f"  Avant : {sample}")
    print(f"  Après : {normalize_text(sample)}")

    if not dry_run:
        return normalized_rows
    return None


def save_normalized(rows, meta_file: Path, suffix: str = "_normalized"):
    """Sauvegarde les lignes normalisées dans un nouveau fichier."""
    out_path = meta_file.parent / (meta_file.stem + suffix + meta_file.suffix)
    with open(out_path, 'w', encoding='utf-8') as f:
        for clip_id, text in rows:
            f.write(f"{clip_id}|{text}|{text}\n")
    print(f"\n✅ Fichier normalisé sauvegardé : {out_path}")
    print(f"   {len(rows)} entrées prêtes pour le fine-tuning VITS")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Normalise le dataset TTS Vezo pour le tokenizer MMS-TTS.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyser seulement, sans modifier les fichiers")
    args = parser.parse_args()

    print("\n📂 Fichier analysé :", META_TTS)
    rows = check_dataset(META_TTS, dry_run=args.dry_run)

    if not args.dry_run and rows:
        # Fichier complet normalisé
        save_normalized(rows, META_TTS)

        # Train normalisé
        train_rows = check_dataset(META_TRAIN, dry_run=True) or []
        # Ré-normaliser correctement (pas dry-run)
        train_rows2 = []
        with open(META_TRAIN, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    norm = normalize_text(parts[1])
                    if len(norm) >= 5:
                        train_rows2.append((parts[0], norm))
        save_normalized(train_rows2, META_TRAIN)

        # Val normalisé
        val_rows = []
        with open(META_VAL, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    norm = normalize_text(parts[1])
                    if len(norm) >= 5:
                        val_rows.append((parts[0], norm))
        save_normalized(val_rows, META_VAL)

        print("\n💡 Utilisez les fichiers *_normalized.csv pour le fine-tuning.")


if __name__ == "__main__":
    main()
