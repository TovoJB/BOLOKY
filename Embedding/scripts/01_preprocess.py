#!/usr/bin/env python3
"""
01_preprocess.py
================
Préparation du corpus malgache pour l'entraînement FastText.

Actions effectuées :
  - Normalisation Unicode (NFC)
  - Mise en minuscules
  - Suppression des caractères de contrôle et des URLs
  - Détachement ponctuation : « mot, » → « mot , » (évite les tokens collés)
  - Suppression des tokens parasites (ponctuation seule, chiffres isolés)
  - Filtrage des lignes trop courtes / trop longues
  - Déduplication exacte (optionnelle, activée par défaut)
  - Division train / valid (95 / 5)

Usage :
  python3 scripts/01_preprocess.py \
      --input  all_dataFinal.txt \
      --output_dir . \
      --min_len 3 \
      --max_len 512 \
      --deduplicate \
      --valid_ratio 0.05
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Patterns de nettoyage
# ──────────────────────────────────────────────
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
CTRL_RE   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MULTI_SP  = re.compile(r" {2,}")

# Ponctuation à détacher des mots (insère un espace avant/après)
PUNCT_ATTACH = re.compile(
    r"""(?<=[\w\u00C0-\u024F])   # après un caractère de mot
        ([.,;:!?()\[\]{}\"'«»…&@#%=/+*\\|~^`<>])  # ponctuation
    |                              # OU
        ([.,;:!?()\[\]{}\"'«»…&@#%=/+*\\|~^`<>])
        (?=[\w\u00C0-\u024F])      # avant un caractère de mot
    """,
    re.VERBOSE,
)

# Token parasite : ne contient aucune lettre (chiffres seuls, ponctuation seule, …)
NOISY_TOKEN = re.compile(r"^[^\w]*$|^[\d]+$")  # pas de lettre OU que des chiffres


def clean_line(text: str) -> str:
    """Normalise, nettoie et tokenise une ligne de texte."""
    # 1. Normalisation Unicode + strip
    text = unicodedata.normalize("NFC", text)
    text = text.strip()

    # 2. Suppression URLs et caractères de contrôle
    text = URL_RE.sub("", text)
    text = CTRL_RE.sub("", text)

    # 3. Minuscules
    text = text.lower()

    # 4. Détachement ponctuation collée aux mots
    #    « mot,  » → « mot ,  »  |  « (mot » → « ( mot »
    text = PUNCT_ATTACH.sub(r" \1\2 ", text)

    # 5. Suppression des tokens parasites (ponctuation seule, chiffres seuls)
    tokens = [t for t in text.split() if not NOISY_TOKEN.match(t)]
    text = " ".join(tokens)

    # 6. Normalisation des espaces multiples
    text = MULTI_SP.sub(" ", text)
    return text.strip()


def is_valid(text: str, min_len: int, max_len: int) -> bool:
    """Filtre les lignes trop courtes, trop longues ou majoritairement numériques."""
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < min_len or len(tokens) > max_len:
        return False
    # Rejette si >50 % des tokens sont des chiffres
    num_tokens = sum(1 for t in tokens if re.fullmatch(r"[\d.,]+", t))
    if num_tokens / len(tokens) > 0.5:
        return False
    return True


def preprocess(
    input_path: Path,
    output_dir: Path,
    min_len: int,
    max_len: int,
    deduplicate: bool,
    valid_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "corpus.train.txt"
    valid_path = output_dir / "corpus.valid.txt"

    seen = set() if deduplicate else None
    total = kept = dup = 0
    valid_every = max(1, int(1 / valid_ratio))  # 1 ligne sur N → valid

    log.info(f"Lecture de : {input_path} ({input_path.stat().st_size / 1e6:.1f} Mo)")

    with (
        open(input_path,  encoding="utf-8", errors="replace") as fin,
        open(train_path,  "w", encoding="utf-8") as f_train,
        open(valid_path,  "w", encoding="utf-8") as f_valid,
    ):
        for line in fin:
            total += 1
            cleaned = clean_line(line)

            if not is_valid(cleaned, min_len, max_len):
                continue

            if deduplicate:
                if cleaned in seen:
                    dup += 1
                    continue
                seen.add(cleaned)

            kept += 1
            if kept % valid_every == 0:
                f_valid.write(cleaned + "\n")
            else:
                f_train.write(cleaned + "\n")

            if total % 500_000 == 0:
                log.info(f"  {total:>8,} lus | {kept:>8,} gardés | {dup:>8,} dups")

    log.info("=" * 60)
    log.info(f"Total lu      : {total:,}")
    log.info(f"Conservé      : {kept:,}  ({kept/total*100:.1f} %)")
    log.info(f"Dédupliqués   : {dup:,}")
    log.info(f"Train         : {train_path}")
    log.info(f"Valid         : {valid_path}")


# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prétraitement corpus FastText")
    parser.add_argument("--input",       default="all_dataFinal.txt")
    parser.add_argument("--output_dir",  default=".")
    parser.add_argument("--min_len",     type=int, default=3,    help="Nb tokens min")
    parser.add_argument("--max_len",     type=int, default=512,  help="Nb tokens max")
    parser.add_argument("--deduplicate", action="store_true", default=True)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    args = parser.parse_args()

    preprocess(
        input_path  = Path(args.input),
        output_dir  = Path(args.output_dir),
        min_len     = args.min_len,
        max_len     = args.max_len,
        deduplicate = args.deduplicate,
        valid_ratio = args.valid_ratio,
    )


if __name__ == "__main__":
    main()
