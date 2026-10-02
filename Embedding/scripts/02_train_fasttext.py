#!/usr/bin/env python3
"""
02_train_fasttext.py
====================
Entraîne un modèle FastText (skipgram ou cbow) sur le corpus malgache.

Le FastText gère nativement les sous-mots (n-grammes de caractères),
ce qui est particulièrement adapté au malgache, langue agglutinante
riche en affixes (mi-, ma-, man-, -ana, -ina, etc.).

Usage :
  python3 scripts/02_train_fasttext.py \
      --input  corpus.train.txt \
      --output models/mg_fasttext \
      --model  skipgram \
      --dim    300 \
      --epoch  10 \
      --lr     0.05 \
      --minCount 5 \
      --minn   2 \
      --maxn   6 \
      --thread 8
"""

import argparse
import logging
import time
from pathlib import Path

import fasttext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train(args) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Modèle        : {args.model}")
    log.info(f"Corpus        : {input_path} ({input_path.stat().st_size / 1e6:.1f} Mo)")
    log.info(f"Dimensions    : {args.dim}")
    log.info(f"Époques       : {args.epoch}")
    log.info(f"Lr            : {args.lr}")
    log.info(f"N-grammes     : minn={args.minn}  maxn={args.maxn}  (sous-mots)")
    log.info(f"minCount      : {args.minCount}")
    log.info(f"Threads       : {args.thread}")
    log.info(f"Sortie        : {output_path}")
    log.info("=" * 60)

    t0 = time.time()

    model = fasttext.train_unsupervised(
        input      = str(input_path),
        model      = args.model,          # "skipgram" | "cbow"
        dim        = args.dim,            # taille des vecteurs
        epoch      = args.epoch,          # nombre de passes
        lr         = args.lr,             # taux d'apprentissage
        wordNgrams = 1,                   # n-grammes de mots (1 = unigrams)
        minn       = args.minn,           # n-gramme de caractères min
        maxn       = args.maxn,           # n-gramme de caractères max
        minCount   = args.minCount,       # fréquence minimale d'un mot
        bucket     = args.bucket,         # nb de buckets pour les n-grammes
        neg        = args.neg,            # négatifs par exemple
        loss       = args.loss,           # "ns" | "hs" | "softmax"
        thread     = args.thread,         # parallélisme
        verbose    = 2,
    )

    elapsed = time.time() - t0
    log.info(f"Entraînement terminé en {elapsed/60:.1f} min")

    # Sauvegarde du modèle complet (.bin)
    bin_path = str(output_path) + ".bin"
    model.save_model(bin_path)
    log.info(f"Modèle sauvegardé : {bin_path}")

    # Taille du vocabulaire
    words = model.get_words()
    log.info(f"Vocabulaire       : {len(words):,} mots")

    # Test rapide sur quelques mots malgaches importants
    test_words = ["mianatra", "mahay", "teny", "any", "misotro", "mihinana"]
    log.info("\nTest – mots les plus proches (top 5) :")
    for w in test_words:
        try:
            neighbors = model.get_nearest_neighbors(w, k=5)
            scores = ", ".join(f"{nw}({sc:.3f})" for sc, nw in neighbors)
            log.info(f"  {w:20s} → {scores}")
        except Exception:
            log.info(f"  {w:20s} → (hors vocabulaire)")

    log.info("Terminé.")


# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Entraînement FastText – Malgache")
    parser.add_argument("--input",      default="corpus.train.txt")
    parser.add_argument("--output",     default="models/mg_fasttext")
    parser.add_argument("--model",      default="skipgram", choices=["skipgram", "cbow"])
    parser.add_argument("--dim",        type=int,   default=300)
    parser.add_argument("--epoch",      type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--minCount",   type=int,   default=5)
    parser.add_argument("--minn",       type=int,   default=2,
                        help="Min char n-gram (2 capture les préfixes ma-, mi-...)")
    parser.add_argument("--maxn",       type=int,   default=6,
                        help="Max char n-gram (6 capture les suffixes -ina, -ana...)")
    parser.add_argument("--bucket",     type=int,   default=2_000_000)
    parser.add_argument("--neg",        type=int,   default=5)
    parser.add_argument("--loss",       default="ns", choices=["ns", "hs", "softmax"])
    parser.add_argument("--thread",     type=int,   default=8)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
