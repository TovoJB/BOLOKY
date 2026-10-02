#!/usr/bin/env bash
# =============================================================
#  run_pipeline.sh
#  Pipeline complet : prétraitement → entraînement → évaluation
#  Usage : bash run_pipeline.sh [--quick]
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"   # /home/tovo/Bureau/Boloky/Embedding

cd "$ROOT"

# ── Paramètres par défaut (mode complet) ──────────────────────
DIM=300
EPOCH=10
LR=0.05
MINN=2
MAXN=6
THREAD=$(nproc)
MODEL_TYPE="skipgram"

# Mode rapide pour test
if [[ "${1:-}" == "--quick" ]]; then
    echo "[INFO] Mode rapide activé (dim=100, epoch=3)"
    DIM=100
    EPOCH=3
fi

echo "======================================================"
echo "  FastText Malgache – Pipeline d'entraînement"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Threads : $THREAD"
echo "======================================================"

# ── Étape 1 : Prétraitement ────────────────────────────────────
echo ""
echo "─── Étape 1 : Prétraitement ───"
python3 scripts/01_preprocess.py \
    --input       all_dataFinal.txt \
    --output_dir  . \
    --min_len     3 \
    --max_len     512 \
    --deduplicate \
    --valid_ratio 0.05

echo "  ✓ corpus.train.txt et corpus.valid.txt générés"

# ── Étape 2 : Installation FastText (si absent) ────────────────
if ! python3 -c "import fasttext" 2>/dev/null; then
    echo ""
    echo "─── Installation de fasttext ───"
    pip install fasttext-wheel --quiet
fi

# ── Étape 3 : Entraînement ─────────────────────────────────────
echo ""
echo "─── Étape 2 : Entraînement FastText ($MODEL_TYPE) ───"
python3 scripts/02_train_fasttext.py \
    --input     corpus.train.txt \
    --output    models/mg_fasttext \
    --model     $MODEL_TYPE \
    --dim       $DIM \
    --epoch     $EPOCH \
    --lr        $LR \
    --minn      $MINN \
    --maxn      $MAXN \
    --minCount  5 \
    --thread    $THREAD

echo "  ✓ Modèle sauvegardé dans models/"

# ── Étape 4 : Évaluation ───────────────────────────────────────
echo ""
echo "─── Étape 3 : Évaluation ───"
python3 scripts/03_evaluate.py \
    --model      models/mg_fasttext.bin \
    --output_dir eval/

echo ""
echo "======================================================"
echo "  Pipeline terminé !"
echo "  Modèle    : models/mg_fasttext.bin"
echo "  Résultats : eval/"
echo "======================================================"
