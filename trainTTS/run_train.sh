#!/usr/bin/env bash
# Script officiel de lancement et reprise automatique du Fine-Tuning Vezo TTS V4
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="$SCRIPT_DIR/venv/bin/python"
CONFIG_FILE="$SCRIPT_DIR/vezo_training_config.json"
TRAIN_SCRIPT="$SCRIPT_DIR/finetune-hf-vits/run_vits_finetuning.py"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ Erreur : environnement virtuel non trouvé dans $SCRIPT_DIR/venv"
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/finetune-hf-vits:$SCRIPT_DIR/finetune-hf-vits/monotonic_align:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Gestion du nombre d'époques personnalisé
EPOCHS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --preprocess|-p)
            echo "🎙️ Lancement du pré-traitement audio haute qualité..."
            "$PYTHON_EXEC" "$SCRIPT_DIR/preprocess_audio_dataset.py"
            shift
            ;;
        [0-9]*)
            EPOCHS="$1"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Si le nombre d'époques n'a pas été passé en ligne de commande, demander interactivement
if [ -z "$EPOCHS" ]; then
    echo "========================================================"
    echo "🎙️  CONFIGURATION DU FINE-TUNING TTS VEZO V4"
    echo "========================================================"
    read -p "👉 Entrez le nombre d'époques souhaité [défaut: 40] : " USER_INPUT
    EPOCHS="${USER_INPUT:-40}"
fi

# Validation numérique
if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -le 0 ]; then
    echo "⚠️ Entrée invalide, utilisation de la valeur par défaut : 40 époques"
    EPOCHS=40
fi

# Détection et configuration automatique de la reprise
"$PYTHON_EXEC" -c "
import json, os
from pathlib import Path

cfg_path = '$CONFIG_FILE'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

cfg['num_train_epochs'] = int($EPOCHS)
output_dir = Path(cfg.get('output_dir', '$SCRIPT_DIR/models/mms-tts-vezo-finetuned-v4'))

# Vérifier si des checkpoints existent pour reprise automatique
has_checkpoints = False
if output_dir.exists():
    ckpts = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    if len(ckpts) > 0:
        has_checkpoints = True

if has_checkpoints:
    cfg['resume_from_checkpoint'] = 'latest'
    print(f'🔄 Checkpoint(s) détecté(s) dans {output_dir.name} -> Reprise automatique activée.')
else:
    cfg['resume_from_checkpoint'] = None
    print(f'✨ Aucun checkpoint existant -> Démarrage propre depuis {Path(cfg.get(\"model_name_or_path\")).name}.')

with open(cfg_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=4)
"

OUTPUT_DIR=$("$PYTHON_EXEC" -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")

echo ""
echo "========================================================"
echo "🚀 Lancement du Fine-Tuning Vezo TTS V4"
echo "   Époques configurées : $EPOCHS"
echo "   Sauvegarde continue : max 2 checkpoints (save_total_limit: 2)"
echo "   Dossier de sortie   : $OUTPUT_DIR"
echo "========================================================"
echo ""

exec "$PYTHON_EXEC" "$TRAIN_SCRIPT" "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
