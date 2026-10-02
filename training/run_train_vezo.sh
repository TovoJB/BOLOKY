#!/bin/bash
# Script de lancement pour l'entraînement Merina -> Vezo
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV_PYTHON="/home/tovo/Bureau/scrap/venv/bin/python"
VENV_NVIDIA="/home/tovo/Bureau/scrap/venv/lib/python3.12/site-packages/nvidia"

if [ -d "$VENV_NVIDIA" ]; then
    NVIDIA_LIB_PATHS=$(find "$VENV_NVIDIA" -type d -name "lib" | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}"
fi

echo "=========================================================="
echo "Démarrage de l'entraînement Merina -> Vezo"
echo "Python : $VENV_PYTHON"
echo "=========================================================="

"$VENV_PYTHON" "$DIR/train_merina_to_vezo.py"
