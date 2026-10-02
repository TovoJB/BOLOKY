#!/usr/bin/env bash
# run_venv.sh - Lance n'importe quel script Python du projet avec le venv CUDA
# Usage:
#   ./run_venv.sh <script.py> [args...]
# Exemple:
#   ./run_venv.sh convert_discriminator.py --language mlg --output models/mms-tts-mlg-with-disc

PYTHON=/home/tovo/Bureau/scrap/venv/bin/python
PRELOAD_SCRIPT=/home/tovo/Bureau/scrap/cuda_preload.py

# Script de préchargement en ligne si nécessaire
cat > "$PRELOAD_SCRIPT" << 'ENDPY'
import os, sys, ctypes, glob
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, "lib", f"python{v.major}.{v.minor}", "site-packages")
for lib in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p): ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
ENDPY

# Exécuter avec préchargement
exec "$PYTHON" -c "
import runpy, sys, os, ctypes
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, 'lib', f'python{v.major}.{v.minor}', 'site-packages')
for lib in ['nvidia/nvjitlink/lib/libnvJitLink.so.12', 'nvidia/cuda_runtime/lib/libcudart.so.12']:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p): ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
import importlib.util
spec = importlib.util.spec_from_file_location('__main__', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[1:]
spec.loader.exec_module(mod)
" "$@"
