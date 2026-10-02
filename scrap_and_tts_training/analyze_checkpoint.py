#!/usr/bin/env python3
"""
Analyse les fichiers checkpoint pour comprendre la structure.
"""
import os, sys, ctypes

VENV_PYTHON = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and sys.executable != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

for p in ["nvidia/nvjitlink/lib/libnvJitLink.so.12", "nvidia/cuda_runtime/lib/libcudart.so.12"]:
    lib_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", p)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

from safetensors.torch import load_file

sd1 = load_file('/home/tovo/Bureau/scrap/models/mms-tts-vezo-finetuned/checkpoint-600/model.safetensors')
sd2 = load_file('/home/tovo/Bureau/scrap/models/mms-tts-vezo-finetuned/checkpoint-600/model_1.safetensors')

print('=== model.safetensors ===')
print(f'Keys: {len(sd1)}')
for k in sorted(sd1.keys())[:15]:
    print(f'  {k}: {list(sd1[k].shape)}')
print('...')
for k in sorted(sd1.keys())[-5:]:
    print(f'  {k}: {list(sd1[k].shape)}')

print()
print('=== model_1.safetensors ===')
print(f'Keys: {len(sd2)}')
for k in sorted(sd2.keys())[:15]:
    print(f'  {k}: {list(sd2[k].shape)}')
print('...')
for k in sorted(sd2.keys())[-5:]:
    print(f'  {k}: {list(sd2[k].shape)}')

has_decoder_1 = any('decoder' in k for k in sd1)
has_decoder_2 = any('decoder' in k for k in sd2)
has_text_enc_1 = any('text_encoder' in k for k in sd1)
has_text_enc_2 = any('text_encoder' in k for k in sd2)
has_disc_1 = any('discriminator' in k or 'period' in k or 'scale' in k for k in sd1)
has_disc_2 = any('discriminator' in k or 'period' in k or 'scale' in k for k in sd2)

print(f'\nmodel.safetensors: decoder={has_decoder_1}, text_encoder={has_text_enc_1}, discriminator={has_disc_1}')
print(f'model_1.safetensors: decoder={has_decoder_2}, text_encoder={has_text_enc_2}, discriminator={has_disc_2}')

param_keys_1 = [k for k in sd1 if 'parametrizations' in k]
param_keys_2 = [k for k in sd2 if 'parametrizations' in k]
print(f'\nParametrized keys in model.safetensors: {len(param_keys_1)}')
print(f'Parametrized keys in model_1.safetensors: {len(param_keys_2)}')

# Check what keys the VitsModel expects
from transformers import VitsModel
model = VitsModel.from_pretrained('/home/tovo/Bureau/scrap/models/mms-tts-mlg-base')
expected_keys = set(model.state_dict().keys())
print(f'\n=== VitsModel expected keys: {len(expected_keys)} ===')
param_expected = [k for k in expected_keys if 'parametrizations' in k]
print(f'Expected parametrized keys: {len(param_expected)}')
for k in sorted(param_expected)[:10]:
    print(f'  {k}')
print('...')
