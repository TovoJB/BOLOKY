import os, sys, ctypes
v = sys.version_info
site_pkgs = os.path.join(sys.prefix, 'lib', f'python{v.major}.{v.minor}', 'site-packages')
for lib in ['nvidia/nvjitlink/lib/libnvJitLink.so.12', 'nvidia/cuda_runtime/lib/libcudart.so.12']:
    p = os.path.join(site_pkgs, lib)
    if os.path.exists(p):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except Exception: pass

import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

BASE_DIR = Path('/home/tovo/Bureau/scrap')
MODELS_DIR = BASE_DIR / 'models'
CKPT_DIR = MODELS_DIR / 'mms-tts-vezo-finetuned' / 'checkpoint-7200'
REF_BASE_DIR = MODELS_DIR / 'mms-tts-mlg-with-disc'
V4_BASE_DIR = MODELS_DIR / 'mms-tts-vezo-v4-base'

V4_BASE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copier configs
for cfg in ['config.json', 'tokenizer_config.json', 'vocab.json', 'added_tokens.json', 'preprocessor_config.json']:
    src = REF_BASE_DIR / cfg
    if src.exists():
        import shutil
        shutil.copy2(src, V4_BASE_DIR / cfg)

# 2. Charger les poids de checkpoint-7200
gen_sd = load_file(str(CKPT_DIR / 'model.safetensors'))
disc_sd = load_file(str(CKPT_DIR / 'model_1.safetensors'))

raw_sd = {}
for k, v in gen_sd.items():
    raw_sd[k] = v
for k, v in disc_sd.items():
    if k.startswith('discriminators.'):
        raw_sd['discriminator.' + k] = v
    elif k.startswith('discriminator.'):
        raw_sd[k] = v
    else:
        raw_sd['discriminator.discriminators.' + k] = v

# Step A: Collapse all parametrizations / weight_g / weight_v into .weight
collapsed_sd = {}
keys = set(raw_sd.keys())
handled = set()

for k in sorted(keys):
    if k in handled:
        continue
    if 'parametrizations.weight.original0' in k:
        base_k = k.replace('.parametrizations.weight.original0', '')
        v_k = base_k + '.parametrizations.weight.original1'
        if v_k in keys:
            g = raw_sd[k]
            v = raw_sd[v_k]
            norm_v = torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
            weight = g * (v / (norm_v + 1e-12))
            collapsed_sd[base_k + '.weight'] = weight
            handled.add(k)
            handled.add(v_k)
            continue
    elif 'parametrizations.weight.original1' in k:
        continue
    elif k.endswith('.weight_g'):
        base_k = k[:-9]
        v_k = base_k + '.weight_v'
        if v_k in keys:
            g = raw_sd[k]
            v = raw_sd[v_k]
            norm_v = torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
            weight = g * (v / (norm_v + 1e-12))
            collapsed_sd[base_k + '.weight'] = weight
            handled.add(k)
            handled.add(v_k)
            continue
    elif k.endswith('.weight_v'):
        continue
    else:
        collapsed_sd[k] = raw_sd[k]
        handled.add(k)

# Step B: For Wavenet in posterior_encoder and flow.flows, convert .weight to parametrizations
final_sd = {}
for k, val in collapsed_sd.items():
    if ('posterior_encoder.wavenet' in k or 'flow.flows' in k) and k.endswith('.weight') and 'conv_pre' not in k and 'conv_post' not in k:
        base_k = k[:-7]
        g = torch.linalg.vector_norm(val, dim=tuple(range(1, val.ndim)), keepdim=True)
        v = val
        final_sd[base_k + '.parametrizations.weight.original0'] = g
        final_sd[base_k + '.parametrizations.weight.original1'] = v
    else:
        final_sd[k] = val

ref_sd = load_file(str(REF_BASE_DIR / 'model.safetensors'))
ref_keys = set(ref_sd.keys())
final_keys = set(final_sd.keys())

missing = ref_keys - final_keys
unexpected = final_keys - ref_keys

print(f'Missing keys vs reference base ({len(missing)}): {list(missing)[:5]}')
print(f'Unexpected keys vs reference base ({len(unexpected)}): {list(unexpected)[:5]}')

save_file(final_sd, str(V4_BASE_DIR / 'model.safetensors'))
print(f'✅ Modèle V4 sauvegardé avec succès dans {V4_BASE_DIR}')

# Test load
sys.path.insert(0, str(BASE_DIR / 'finetune-hf-vits'))
from utils import VitsModelForPreTraining
m = VitsModelForPreTraining.from_pretrained(str(V4_BASE_DIR))
print('🎉 TEST DE CHARGEMENT RÉUSSI À 100% !')
