#!/usr/bin/env python3
"""
Correctif complet pour charger un checkpoint VITS (accelerator.save_state)
dans VitsModel.from_pretrained().

Le checkpoint mélange :
  - Ancien format weight_norm : weight_g / weight_v 
  - Nouveau format parametrizations : parametrizations.weight.original0/1
  - Poids simples : .weight

Le VitsModel (avec PyTorch >= 2.0) attend :
  - decoder.resblocks/upsampler : parametrizations.weight.original0/1
  - flow.wavenet, posterior_encoder.wavenet : parametrizations.weight.original0/1  
  - flow.conv_pre/conv_post : .weight simple (car remove_weight_norm est appelé à la fin)

La solution : charger le modèle VITS avec apply_weight_norm, injecter les poids
du checkpoint en les convertissant au bon format, puis sauvegarder proprement.
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

import torch
import torch.nn as nn
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "models" / "mms-tts-vezo-finetuned"
BASE_MODEL_DIR = BASE_DIR / "models" / "mms-tts-mlg-base"

def get_latest_checkpoint(models_dir):
    ckpts = [d for d in models_dir.glob("checkpoint-*") if d.is_dir()]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.name.split("-")[1]))
    return ckpts[-1]

CHECKPOINT_DIR = get_latest_checkpoint(OUTPUT_DIR) or (OUTPUT_DIR / "checkpoint-600")
SAMPLES_DIR = BASE_DIR / "tts_samples"


def reconstruct_weight_from_gv(g, v, dim=0):
    """Reconstruit weight depuis weight_g et weight_v (ancien format weight_norm)."""
    dims = list(range(v.ndim))
    dims.remove(dim)
    norm_v = torch.norm_except_dim(v, 2, dim)
    weight = g * (v / norm_v)
    return weight


def main():
    print("=" * 65)
    print("🔧 CORRECTIF COMPLET CHECKPOINT VITS")
    print("=" * 65)

    # 1. Charger le state dict du checkpoint
    ckpt_file = CHECKPOINT_DIR / "model.safetensors"
    print(f"\n📂 Chargement du checkpoint : {ckpt_file}")
    ckpt_sd = load_file(str(ckpt_file))
    print(f"   Clés dans le checkpoint : {len(ckpt_sd)}")

    # 2. Charger le modèle de base (avec weight_norm appliqué via __init__)
    print(f"\n📂 Chargement du modèle de base : {BASE_MODEL_DIR}")
    model = VitsModel.from_pretrained(str(BASE_MODEL_DIR))
    expected_sd = model.state_dict()
    print(f"   Clés attendues par le modèle : {len(expected_sd)}")

    # 3. Construire un mapping du checkpoint vers le format attendu
    # Analyser les clés du checkpoint
    ckpt_old_wn = {}   # ancien format weight_norm: base -> {g: tensor, v: tensor}
    ckpt_new_wn = {}   # nouveau format parametrizations: base -> {0: tensor, 1: tensor}
    ckpt_plain = {}    # poids simples

    for key, val in ckpt_sd.items():
        if key.endswith(".weight_g"):
            base = key[:-len(".weight_g")]
            ckpt_old_wn.setdefault(base, {})["g"] = val
        elif key.endswith(".weight_v"):
            base = key[:-len(".weight_v")]
            ckpt_old_wn.setdefault(base, {})["v"] = val
        elif ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            ckpt_new_wn.setdefault(base, {})["0"] = val
        elif ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", "")
            ckpt_new_wn.setdefault(base, {})["1"] = val
        else:
            ckpt_plain[key] = val

    print(f"\n📊 Checkpoint analysis:")
    print(f"   Old weight_norm (weight_g/v) layers : {len(ckpt_old_wn)}")
    print(f"   New parametrizations layers         : {len(ckpt_new_wn)}")
    print(f"   Plain weight keys                   : {len(ckpt_plain)}")

    # 4. Analyser les clés attendues par le modèle
    expected_new_wn = {}  # parametrizations attendues
    expected_plain = {}   # poids simples attendus

    for key, val in expected_sd.items():
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            expected_new_wn.setdefault(base, {})["0"] = val
        elif ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", "")
            expected_new_wn.setdefault(base, {})["1"] = val
        else:
            expected_plain[key] = val

    print(f"\n📊 Model expects:")
    print(f"   Parametrized layers : {len(expected_new_wn)}")
    print(f"   Plain weight keys   : {len(expected_plain)}")

    # 5. Construire le nouveau state dict
    new_sd = {}
    matched = 0
    converted = 0
    missing = 0

    for key, expected_val in expected_sd.items():
        # Case A: Key exists directly in checkpoint plain weights
        if key in ckpt_plain:
            new_sd[key] = ckpt_plain[key]
            matched += 1
            continue

        # Case B: Model expects parametrizations.weight.original0/1
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            # B1: Checkpoint has old-style weight_g/v for this layer
            if base in ckpt_old_wn and "g" in ckpt_old_wn[base]:
                new_sd[key] = ckpt_old_wn[base]["g"]
                converted += 1
                continue
            # B2: Checkpoint has new-style parametrizations
            if base in ckpt_new_wn and "0" in ckpt_new_wn[base]:
                new_sd[key] = ckpt_new_wn[base]["0"]
                matched += 1
                continue
            # B3: Checkpoint has plain .weight → decompose into g, v
            weight_key = f"{base}.weight"
            if weight_key in ckpt_plain:
                w = ckpt_plain[weight_key]
                # g = norms of w along dim 0
                dims = list(range(w.ndim))
                dims.remove(0)
                if dims:
                    g = torch.norm(w, dim=dims, keepdim=True)
                else:
                    g = torch.norm(w).unsqueeze(0)
                new_sd[key] = g
                converted += 1
                continue

        if ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", "")
            # B1: Checkpoint has old-style weight_v
            if base in ckpt_old_wn and "v" in ckpt_old_wn[base]:
                new_sd[key] = ckpt_old_wn[base]["v"]
                converted += 1
                continue
            # B2: Checkpoint has new-style parametrizations
            if base in ckpt_new_wn and "1" in ckpt_new_wn[base]:
                new_sd[key] = ckpt_new_wn[base]["1"]
                matched += 1
                continue
            # B3: Checkpoint has plain .weight → v = w itself
            weight_key = f"{base}.weight"
            if weight_key in ckpt_plain:
                new_sd[key] = ckpt_plain[weight_key]
                converted += 1
                continue

        # Case C: Model expects plain .weight but checkpoint has parametrized
        if key.endswith(".weight"):
            base = key[:-len(".weight")]
            # C1: Old-style weight_g/v → reconstruct
            if base in ckpt_old_wn and "g" in ckpt_old_wn[base] and "v" in ckpt_old_wn[base]:
                w = reconstruct_weight_from_gv(ckpt_old_wn[base]["g"], ckpt_old_wn[base]["v"])
                new_sd[key] = w
                converted += 1
                continue
            # C2: New-style parametrizations → reconstruct
            if base in ckpt_new_wn and "0" in ckpt_new_wn[base] and "1" in ckpt_new_wn[base]:
                g, v = ckpt_new_wn[base]["0"], ckpt_new_wn[base]["1"]
                w = reconstruct_weight_from_gv(g, v)
                new_sd[key] = w
                converted += 1
                continue

        # Fallback: use model's initialized value (likely random)
        print(f"  ⚠️  MISSING from checkpoint: {key}")
        new_sd[key] = expected_val
        missing += 1

    print(f"\n📊 Résultat du mapping:")
    print(f"   Matched directement   : {matched}")
    print(f"   Convertis (format)    : {converted}")
    print(f"   Manquants (aléatoire) : {missing}")
    print(f"   Total                 : {len(new_sd)} / {len(expected_sd)}")

    # 6. Charger les poids dans le modèle
    print("\n🔄 Chargement des poids dans le modèle...")
    result = model.load_state_dict(new_sd, strict=True)
    print(f"   Missing keys  : {len(result.missing_keys)}")
    print(f"   Unexpected keys: {len(result.unexpected_keys)}")

    # 7. Retirer weight_norm UNIQUEMENT des mêmes couches que le script d'entraînement
    # (lignes 1504-1507 de run_vits_finetuning.py)
    # NE PAS retirer des wavenet/posterior_encoder car VitsModel.__init__ les re-applique
    print("\n🔧 Retrait du weight_norm (decoder + flow.conv_pre/post seulement)...")

    def safe_remove_weight_norm(module):
        """Remove weight_norm compatible avec l'ancien et le nouveau PyTorch."""
        try:
            nn.utils.remove_weight_norm(module)
        except (ValueError, AttributeError):
            try:
                nn.utils.parametrize.remove_parametrizations(module, "weight")
            except Exception:
                pass

    # Decoder: upsampler + resblocks (comme model.decoder.remove_weight_norm())
    for layer in model.decoder.upsampler:
        safe_remove_weight_norm(layer)
    for block in model.decoder.resblocks:
        for conv in block.convs1:
            safe_remove_weight_norm(conv)
        for conv in block.convs2:
            safe_remove_weight_norm(conv)

    # Flow: conv_pre et conv_post seulement
    for flow in model.flow.flows:
        safe_remove_weight_norm(flow.conv_pre)
        safe_remove_weight_norm(flow.conv_post)

    # Vérifier les clés restantes
    final_sd = model.state_dict()
    param_keys = [k for k in final_sd if 'parametrizations' in k]
    plain_keys = [k for k in final_sd if k.endswith('.weight') and 'parametrizations' not in k]
    print(f"  Clés parametrisées restantes : {len(param_keys)} (attendu : flow.wavenet + posterior_encoder.wavenet)")
    print(f"  Clés .weight simples         : {len(plain_keys)}")

    # 8. Sauvegarder
    output_file = OUTPUT_DIR / "model.safetensors"
    if output_file.exists():
        backup = OUTPUT_DIR / "model.safetensors.backup"
        if not backup.exists():
            shutil.copy2(str(output_file), str(backup))
            print(f"\n💾 Backup : {backup}")

    print(f"💾 Sauvegarde du modèle corrigé...")
    model.save_pretrained(str(OUTPUT_DIR))

    # Copier le tokenizer depuis le base model si nécessaire  
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))

    # 9. Vérification
    print("\n🔍 Vérification : rechargement et inférence...")
    model2 = VitsModel.from_pretrained(str(OUTPUT_DIR))

    inputs = tokenizer("ty boky misy ty tantaran'i jesosy kristy", return_tensors="pt")
    with torch.no_grad():
        output = model2(**inputs)

    waveform = output.waveform.squeeze().numpy()
    print(f"   Waveform shape : {waveform.shape}")
    print(f"   min={waveform.min():.4f}, max={waveform.max():.4f}, mean={waveform.mean():.4f}")

    if any(map(lambda x: x != x, [waveform.min(), waveform.max()])):
        print("   ❌ NaN détecté dans le waveform !")
    elif abs(waveform.max()) < 0.01:
        print("   ⚠️  Signal très faible")
    else:
        print("   ✅ Signal OK !")

        # Sauvegarder un échantillon de test
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        sample_file = SAMPLES_DIR / "test_fix_checkpoint.wav"
        sample_rate = model2.config.sampling_rate
        scipy.io.wavfile.write(str(sample_file), rate=sample_rate, data=waveform)
        print(f"\n🔊 Échantillon sauvegardé : {sample_file}")

    print("\n" + "=" * 65)
    print("🏁 Terminé ! Testez avec : ./test_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
