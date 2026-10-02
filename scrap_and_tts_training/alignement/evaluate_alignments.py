import os
import sys
sys.path.insert(0, "/home/tovo/Bureau/mms-tts-mlg/alignement/examples/mms/data_prep")

import json
import torch
import torchaudio
import argparse
from tqdm import tqdm
from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    load_model_dict,
)
import torchaudio.functional as F

SAMPLING_FREQ = 16000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_alignment_score(model, dictionary, audio_file, text, uroman_path, lang):
    # 1. Prepare audio
    try:
        waveform, sr = torchaudio.load(audio_file)
        if sr != SAMPLING_FREQ:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLING_FREQ)
        waveform = waveform.to(DEVICE)
    except Exception as e:
        return {"error": f"Audio load failed: {e}"}

    # 2. Prepare text
    try:
        norm_text = text_normalize(text.strip(), lang)
        # get_uroman_tokens expects a list of strings
        tokens = get_uroman_tokens([norm_text], uroman_path, lang)[0]
    except Exception as e:
        return {"error": f"Text processing failed: {e}"}

    # 3. Get Emissions
    try:
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions[0], dim=-1)
    except Exception as e:
        return {"error": f"Emission generation failed: {e}"}

    # 4. Forced Alignment Score
    try:
        token_indices = [dictionary[c] for c in tokens.split(" ") if c in dictionary]
        if not token_indices:
             return {"error": "Empty tokens after dict lookup"}

        blank = dictionary["<blank>"]
        targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
        
        input_lengths = torch.tensor([emissions.shape[0]], dtype=torch.int32)
        target_lengths = torch.tensor([targets.shape[0]], dtype=torch.int32)
        
        emissions_batch = emissions.unsqueeze(0)
        targets_batch = targets.unsqueeze(0)
        
        _, scores = F.forced_align(
            emissions_batch, targets_batch, input_lengths, target_lengths, blank=blank
        )
        
        # scores[0] is an array of scores for each token/frame, so we take the sum
        score = scores[0].sum().item()
        normalized_score = score / len(token_indices)
        
        return {
            "score": score,
            "normalized_score": normalized_score,
            "emissions_length": emissions.shape[0],
            "target_length": len(token_indices)
        }

    except Exception as e:
        return {"error": f"Alignment failed: {e}"}

def main(args):
    print("Loading model and dictionary...")
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    
    print(f"Reading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    results = []
    errors = []
    
    print(f"Processing {len(lines)} files...")
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) != 2:
            errors.append({"line": line, "error": "Invalid format"})
            continue
            
        rel_audio_path, text = parts
        audio_path = os.path.join(args.data_dir, rel_audio_path)
        
        if not os.path.exists(audio_path):
             errors.append({"line": line, "error": "Audio file not found"})
             continue
             
        res = compute_alignment_score(
            model, dictionary, audio_path, text, args.uroman_path, args.lang
        )
        
        res["audio_path"] = rel_audio_path
        res["text"] = text
        
        if "error" in res:
             errors.append(res)
        else:
             results.append(res)
             
    out_file = os.path.join(args.out_dir, "alignment_scores.json")
    print(f"\nSaving results to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": len(lines),
                "successful": len(results),
                "errors": len(errors)
            },
            "results": results,
            "errors": errors
        }, f, indent=2, ensure_ascii=False)
        
    if results:
         avg_norm_score = sum(r["normalized_score"] for r in results) / len(results)
         print(f"Average Normalized Score: {avg_norm_score:.4f}")
    print(f"Errors encountered: {len(errors)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/tovo/Bureau/mms-tts-mlg/data", help="Data directory")
    parser.add_argument("--uroman_path", default="/home/tovo/Bureau/mms-tts-mlg/alignement/uroman/bin", help="Path to uroman/bin")
    parser.add_argument("--lang", default="mlg", help="Language code")
    parser.add_argument("--out_dir", default="/home/tovo/Bureau/mms-tts-mlg/alignement", help="Output directory")
    args = parser.parse_args()
    main(args)
