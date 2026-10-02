import os
import torch
import scipy.io.wavfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTextToWaveform

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "mms-tts-mlg-base"
OUTPUT_DIR = BASE_DIR / "tts_samples"
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_speech(text, output_filename="sample_output.wav", model_path=MODEL_DIR):
    print(f"Loading MMS-TTS model from: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTextToWaveform.from_pretrained(str(model_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    
    print(f"Generating audio for text: \"{text}\"")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model(**inputs).waveform
        
    waveform = output.cpu().squeeze().numpy()
    sample_rate = model.config.sampling_rate
    
    out_path = OUTPUT_DIR / output_filename
    scipy.io.wavfile.write(str(out_path), rate=sample_rate, data=waveform)
    print(f"✓ Audio generated successfully saved to: {out_path} (Sample Rate: {sample_rate}Hz)")
    return out_path

if __name__ == "__main__":
    # Test sample with Malagasy and Vezo phrases
    test_phrase_mg = "Ny boky mirakitra ny tantaran'i Jesosy Kristy, zanak'i Davida."
    test_phrase_vz = "Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida."
    
    print("=== TEST 1: Malagasy officiel ===")
    generate_speech(test_phrase_mg, "test_malagasy_officiel.wav")
    
    print("\n=== TEST 2: Dialecte Vezo ===")
    generate_speech(test_phrase_vz, "test_vezo.wav")
