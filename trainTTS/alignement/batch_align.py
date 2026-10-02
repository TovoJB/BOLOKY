import os
import sys
import glob
import json
import torch
import torchaudio
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/tovo/Bureau/mms-tts-mlg/alignement/examples/mms/data_prep")
from text_normalization import text_normalize
from align_utils import get_uroman_tokens, load_model_dict, get_spans
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)

def generate_emissions(model, waveform, total_duration):
    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)
            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            
            start_frame = int(SAMPLING_FREQ * input_start_time)
            end_frame = int(SAMPLING_FREQ * input_end_time)
            waveform_split = waveform[:, start_frame : end_frame]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[emission_start_frame - offset : emission_end_frame - offset, :]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)
    return emissions, stride

def merge_repeats(path, idx_to_token_map):
    from align_utils import Segment
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments

def get_alignments(waveform, total_duration, tokens, model, dictionary):
    emissions, stride = generate_emissions(model, waveform, total_duration)
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        token_indices = []
    blank = dictionary["<blank>"]
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor([emissions.shape[0]], dtype=torch.int32)
    target_lengths = torch.tensor([targets.shape[0]], dtype=torch.int32)
    path, _ = F.forced_align(
        emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
    )
    path = path.squeeze().to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride

def process_file(audio_path, text_path, out_wav_dir, model, dictionary, uroman_path, prefix):
    # 1. Read transcripts
    with open(text_path, 'r', encoding='utf-8') as f:
        transcripts = [line.strip() for line in f if line.strip()]
        
    if not transcripts:
        return []

    # 2. Normalize and get tokens
    norm_transcripts = [text_normalize(line, "mlg") for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, uroman_path, "mlg")

    # 3. Load Audio and resample for alignment
    try:
        orig_waveform, orig_sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []
        
    total_duration = orig_waveform.shape[1] / orig_sr
    
    # We resample for alignment, but we will slice the ORIGINAL audio (or resample it to 22050 for TTS)
    TARGET_TTS_SR = 22050
    if orig_sr != TARGET_TTS_SR:
        tts_waveform = F.resample(orig_waveform, orig_sr, TARGET_TTS_SR)
    else:
        tts_waveform = orig_waveform
        
    if orig_sr != SAMPLING_FREQ:
        align_waveform = F.resample(orig_waveform, orig_sr, SAMPLING_FREQ)
    else:
        align_waveform = orig_waveform
        
    align_waveform = align_waveform.to(DEVICE)

    # 4. Get Alignments
    try:
        segments, stride = get_alignments(align_waveform, total_duration, tokens, model, dictionary)
        spans = get_spans(tokens, segments)
    except Exception as e:
        print(f"Error aligning {audio_path}: {e}")
        return []

    # 5. Segment and save
    metadata_entries = []
    
    for i, t in enumerate(transcripts):
        if i >= len(spans):
            break
        span = spans[i]
        if not span:
            continue
            
        seg_start_idx = span[0].start
        seg_end_idx = span[-1].end

        audio_start_sec = seg_start_idx * stride / 1000.0
        audio_end_sec = seg_end_idx * stride / 1000.0 

        # Calculate frames in TTS waveform
        start_frame = int(audio_start_sec * TARGET_TTS_SR)
        end_frame = int(audio_end_sec * TARGET_TTS_SR)
        
        chunk = tts_waveform[:, start_frame:end_frame]
        
        # ID generation logic: e.g. AT-01-Genesisy-ch_01-T0013015.wav
        # We'll use start time in ms as the unique identifier
        ms_id = int(audio_start_sec * 1000)
        file_name = f"{prefix}-T{ms_id:07d}.wav"
        out_file = os.path.join(out_wav_dir, file_name)
        
        torchaudio.save(out_file, chunk, TARGET_TTS_SR)
        
        metadata_entries.append(f"wavs/{file_name}|{t}")
        
    return metadata_entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_brute", default="/home/tovo/Bureau/mms-tts-mlg/dataBrute", help="Raw dataset dir")
    parser.add_argument("--out_dir", default="/home/tovo/Bureau/mms-tts-mlg/data_aligned", help="Output directory")
    parser.add_argument("--uroman_path", default="/home/tovo/Bureau/mms-tts-mlg/alignement/uroman/bin", help="Path to uroman/bin")
    args = parser.parse_args()

    audio_base = os.path.join(args.data_brute, "BibleAudioData")
    text_base = os.path.join(args.data_brute, "BibleTextData")
    
    out_wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(out_wav_dir, exist_ok=True)
    
    print("Loading model and dictionary...")
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    
    all_metadata = []
    
    # Find all mp3 files
    mp3_files = glob.glob(os.path.join(audio_base, "**", "*.mp3"), recursive=True)
    
    print(f"Found {len(mp3_files)} audio files to process.")
    
    for audio_path in tqdm(mp3_files):
        # Resolve text path
        # Audio: .../BibleAudioData/AT/01-Genesisy/ch_01.mp3
        # Text:  .../BibleTextData/AT/01-Genesisy/ch_01.txt
        rel_path = os.path.relpath(audio_path, audio_base)
        text_path = os.path.join(text_base, rel_path).replace(".mp3", ".txt")
        
        if not os.path.exists(text_path):
            print(f"Text file not found for {audio_path}")
            continue
            
        # Create a prefix: AT-01-Genesisy-ch_01
        parts = rel_path.replace(".mp3", "").split(os.sep)
        prefix = "-".join(parts)
        
        entries = process_file(audio_path, text_path, out_wav_dir, model, dictionary, args.uroman_path, prefix)
        all_metadata.extend(entries)
        
    # Write metadata.csv
    meta_file = os.path.join(args.out_dir, "metadata.csv")
    with open(meta_file, 'w', encoding='utf-8') as f:
        for entry in all_metadata:
            f.write(entry + "\n")
            
    print(f"Alignment complete. Generated {len(all_metadata)} segments.")
    print(f"Dataset saved in {args.out_dir}")

if __name__ == "__main__":
    main()
