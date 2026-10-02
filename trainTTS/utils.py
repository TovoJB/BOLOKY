import os
import re
import json
import time
import subprocess
import urllib.request
from pathlib import Path
from config import HEADERS, DATA_DIR, AUDIO_RAW_DIR, AUDIO_SEGMENTS_DIR, TEXT_DIR, METADATA_DIR, RAW_JSON_DIR

def init_directories():
    for d in [AUDIO_RAW_DIR, AUDIO_SEGMENTS_DIR, TEXT_DIR, METADATA_DIR, RAW_JSON_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def fetch_html(url, retries=3, delay=1.0):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay * (i + 1))
            else:
                print(f"[ERROR] Failed to fetch {url}: {e}")
                return None

def fetch_json(url, retries=3, delay=1.0):
    content = fetch_html(url, retries, delay)
    if content:
        try:
            return json.loads(content)
        except Exception as e:
            print(f"[ERROR] JSON parse error from {url}: {e}")
    return None

def download_file(url, target_path, retries=3):
    if target_path.exists() and target_path.stat().st_size > 0:
        return True
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as out_f:
                out_f.write(resp.read())
            return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(2)
            else:
                print(f"[ERROR] Failed to download {url} -> {target_path}: {e}")
    return False

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[*+\u200b\u200e\u200f\ufeff]", "", text)
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_time_str(time_str):
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = float(parts[0]), float(parts[1])
            return m * 60 + s
        return float(time_str)
    except Exception:
        return 0.0

def slice_audio(input_mp3, start_sec, duration_sec, output_wav, sample_rate=22050):
    if output_wav.exists() and output_wav.stat().st_size > 1000:
        return True
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", str(input_mp3),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_wav)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception as e:
        print(f"[ERROR] ffmpeg slice error: {e}")
        return False
