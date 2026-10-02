import os
import re
import csv
import json
from pathlib import Path
from bs4 import BeautifulSoup
from config import (
    BIBLE_BOOKS, JW_BASE_URL, MEDIA_API_URL,
    AUDIO_RAW_DIR, AUDIO_SEGMENTS_DIR, TEXT_DIR, METADATA_DIR, RAW_JSON_DIR
)
from utils import (
    init_directories, fetch_html, fetch_json, download_file,
    clean_text, parse_time_str, slice_audio
)

def extract_chapter_verses_from_html(html):
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    verses = {}
    
    # JW Bible pages have verses in #bibleText span.verse
    for span in soup.select("#bibleText span.verse, span[id^=\"v\"]"):
        v_id = span.get("id", "")
        m = re.match(r"v\d{2}\d{3}(\d{3})", v_id)
        if not m:
            continue
        v_num = int(m.group(1))
        
        # Clone tag to avoid mutating
        span_copy = BeautifulSoup(str(span), "html.parser").find("span")
        for bad in span_copy.select("a, .superscript, .parabreak, .verseNum, .extRef"):
            bad.decompose()
            
        txt = clean_text(span_copy.get_text(" ", strip=True))
        txt = re.sub(r"^\d+\s*", "", txt) # remove leading verse number if any
        txt = clean_text(txt)
        if txt:
            verses[v_num] = txt
            
    return verses

def process_bible(download_audio=True, slice_clips=True, limit_books=None):
    init_directories()
    
    asr_rows = []
    tts_rows = []
    parallel_rows = []
    
    books_to_process = BIBLE_BOOKS[:limit_books] if limit_books else BIBLE_BOOKS
    
    print(f"=== PROCESSING BIBLE: {len(books_to_process)} BOOKS ===")
    
    for b_idx, book in enumerate(books_to_process, 1):
        book_num = book["num"]
        symbol = book["symbol"]
        slug_vz = book["slug_vz"]
        slug_mg = book["slug_mg"]
        name_vz = book["name_vz"]
        chapters_count = book["chapters"]
        
        print(f"\n[{b_idx}/{len(books_to_process)}] Processing {name_vz} ({symbol}) - {chapters_count} chapters...")
        
        for ch in range(1, chapters_count + 1):
            # 1. Fetch Vezo Media API metadata
            api_url = f"{MEDIA_API_URL}?booknum={book_num}&output=json&pub=nwt&fileformat=MP3&alllangs=0&track={ch}&langwritten=VZ&txtCMSLang=VZ"
            media_data = fetch_json(api_url)
            
            # Save raw json markers
            json_file = RAW_JSON_DIR / f"nwt_{book_num:02d}_{symbol}_{ch:02d}.json"
            if media_data:
                with open(json_file, "w", encoding="utf-8") as jf:
                    json.dump(media_data, jf, indent=2, ensure_ascii=False)
            
            # 2. Extract MP3 audio URL and Markers
            mp3_url = None
            markers = []
            if media_data and "files" in media_data:
                vz_files = media_data.get("files", {}).get("VZ", {}).get("MP3", [])
                if vz_files:
                    mp3_url = vz_files[0].get("file", {}).get("url")
                    markers = vz_files[0].get("markers", {}).get("markers", [])
            
            # Download MP3
            raw_mp3_path = AUDIO_RAW_DIR / f"nwt_{book_num:02d}_{symbol}_{ch:02d}.mp3"
            if download_audio and mp3_url:
                download_file(mp3_url, raw_mp3_path)
            
            # 3. Scrape Vezo text
            url_vz = f"{JW_BASE_URL}/skg-x-vz/raha-fa-misy/BAIBOLY/nwt/boky/{slug_vz}/{ch}/"
            html_vz = fetch_html(url_vz)
            verses_vz = extract_chapter_verses_from_html(html_vz)
            
            # 4. Scrape Malagasy text
            url_mg = f"{JW_BASE_URL}/mg/zavatra-misy/baiboly/nwt/boky/{slug_mg}/{ch}/"
            html_mg = fetch_html(url_mg)
            verses_mg = extract_chapter_verses_from_html(html_mg)
            
            # Match verses
            all_vnums = sorted(set(list(verses_vz.keys()) + list(verses_mg.keys())))
            markers_by_verse = {m.get("verseNumber"): m for m in markers if "verseNumber" in m}
            
            for v_num in all_vnums:
                v_vz = verses_vz.get(v_num, "")
                v_mg = verses_mg.get(v_num, "")
                
                # Slicing audio per verse if markers exist
                seg_rel_path = ""
                duration = 0.0
                if slice_clips and download_audio and raw_mp3_path.exists() and v_num in markers_by_verse:
                    m = markers_by_verse[v_num]
                    st = parse_time_str(m.get("startTime", "0"))
                    dur = parse_time_str(m.get("duration", "0"))
                    duration = dur
                    
                    seg_filename = f"nwt_{book_num:02d}_{symbol}_{ch:02d}_v{v_num:03d}.wav"
                    seg_path = AUDIO_SEGMENTS_DIR / seg_filename
                    
                    if slice_audio(raw_mp3_path, st, dur, seg_path):
                        seg_rel_path = f"audio/segments/{seg_filename}"
                        
                        # Add to ASR & TTS metadata if Vezo transcript exists
                        if v_vz:
                            asr_rows.append({
                                "audio_path": seg_rel_path,
                                "duration": round(duration, 3),
                                "transcript": v_vz,
                                "language": "skg-x-vz",
                                "source": f"nwt_{symbol}_{ch}_{v_num}"
                            })
                            
                            tts_rows.append({
                                "id": f"nwt_{book_num:02d}_{symbol}_{ch:02d}_v{v_num:03d}",
                                "text": v_vz,
                                "normalized_text": v_vz,
                                "audio_path": seg_rel_path
                            })
                
                # Add to Parallel Text Dataset
                if v_vz or v_mg:
                    parallel_rows.append({
                        "id": f"nwt_{book_num:02d}_{symbol}_{ch:02d}_v{v_num:03d}",
                        "book": name_vz,
                        "chapter": ch,
                        "verse": v_num,
                        "malagasy_officiel": v_mg,
                        "vezo": v_vz,
                        "has_audio": bool(seg_rel_path)
                    })
            
            print(f"  Ch.{ch:02d}: {len(all_vnums)} verses | Audio: {bool(mp3_url)} ({len(markers)} markers)")
            
    # Save CSV / JSONL files
    save_datasets(asr_rows, tts_rows, parallel_rows)
    print("\n=== BIBLE PROCESSING COMPLETED ===")

def save_datasets(asr_rows, tts_rows, parallel_rows):
    # 1. Parallel Bible CSV
    bible_csv = TEXT_DIR / "parallel_bible.csv"
    with open(bible_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "book", "chapter", "verse", "malagasy_officiel", "vezo", "has_audio"])
        writer.writeheader()
        writer.writerows(parallel_rows)
    print(f"Saved {len(parallel_rows)} parallel verses to {bible_csv}")
    
    # 2. ASR Metadata CSV
    asr_csv = METADATA_DIR / "metadata_asr_bible.csv"
    with open(asr_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "duration", "transcript", "language", "source"])
        writer.writeheader()
        writer.writerows(asr_rows)
    print(f"Saved {len(asr_rows)} ASR records to {asr_csv}")
    
    # 3. TTS Metadata CSV (LJSpeech format: id|text|normalized_text)
    tts_csv = METADATA_DIR / "metadata_tts_bible.csv"
    with open(tts_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for row in tts_rows:
            writer.writerow([row["id"], row["text"], row["normalized_text"]])
    print(f"Saved {len(tts_rows)} TTS records to {tts_csv}")

if __name__ == "__main__":
    # Test on first book (Matio)
    process_bible(download_audio=True, slice_clips=True, limit_books=1)
