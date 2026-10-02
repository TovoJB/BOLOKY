import os
import re
import csv
import json
from pathlib import Path
from bs4 import BeautifulSoup
from config import (
    AUDIO_PUBLICATIONS, TEXT_PUBLICATIONS, JW_BASE_URL, MEDIA_API_URL,
    AUDIO_RAW_DIR, AUDIO_SEGMENTS_DIR, TEXT_DIR, METADATA_DIR, RAW_JSON_DIR
)
from utils import (
    init_directories, fetch_html, fetch_json, download_file,
    clean_text, parse_time_str, slice_audio
)

# Explicit TOC URL mappings for books and brochures
BOOK_TOC_URLS = {
    "lfb": "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/mianatsy-tantara-amy-baiboly/",
    "bh":  "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/ampianary-ty-baiboly/",
    "bhs": "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/fianara-baiboly/",
    "hf":  "https://www.jw.org/skg-x-vz/raha-fa-misy/bokikely/fianakavia-sambatsy/",
    "ypq": "https://www.jw.org/skg-x-vz/raha-fa-misy/bokikely/manontany-tanora/",
    "jl":  "https://www.jw.org/skg-x-vz/raha-fa-misy/bokikely/jehovah-raha-tea/",
    "yc":  "https://www.jw.org/skg-x-vz/raha-fa-misy/bokikely/ampianaro-ty-ananao/",
    "th":  "https://www.jw.org/skg-x-vz/raha-fa-misy/bokikely/vakiteny-noho-fampianara/",
    "wcg": "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/mila-herim-po-ty-miaraky-mandeha-amy-ndranahary/",
    "scl": "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/andininteny-manampy-amy-fiainantsika-kristiana/",
    "lff": "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/ho-azonao-tavy-ty-fiaina-zisiky-farany/",
    "rr":  "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/fivavahana-madio/",
    "lv":  "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/fitiava-ndranahary/",
    "my":  "https://www.jw.org/skg-x-vz/raha-fa-misy/boky/Boky-Misy-Tantara-Baka-Amy-Baiboly/"
}

def get_chapter_links_from_toc(toc_url):
    html = fetch_html(toc_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    chapter_urls = []
    
    # Path prefix to filter relevant chapter links
    parsed_path = toc_url.replace(JW_BASE_URL, "").rstrip("/")
    
    for a in soup.select(".toc a, #toc a, .synopsis a, .bodyTxt a"):
        href = a.get("href", "")
        if parsed_path in href and href != parsed_path and href != (parsed_path + "/"):
            full_url = JW_BASE_URL + href if href.startswith("/") else href
            if full_url not in chapter_urls:
                chapter_urls.append(full_url)
                
    return chapter_urls

def extract_paragraphs_from_soup(soup):
    paras = {}
    for p in soup.select("#article p[id^=\"p\"], #article div.par"):
        pid = p.get("id", "")
        if not pid:
            continue
        p_copy = BeautifulSoup(str(p), "html.parser").find(p.name)
        for bad in p_copy.select("a.footnoteLink, .superscript, .parabreak, .extRef"):
            bad.decompose()
        txt = clean_text(p_copy.get_text(" ", strip=True))
        if len(txt) > 5:
            paras[pid] = txt
    return paras

def scrape_parallel_text_for_book(pub_code, toc_url):
    chapter_urls = get_chapter_links_from_toc(toc_url)
    print(f"  Scraping parallel text: found {len(chapter_urls)} chapters...")
    
    parallel_rows = []
    
    for ch_idx, vz_ch_url in enumerate(chapter_urls, 1):
        html_vz = fetch_html(vz_ch_url)
        if not html_vz:
            continue
        soup_vz = BeautifulSoup(html_vz, "html.parser")
        
        # Get matching Malagasy URL via hreflang="mg"
        mg_link = soup_vz.find("link", attrs={"hreflang": "mg"})
        mg_ch_url = mg_link.get("href") if mg_link else None
        
        if not mg_ch_url:
            continue
            
        html_mg = fetch_html(mg_ch_url)
        if not html_mg:
            continue
        soup_mg = BeautifulSoup(html_mg, "html.parser")
        
        pvz = extract_paragraphs_from_soup(soup_vz)
        pmg = extract_paragraphs_from_soup(soup_mg)
        
        # Get chapter title
        h1_vz = soup_vz.find("h1")
        ch_title = clean_text(h1_vz.get_text(" ", strip=True)) if h1_vz else f"Chapter {ch_idx}"
        
        # Find common paragraph IDs
        common_ids = sorted(
            set(pvz.keys()).intersection(set(pmg.keys())),
            key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else 0
        )
        
        for pid in common_ids:
            parallel_rows.append({
                "id": f"{pub_code}_ch{ch_idx:03d}_{pid}",
                "pub_code": pub_code,
                "chapter_num": ch_idx,
                "chapter_title": ch_title,
                "para_id": pid,
                "malagasy_officiel": pmg[pid],
                "vezo": pvz[pid]
            })
            
    print(f"  Extracted {len(parallel_rows)} parallel paragraphs for [{pub_code}].")
    return parallel_rows

def process_audio_publication(pub_code, download_audio=True, slice_clips=True, scrape_text=True):
    init_directories()
    
    api_url = f"{MEDIA_API_URL}?pub={pub_code}&fileformat=MP3&output=json&alllangs=0&langwritten=VZ&txtCMSLang=VZ"
    media_data = fetch_json(api_url)
    
    pub_info = AUDIO_PUBLICATIONS.get(pub_code, {})
    pub_title = pub_info.get("title_vz", pub_code)
    
    print(f"\n=======================================================")
    print(f">>> PROCESSING BOOK: {pub_title} [{pub_code}]")
    print(f"=======================================================")
    
    # 1. Scrape parallel text if TOC is known
    parallel_rows = []
    if scrape_text and pub_code in BOOK_TOC_URLS:
        parallel_rows = scrape_parallel_text_for_book(pub_code, BOOK_TOC_URLS[pub_code])
        
    asr_rows = []
    tts_rows = []
    
    # 2. Process Audio if available
    if media_data and "files" in media_data:
        json_file = RAW_JSON_DIR / f"{pub_code}.json"
        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(media_data, jf, indent=2, ensure_ascii=False)
            
        mp3_files = media_data.get("files", {}).get("VZ", {}).get("MP3", [])
        print(f"\n  Downloading & indexing {len(mp3_files)} audio tracks...")
        
        for idx, item in enumerate(mp3_files, 1):
            track = item.get("track", idx)
            title = clean_text(item.get("title", f"Track {track}"))
            file_url = item.get("file", {}).get("url")
            duration = float(item.get("duration", 0.0))
            
            markers_obj = item.get("markers") or {}
            markers = markers_obj.get("markers", []) if isinstance(markers_obj, dict) else []
            
            raw_mp3_filename = f"{pub_code}_tr{track:03d}.mp3"
            raw_mp3_path = AUDIO_RAW_DIR / raw_mp3_filename
            
            if download_audio and file_url:
                download_file(file_url, raw_mp3_path)
                
            if markers:
                for m_idx, m in enumerate(markers, 1):
                    st = parse_time_str(m.get("startTime", "0"))
                    dur = parse_time_str(m.get("duration", "0"))
                    marker_num = m.get("verseNumber", m_idx)
                    
                    seg_filename = f"{pub_code}_tr{track:03d}_m{marker_num:03d}.wav"
                    seg_path = AUDIO_SEGMENTS_DIR / seg_filename
                    
                    if slice_clips and download_audio and raw_mp3_path.exists():
                        if slice_audio(raw_mp3_path, st, dur, seg_path):
                            seg_rel_path = f"audio/segments/{seg_filename}"
                            asr_rows.append({
                                "audio_path": seg_rel_path,
                                "duration": round(dur, 3),
                                "transcript": title,
                                "language": "skg-x-vz",
                                "source": f"{pub_code}_{track}_{marker_num}"
                            })
                            tts_rows.append({
                                "id": f"{pub_code}_tr{track:03d}_m{marker_num:03d}",
                                "text": title,
                                "normalized_text": title,
                                "audio_path": seg_rel_path
                            })
                print(f"    Track {track:03d} ({title[:35]}...): {len(markers)} markers cut")
            else:
                raw_rel_path = f"audio/raw/{raw_mp3_filename}"
                asr_rows.append({
                    "audio_path": raw_rel_path,
                    "duration": round(duration, 3),
                    "transcript": title,
                    "language": "skg-x-vz",
                    "source": f"{pub_code}_{track}"
                })
                tts_rows.append({
                    "id": f"{pub_code}_tr{track:03d}",
                    "text": title,
                    "normalized_text": title,
                    "audio_path": raw_rel_path
                })
                print(f"    Track {track:03d} ({title[:40]}...): {duration:.1f}s")
                
    return asr_rows, tts_rows, parallel_rows

def process_all_books(pubs_list=None, download_audio=True, slice_clips=True, scrape_text=True):
    init_directories()
    
    target_pubs = pubs_list if pubs_list else list(AUDIO_PUBLICATIONS.keys())
    
    all_asr = []
    all_tts = []
    all_parallel = []
    
    for pub_code in target_pubs:
        asr, tts, parallel = process_audio_publication(pub_code, download_audio, slice_clips, scrape_text)
        all_asr.extend(asr)
        all_tts.extend(tts)
        all_parallel.extend(parallel)
        
    # Save parallel books text
    if all_parallel:
        parallel_csv = TEXT_DIR / "parallel_books.csv"
        file_exists = parallel_csv.exists()
        with open(parallel_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "pub_code", "chapter_num", "chapter_title", "para_id", "malagasy_officiel", "vezo"])
            writer.writeheader()
            writer.writerows(all_parallel)
        print(f"\nSaved {len(all_parallel)} parallel book paragraphs to {parallel_csv}")
        
    # Save ASR & TTS metadata
    if all_asr:
        asr_csv = METADATA_DIR / "metadata_asr_books.csv"
        with open(asr_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "duration", "transcript", "language", "source"])
            writer.writeheader()
            writer.writerows(all_asr)
        print(f"Saved {len(all_asr)} ASR records to {asr_csv}")
        
    if all_tts:
        tts_csv = METADATA_DIR / "metadata_tts_books.csv"
        with open(tts_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            for row in all_tts:
                writer.writerow([row["id"], row["text"], row["normalized_text"]])
        print(f"Saved {len(all_tts)} TTS records to {tts_csv}")
        
    print(f"\n=== ALL BOOKS PROCESSED: {len(target_pubs)} publications ({len(all_parallel)} text pairs, {len(all_asr)} audio entries) ===")

if __name__ == "__main__":
    process_all_books(download_audio=True, slice_clips=True, scrape_text=True)
