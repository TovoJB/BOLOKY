import os
import re
import requests
from bs4 import BeautifulSoup
import time

BOOKS = [
  { "name": "Matio", "ref": "TV-40-MAT-001.html", "dir": "01-Matio" },
  { "name": "Marka", "ref": "TV-41-MRK-001.html", "dir": "02-Marka" },
  { "name": "Lioka", "ref": "TV-42-LUK-001.html", "dir": "03-Lioka" },
  { "name": "Jaona", "ref": "TV-43-JHN-001.html", "dir": "04-Jaona" },
  { "name": "Asan'ny Apostoly", "ref": "TV-44-ACT-001.html", "dir": "05-Asa" },
  { "name": "Romana", "ref": "TV-45-ROM-001.html", "dir": "06-Romana" },
  { "name": "I Korintiana", "ref": "TV-46-1CO-001.html", "dir": "07-1Korintiana" },
  { "name": "II Korintiana", "ref": "TV-47-2CO-001.html", "dir": "08-2Korintiana" },
  { "name": "Galatiana", "ref": "TV-48-GAL-001.html", "dir": "09-Galatiana" },
  { "name": "Efesiana", "ref": "TV-49-EPH-001.html", "dir": "10-Efesiana" },
  { "name": "Filipiana", "ref": "TV-50-PHP-001.html", "dir": "11-Filipiana" },
  { "name": "Kolosiana", "ref": "TV-51-COL-001.html", "dir": "12-Kolosiana" },
  { "name": "I Tesaloniana", "ref": "TV-52-1TH-001.html", "dir": "13-1Tesaloniana" },
  { "name": "II Tesaloniana", "ref": "TV-53-2TH-001.html", "dir": "14-2Tesaloniana" },
  { "name": "I Timoty", "ref": "TV-54-1TI-001.html", "dir": "15-1Timoty" },
  { "name": "II Timoty", "ref": "TV-55-2TI-001.html", "dir": "16-2Timoty" },
  { "name": "Titosy", "ref": "TV-56-TIT-001.html", "dir": "17-Titosy" },
  { "name": "Filemona", "ref": "TV-57-PHM-001.html", "dir": "18-Filemona" },
  { "name": "Hebreo", "ref": "TV-58-HEB-001.html", "dir": "19-Hebreo" },
  { "name": "Jakoba", "ref": "TV-59-JAS-001.html", "dir": "20-Jakoba" },
  { "name": "I Petera", "ref": "TV-60-1PE-001.html", "dir": "21-1Petera" },
  { "name": "II Petera", "ref": "TV-61-2PE-001.html", "dir": "22-2Petera" },
  { "name": "I Jaona", "ref": "TV-62-1JN-001.html", "dir": "23-1Jaona" },
  { "name": "II Jaona", "ref": "TV-63-2JN-001.html", "dir": "24-2Jaona" },
  { "name": "III Jaona", "ref": "TV-64-3JN-001.html", "dir": "25-3Jaona" },
  { "name": "Joda", "ref": "TV-65-JUD-001.html", "dir": "26-Joda" },
  { "name": "Apokalypsy", "ref": "TV-66-REV-001.html", "dir": "27-Apokalipsy" },
]

BASE_URL = "https://www.scriptureearth.org/data/mlg/sab/sih/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
OUT_DIR = "./NT_Sihanaka"

def get_html(url):
    retries = 3
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                r.encoding = 'utf-8'
                return r.text
        except Exception as e:
            pass
        time.sleep(1)
    return None

def parse_verses(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', id='content')
    if not content:
        return {}

    verses = {}
    current_verse = None
    verse_text = []

    ignore_classes = {'s1', 's2', 'r', 'mt', 'b', 'reflink'}

    for el in content.descendants:
        if isinstance(el, str):
            text = el.strip()
            if text and current_verse is not None:
                parent_classes = []
                for p in el.parents:
                    if p.name == 'div' and p.get('id') == 'content':
                        break
                    parent_classes.extend(p.get('class', []))
                
                if not any(c in ignore_classes for c in parent_classes):
                    if 'v' not in el.parent.get('class', []) and 'c-drop' not in el.parent.get('class', []):
                        verse_text.append(text)
        elif el.name == 'div' and 'c-drop' in el.get('class', []):
            if current_verse:
                verses[current_verse] = " ".join(verse_text)
            current_verse = el.text.strip()
            verse_text = []
        elif el.name == 'span' and 'v' in el.get('class', []):
            if current_verse:
                verses[current_verse] = " ".join(verse_text)
            current_verse = el.text.strip()
            verse_text = []

    if current_verse:
        verses[current_verse] = " ".join(verse_text)

    cleaned = {}
    for v, text in verses.items():
        text = re.sub(r'\s+', ' ', text).strip().lower()
        cleaned[v] = text
    return cleaned

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    for book in BOOKS:
        book_dir = os.path.join(OUT_DIR, book['dir'])
        if not os.path.exists(book_dir):
            os.makedirs(book_dir)
            
        print(f"Scraping {book['name']}...")
        first_ref = book['ref']
        first_html = get_html(BASE_URL + first_ref)
        if not first_html:
            print(f"Failed to get {first_ref}")
            continue
            
        match = re.search(r'var chapters\s*=\s*\[.*?end:\s*(\d+).*?\];', first_html)
        if match:
            num_chapters = int(match.group(1))
        else:
            print(f"Could not find chapter count for {book['name']}, defaulting to 1")
            num_chapters = 1
            
        base_ref_match = re.search(r'var baseRef\s*=\s*"([^"]+)";', first_html)
        if base_ref_match:
            base_ref = base_ref_match.group(1)
        else:
            base_ref = first_ref.rsplit('-', 1)[0] + '-'
            
        for ch in range(1, num_chapters + 1):
            ch_url = f"{BASE_URL}{base_ref}{ch:03d}.html"
            html = get_html(ch_url)
            if not html:
                print(f"Failed to get chapter {ch} for {book['name']}")
                continue
                
            verses = parse_verses(html)
            out_file = os.path.join(book_dir, f"ch_{ch:02d}.txt")
            
            with open(out_file, 'w', encoding='utf-8') as f:
                for v_num, v_text in verses.items():
                    f.write(f"{v_num} {v_text}\n")
            print(f"Saved {out_file}")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
