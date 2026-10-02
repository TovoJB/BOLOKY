import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_vezo"

# Output directories
AUDIO_RAW_DIR = DATA_DIR / "audio" / "raw"
AUDIO_SEGMENTS_DIR = DATA_DIR / "audio" / "segments"
TEXT_DIR = DATA_DIR / "parallel_text"
METADATA_DIR = DATA_DIR / "metadata"
RAW_JSON_DIR = DATA_DIR / "raw_markers"

# Base URLs
JW_BASE_URL = "https://www.jw.org"
MEDIA_API_URL = "https://b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS"

# Request Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 26 Bible Books of the Christian Greek Scriptures (New Testament) in Vezo & Malagasy
BIBLE_BOOKS = [
    {"num": 40, "symbol": "Mt", "slug_vz": "matio", "slug_mg": "matio", "name_vz": "Matio", "chapters": 28},
    {"num": 41, "symbol": "Mr", "slug_vz": "marka", "slug_mg": "marka", "name_vz": "Marka", "chapters": 16},
    {"num": 42, "symbol": "Lk", "slug_vz": "lioka", "slug_mg": "lioka", "name_vz": "Lioka", "chapters": 24},
    {"num": 43, "symbol": "Jn", "slug_vz": "jaona", "slug_mg": "jaona", "name_vz": "Jaona", "chapters": 21},
    {"num": 44, "symbol": "Ac", "slug_vz": "Asan%E2%80%99ny-Apostoly", "slug_mg": "Asan%E2%80%99ny-Apostoly", "name_vz": "Asan’ny Apostoly", "chapters": 28},
    {"num": 45, "symbol": "Ro", "slug_vz": "romanina", "slug_mg": "romanina", "name_vz": "Romanina", "chapters": 16},
    {"num": 46, "symbol": "1Co", "slug_vz": "1-korintianina", "slug_mg": "1-korintianina", "name_vz": "1 Korintianina", "chapters": 16},
    {"num": 47, "symbol": "2Co", "slug_vz": "2-korintianina", "slug_mg": "2-korintianina", "name_vz": "2 Korintianina", "chapters": 13},
    {"num": 48, "symbol": "Ga", "slug_vz": "galatianina", "slug_mg": "galatianina", "name_vz": "Galatianina", "chapters": 6},
    {"num": 49, "symbol": "Ef", "slug_vz": "efesianina", "slug_mg": "efesianina", "name_vz": "Efesianina", "chapters": 6},
    {"num": 50, "symbol": "Php", "slug_vz": "filipianina", "slug_mg": "filipianina", "name_vz": "Filipianina", "chapters": 4},
    {"num": 51, "symbol": "Col", "slug_vz": "kolosianina", "slug_mg": "kolosianina", "name_vz": "Kolosianina", "chapters": 4},
    {"num": 52, "symbol": "1Th", "slug_vz": "1-tesalonianina", "slug_mg": "1-tesalonianina", "name_vz": "1 Tesalonianina", "chapters": 5},
    {"num": 53, "symbol": "2Th", "slug_vz": "2-tesalonianina", "slug_mg": "2-tesalonianina", "name_vz": "2 Tesalonianina", "chapters": 3},
    {"num": 54, "symbol": "1Ti", "slug_vz": "1-timoty", "slug_mg": "1-timoty", "name_vz": "1 Timoty", "chapters": 6},
    {"num": 55, "symbol": "2Ti", "slug_vz": "2-timoty", "slug_mg": "2-timoty", "name_vz": "2 Timoty", "chapters": 4},
    {"num": 56, "symbol": "Tit", "slug_vz": "titosy", "slug_mg": "titosy", "name_vz": "Titosy", "chapters": 3},
    {"num": 57, "symbol": "Phm", "slug_vz": "filemona", "slug_mg": "filemona", "name_vz": "Filemona", "chapters": 1},
    {"num": 58, "symbol": "Heb", "slug_vz": "hebreo", "slug_mg": "hebreo", "name_vz": "Hebreo", "chapters": 13},
    {"num": 59, "symbol": "Jas", "slug_vz": "jakoba", "slug_mg": "jakoba", "name_vz": "Jakoba", "chapters": 5},
    {"num": 60, "symbol": "1Pe", "slug_vz": "1-petera", "slug_mg": "1-petera", "name_vz": "1 Petera", "chapters": 5},
    {"num": 61, "symbol": "2Pe", "slug_vz": "2-petera", "slug_mg": "2-petera", "name_vz": "2 Petera", "chapters": 3},
    {"num": 62, "symbol": "1Jn", "slug_vz": "1-jaona", "slug_mg": "1-jaona", "name_vz": "1 Jaona", "chapters": 5},
    {"num": 63, "symbol": "2Jn", "slug_vz": "2-jaona", "slug_mg": "2-jaona", "name_vz": "2 Jaona", "chapters": 1},
    {"num": 64, "symbol": "3Jn", "slug_vz": "3-jaona", "slug_mg": "3-jaona", "name_vz": "3 Jaona", "chapters": 1},
    {"num": 65, "symbol": "Jud", "slug_vz": "joda", "slug_mg": "joda", "name_vz": "Joda", "chapters": 1}
]

# Major study books & brochures with Vezo audio
AUDIO_PUBLICATIONS = {
    "lfb": {"title_vz": "Hahay Raha Maro ze Mianatsy Baiboly", "title_mg": "Mianara Amin’ny Tantaran’ny Baiboly", "has_audio": True},
    "ll":  {"title_vz": "Janjino Ndranahary le ho Velo tsy Misy Farany Iha", "title_mg": "Mihainoa An’Andriamanitra dia Hiaina Mandrakizay Ianao", "has_audio": True},
    "bh":  {"title_vz": "Ino Marina ro Ampianary ty Baiboly?", "title_mg": "Inona Marina no Ampianarin’ny Baiboly?", "has_audio": True},
    "bhs": {"title_vz": "Ino ty Azonao Ianara amy Baiboly Ao?", "title_mg": "Inona no Azonao Ianarana ao Amin’ny Baiboly?", "has_audio": True},
    "fg":  {"title_vz": "Vaovao Soa Baka Amy Ndranahary!", "title_mg": "Vaovao Tsara avy Amin’Andriamanitra!", "has_audio": True},
    "hf":  {"title_vz": "Afaky ho Sambatsy ty Fianakavianao", "title_mg": "Afaka Sambatra ny Fianakavianao", "has_audio": True},
    "ypq": {"title_vz": "Fanontanea 10 Mampiasa Loha ty Tanora", "title_mg": "Fanontaniana 10 Apetraky ny Tanora", "has_audio": True},
    "jlp": {"title_vz": "Jakoba: Lahilahy Tea Filongoa", "title_mg": "Jakoba: Lehilahy Tia Fihavanana", "has_audio": True}
}

# Text-only parallel publications
TEXT_PUBLICATIONS = [
    "lff", "rr", "wcg", "scl", "lv", "my", "th"
]
