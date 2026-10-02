#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scraper complet et optimisé pour motmalgache.org
Extrait l'ensemble des données du dictionnaire malgache :
- Mots, radicaux, morphologie, formation des dérivés
- Parties du discours, origines/dialectes [vezo, sakalava...], étymologie
- Traductions françaises, définitions (MG/FR/EN), synonymes, exemples
- Domaines de vocabulaire et indexations thématiques
"""

import os
import re
import sys
import json
import time
import random
import sqlite3
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://motmalgache.org"
DB_FILE = "motmalgache.db"
JSONL_FILE = "motmalgache_complet.jsonl"
CSV_FILE = "motmalgache_essentiel.csv"

# Configuration du respect du serveur (Rate Limiting)
MAX_WORKERS = 3           # Réduit le nombre de connexions simultanées
MIN_DELAY = 0.3           # Délai minimum en secondes entre chaque requête
MAX_DELAY = 0.7           # Délai maximum avec variation aléatoire (anti-blocage)
MAX_RETRIES = 3           # Nombre de tentatives en cas d'erreur réseau

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(HEADERS)


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Table de la file d'attente
    c.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            mot TEXT PRIMARY KEY,
            radical TEXT,
            source_index TEXT,
            scraped INTEGER DEFAULT 0
        )
    """)
    # Table des entrées détaillées
    c.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            mot TEXT PRIMARY KEY,
            radical TEXT,
            formation_derive TEXT,
            partie_du_discours TEXT,
            origine_dialecte TEXT,
            etymologie TEXT,
            traductions_fr TEXT,
            definitions_mg TEXT,
            definitions_fr TEXT,
            definitions_en TEXT,
            synonymes_mg TEXT,
            hyponymes_analogues TEXT,
            exemples_mg TEXT,
            morphologie TEXT,
            derives TEXT,
            domaines_vocabulaire TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\[\s*\d+(\.\d+)?(#\d+)?\s*\]", "", text)
    return text.strip()


def compute_derivation_formation(mot, radical):
    if not radical or radical == mot:
        return ""
    w = mot.lower()
    rad = radical.lower()

    if w.startswith("mand") and rad.startswith("l"):
        return f"man- + {radical} (l -> nd)"
    elif w.startswith("mandr") and rad.startswith("r"):
        return f"man- + {radical} (r -> ndr)"
    elif w.startswith("manj") and rad.startswith("z"):
        return f"man- + {radical} (z -> nj)"
    elif w.startswith("mamb") and rad.startswith("b"):
        return f"man- + {radical} (b -> mb)"
    elif w.startswith("mam") and len(rad) > 0 and rad[0] in "fpv":
        return f"man- + {radical} ({rad[0]} -> m)"
    elif w.startswith("man") and len(rad) > 0 and rad[0] in "kst":
        return f"man- + {radical} ({rad[0]} -> n)"
    elif w.startswith("mang") and rad.startswith("h"):
        return f"man- + {radical} (h -> ng)"
    elif w.startswith("mi"):
        return f"mi- + {radical}"
    elif w.startswith("maha"):
        return f"maha- + {radical}"
    elif w.startswith("mampi"):
        return f"mampi- + {radical}"
    elif w.startswith("mifan"):
        return f"mifan- + {radical}"
    elif w.startswith("f"):
        return f"f- (nominalisation) + {radical}"
    elif w.endswith("ana") or w.endswith("ina"):
        suffix = "ana" if w.endswith("ana") else "ina"
        return f"{radical} + -{suffix} (passif)"
    else:
        return f"affixe + {radical}"


def fetch_index_pages():
    """Récupère tous les mots des 8 sections principales"""
    print("[*] Collecte des index principaux...")
    words_found = set()

    indexes = [
        ("alphaLists", "/bins/alphaLists"),
        ("rootLists", "/bins/rootLists"),
        ("ethnicLists", "/bins/ethnicLists"),
        ("taxonLists", "/bins/taxonLists"),
        ("contextLists", "/bins/contextLists"),
        ("grammarLists", "/bins/grammarLists"),
        ("derivLists", "/bins/derivLists"),
        ("elements", "/bins/elements"),
    ]

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 1. Moissonner alphaLists et toutes ses tranches (range=...)
    try:
        r = session.get(BASE_URL + "/bins/alphaLists", timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        range_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "alphaLists" in href and "range=" in href:
                range_links.add(href if href.startswith("http") else BASE_URL + href)

        print(f"  -> {len(range_links)} tranches alphabétiques trouvées.")
        
        all_alpha_urls = [BASE_URL + "/bins/alphaLists"] + list(range_links)
        for idx_url in all_alpha_urls:
            try:
                res = session.get(idx_url, timeout=15)
                sp = BeautifulSoup(res.content, "html.parser")
                for a in sp.find_all("a", href=True):
                    href = a["href"]
                    if "/bins/teny2/" in href:
                        w = href.split("/bins/teny2/")[-1].strip()
                        w = urllib.parse.unquote(w)
                        if w and not w.startswith("#"):
                            words_found.add((w, "", "alphaLists"))
            except Exception as e:
                print(f"Erreur lecture {idx_url}: {e}")

    except Exception as e:
        print(f"Erreur alphaLists: {e}")

    # 2. Moissonner les autres listes
    for name, path in indexes[1:]:
        try:
            r = session.get(BASE_URL + path, timeout=20)
            soup = BeautifulSoup(r.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/bins/teny2/" in href:
                    w = href.split("/bins/teny2/")[-1].strip()
                    w = urllib.parse.unquote(w)
                    if w and not w.startswith("#"):
                        words_found.add((w, "", name))
        except Exception as e:
            print(f"Erreur index {name}: {e}")

    print(f"[+] Total mots uniques trouvés : {len(words_found)}")
    
    c.executemany("""
        INSERT OR IGNORE INTO queue (mot, radical, source_index, scraped)
        VALUES (?, ?, ?, 0)
    """, list(words_found))
    conn.commit()
    conn.close()


def parse_word_page(word):
    """Scrape et parse la fiche détaillée d'un mot avec pause et retry"""
    url = f"{BASE_URL}/bins/teny2/{urllib.parse.quote(word)}"
    
    # Pause aléatoire pour respecter le serveur
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    r = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                break
            elif r.status_code in [429, 503]:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(1 * (attempt + 1))

    if not r or r.status_code != 200:
        return None

    soup = BeautifulSoup(r.content, "html.parser")
    for s in soup(["script", "style", "form"]):
        s.extract()

    entry = {
        "mot": word,
        "radical": "",
        "formation_derive": "",
        "partie_du_discours": "",
        "origine_dialecte": [],
        "etymologie": "",
        "traductions_fr": [],
        "definitions_mg": [],
        "definitions_fr": [],
        "definitions_en": [],
        "synonymes_mg": [],
        "hyponymes_analogues": [],
        "exemples_mg": [],
        "morphologie": {},
        "derives": {},
        "domaines_vocabulaire": []
    }

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) >= 2:
            label = tds[0].get_text(strip=True)
            val = tds[1].get_text(separator=" ", strip=True)
            clean_val = re.sub(r"^\d+\s*", "", val)

            if "Entrée" in label and not entry["mot"]:
                entry["mot"] = clean_val.replace(" ", "")
            elif "Radical" in label and not entry["radical"]:
                entry["radical"] = clean_val.replace(" ", "")
            elif "Partie du discours" in label and not entry["partie_du_discours"]:
                entry["partie_du_discours"] = clean_val
            elif "Morphologie" in label:
                for m in re.finditer(r"(Présent|Passé|Futur|Impératif|Simple|Préfixée)\s*:\s*([^0-9\n\r]+)", val):
                    entry["morphologie"][m.group(1).lower()] = m.group(2).strip().replace("\n", " ")
            elif "Dérivés" in label:
                cat_matches = re.split(r"(Verbes actifs|Verbes passifs|Verbes circonstanciels|Noms|Adjectifs)\s*:", val)
                for i in range(1, len(cat_matches), 2):
                    cat_name = cat_matches[i].strip().lower().replace(" ", "_")
                    items = [clean_text(re.sub(r"^\d+\s*", "", x).replace(" ", "")).strip() for x in cat_matches[i+1].split(",") if x.strip()]
                    entry["derives"][cat_name] = [x for x in items if x]
            elif "Explications en malgache" in label:
                for d in re.split(r"\d+\s*", val):
                    d_clean = clean_text(d)
                    if d_clean and len(d_clean) > 2:
                        entry["definitions_mg"].append(d_clean)
            elif "Explications en français" in label:
                for d in re.split(r"\d+\s*", val):
                    d_clean = clean_text(d)
                    if d_clean and len(d_clean) > 2:
                        entry["definitions_fr"].append(d_clean)
                        # Traductions courtes
                        t_match = re.match(r"^([^(\[.;]+)", d_clean)
                        if t_match:
                            for tr_item in t_match.group(1).split(","):
                                tr_item = tr_item.strip()
                                if tr_item and len(tr_item) > 1 and tr_item not in entry["traductions_fr"]:
                                    entry["traductions_fr"].append(tr_item)
                        # Origines / Dialectes entre crochets
                        for dia in re.findall(r"\[([a-zA-Z\+\-\s]+)\]", d):
                            if dia not in entry["origine_dialecte"]:
                                entry["origine_dialecte"].append(dia)
                        # Étymologie
                        etym = re.findall(r"\((du [^\)]+|sanscrit[^\)]+|arabe[^\)]+|malais[^\)]+)\)", d, re.I)
                        if etym and not entry["etymologie"]:
                            entry["etymologie"] = ", ".join(etym)
            elif "Explications en anglais" in label:
                for d in re.split(r"\d+\s*", val):
                    d_clean = clean_text(d)
                    if d_clean and len(d_clean) > 2:
                        entry["definitions_en"].append(d_clean)
            elif "Exemples" in label:
                for ex in re.split(r"\d+\s*", val):
                    ex_clean = clean_text(ex)
                    if ex_clean and len(ex_clean) > 3:
                        entry["exemples_mg"].append(ex_clean)
            elif "Synonymes" in label:
                syns = [clean_text(re.sub(r"^\d+\s*", "", s)).strip() for s in val.split(",") if s.strip()]
                entry["synonymes_mg"].extend([s for s in syns if s])
            elif "Hyponymes" in label or "Analogues" in label:
                for it in re.split(r"\d+\s*", val):
                    it_clean = clean_text(it)
                    if it_clean and len(it_clean) > 1 and it_clean not in entry["hyponymes_analogues"]:
                        entry["hyponymes_analogues"].append(it_clean)
            elif "Vocabulaire" in label:
                if clean_val not in entry["domaines_vocabulaire"]:
                    entry["domaines_vocabulaire"].append(clean_val)

    # Formation morphologique du dérivé
    entry["formation_derive"] = compute_derivation_formation(entry["mot"], entry["radical"])

    return entry


def process_queue(max_workers=MAX_WORKERS):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT mot FROM queue WHERE scraped = 0")
    pending_words = [row[0] for row in c.fetchall()]
    conn.close()

    total = len(pending_words)
    print(f"[*] Début du scraping des fiches détaillées ({total} mots à traiter, {max_workers} threads, délai {MIN_DELAY}-{MAX_DELAY}s)...")

    count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {executor.submit(parse_word_page, w): w for w in pending_words}

        batch_entries = []
        batch_scraped_words = []

        for future in as_completed(future_to_word):
            word = future_to_word[future]
            try:
                data = future.result()
                if data:
                    batch_entries.append((
                        data["mot"],
                        data["radical"],
                        data["formation_derive"],
                        data["partie_du_discours"],
                        json.dumps(data["origine_dialecte"], ensure_ascii=False),
                        data["etymologie"],
                        json.dumps(data["traductions_fr"], ensure_ascii=False),
                        json.dumps(data["definitions_mg"], ensure_ascii=False),
                        json.dumps(data["definitions_fr"], ensure_ascii=False),
                        json.dumps(data["definitions_en"], ensure_ascii=False),
                        json.dumps(data["synonymes_mg"], ensure_ascii=False),
                        json.dumps(data["hyponymes_analogues"], ensure_ascii=False),
                        json.dumps(data["exemples_mg"], ensure_ascii=False),
                        json.dumps(data["morphologie"], ensure_ascii=False),
                        json.dumps(data["derives"], ensure_ascii=False),
                        json.dumps(data["domaines_vocabulaire"], ensure_ascii=False),
                        json.dumps(data, ensure_ascii=False)
                    ))
                batch_scraped_words.append((word,))
            except Exception as e:
                print(f"Erreur mot '{word}': {e}")

            count += 1
            if count % 50 == 0 or count == total:
                c_conn = sqlite3.connect(DB_FILE)
                c_cur = c_conn.cursor()
                if batch_entries:
                    c_cur.executemany("""
                        INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, batch_entries)
                    batch_entries.clear()
                if batch_scraped_words:
                    c_cur.executemany("UPDATE queue SET scraped = 1 WHERE mot = ?", batch_scraped_words)
                    batch_scraped_words.clear()
                c_conn.commit()
                c_conn.close()

                elapsed = time.time() - start_time
                speed = count / elapsed if elapsed > 0 else 0
                print(f"  [Progression] {count}/{total} mots traités ({count*100/total:.1f}%) - {speed:.1f} mots/sec")


CSV_FILE = "motmalgache.csv"


def export_data():
    """Exporte l'ensemble des données dans le fichier CSV final avec nettoyage et déduplication"""
    print(f"[*] Génération du fichier CSV final : {CSV_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT raw_json FROM entries")
    rows = c.fetchall()

    import csv
    entries_by_word = {}

    for (raw_json_str,) in rows:
        if not raw_json_str:
            continue
        item = json.loads(raw_json_str)
        mot = item.get("mot", "").strip()

        # 1. Ignorer les mots contenant '#' (ex: #mg.n, #mg.pv)
        if "#" in mot:
            continue

        # 2. Supprimer la première lettre si 2 caractères successifs identiques au début
        if len(mot) >= 2 and mot[0].lower() == mot[1].lower():
            mot = mot[1:]
            item["mot"] = mot

        key = mot.lower()
        if key not in entries_by_word:
            entries_by_word[key] = item

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f_csv:
        writer = csv.writer(f_csv, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        # En-têtes des colonnes
        writer.writerow([
            "mot",
            "radical",
            "formation_derive",
            "partie_du_discours",
            "origine_dialecte",
            "etymologie",
            "traductions_fr",
            "definitions_mg",
            "definitions_fr",
            "definitions_en",
            "synonymes_mg",
            "hyponymes_analogues",
            "exemples_mg",
            "morphologie",
            "derives",
            "domaines_vocabulaire"
        ])

        for key in sorted(entries_by_word.keys()):
            item = entries_by_word[key]

            # Formatage propre des champs complexes
            morpho_str = ", ".join([f"{k}: {v}" for k, v in item.get("morphologie", {}).items()]) if isinstance(item.get("morphologie"), dict) else ""
            
            derives_list = []
            if isinstance(item.get("derives"), dict):
                for cat, d_words in item.get("derives", {}).items():
                    if d_words:
                        derives_list.append(f"{cat}: {', '.join(d_words)}")
            derives_str = " | ".join(derives_list)

            writer.writerow([
                item.get("mot", ""),
                item.get("radical", ""),
                item.get("formation_derive", ""),
                item.get("partie_du_discours", ""),
                ", ".join(item.get("origine_dialecte", [])),
                item.get("etymologie", ""),
                ", ".join(item.get("traductions_fr", [])),
                " | ".join(item.get("definitions_mg", [])),
                " | ".join(item.get("definitions_fr", [])),
                " | ".join(item.get("definitions_en", [])),
                ", ".join(item.get("synonymes_mg", [])),
                ", ".join(item.get("hyponymes_analogues", [])),
                " | ".join(item.get("exemples_mg", [])),
                morpho_str,
                derives_str,
                ", ".join(item.get("domaines_vocabulaire", []))
            ])

    conn.close()
    print(f"[+] Export CSV terminé avec succès ({len(entries_by_word)} entrées uniques) : {CSV_FILE}")


if __name__ == "__main__":
    init_db()
    fetch_index_pages()
    process_queue(max_workers=MAX_WORKERS)
    export_data()
