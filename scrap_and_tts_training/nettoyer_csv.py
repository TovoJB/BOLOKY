#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de nettoyage et de filtrage pour motmalgache.csv

Règles de nettoyage appliquées :
1. Suppression des lignes vides ou sans contenu linguistique réel
   (ex: 'acquitter;;;;', 'aerostat;;;;', 'Aerva;;;nom;;;;;;;;;;;;Botanique: genre scientific' sans définition/traduction)
2. Suppression des entrées contenant des tags de désambiguïsation '#' (ex: #mg, #mg.n, #mg.pv)
3. Correction des mots commençant par 2 caractères successifs identiques (ex: 'aaloke' -> 'aloke')
4. Séparation automatique des mots collés :
   - Insertion d'un espace après chaque 'y' suivi d'une lettre (en malgache, 'y' est toujours en fin de mot)
   - Séparation des expressions composées (ex: 'ampahateloambynyfolony' -> 'ampahatelo amby ny folony')
5. Nettoyage des crochets résiduels '[' en fin de texte
6. Fusion intelligente et déduplication par mot unique sans perte de données
7. Export dans le fichier CSV nettoyé : motmalgache_clean.csv
"""

import os
import re
import csv
import sys

INPUT_CSV = "motmalgache.csv"
OUTPUT_CSV = "motmalgache_clean.csv"


def clean_bracket_tails(text):
    """Nettoie les résidus de crochets orphelins '[' et les indices de références"""
    if not text:
        return ""
    # Supprime les [ orphelins en fin de ligne
    text = re.sub(r"\s*\[\s*$", "", text)
    # Supprime les références incomplètes comme [1.23
    text = re.sub(r"\[\s*\d+(\.\d+)?(#\d+)?\s*$", "", text)
    return text.strip()


def unglue_words(text):
    """
    Sépare les mots malgaches collés sans espace :
    - En malgache, la lettre 'y' termine toujours un mot/morphème.
      Si un 'y' est suivi d'une lettre, on insère un espace.
    - Sépare les mots de liaison comme 'amby' dans les fractions et nombres.
    """
    if not text:
        return ""
    
    t = text
    # 1. Sépare 'amby' dans les nombres composés (ex: ampahateloambynyfolony -> ampahatelo amby nyfolony)
    t = re.sub(r"(?<=[a-zA-Z])(amby)(?=[a-zA-Z])", r" \1 ", t)

    # 2. Sépare après chaque 'y' suivi immédiatement d'une lettre (ex: amby ny, ny folony)
    t = re.sub(r"(?<=[a-zA-Z])y(?=[a-zA-Z])", "y ", t)

    # 3. Nettoie les espaces multiples
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_useful_content(row):
    """
    Vérifie si la ligne contient au moins un contenu linguistique significatif :
    traduction, définition, radical, exemple, synonyme, morphologie ou dérivé.
    Évite de conserver les mots orphelins (ex: 'acquitter', 'aerostat', 'Aerva').
    """
    # Indices des colonnes :
    # 1: radical, 6: traductions_fr, 7: def_mg, 8: def_fr, 9: def_en,
    # 10: synonymes_mg, 11: hyponymes_analogues, 12: exemples_mg, 13: morphologie, 14: derives
    indices_linguistiques = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for idx in indices_linguistiques:
        if idx < len(row) and row[idx].strip():
            return True
    return False


def merge_lists(val1, val2, sep=";"):
    """Fusionne deux chaînes représentant des listes séparées par un délimiteur"""
    items1 = [x.strip() for x in val1.split(sep) if x.strip()] if val1 else []
    items2 = [x.strip() for x in val2.split(sep) if x.strip()] if val2 else []
    
    seen = set()
    merged = []
    for it in items1 + items2:
        clean_it = clean_bracket_tails(it)
        if clean_it and clean_it.lower() not in seen:
            seen.add(clean_it.lower())
            merged.append(clean_it)
    return sep.join(merged)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_CSV)
    output_path = os.path.join(script_dir, OUTPUT_CSV)

    if not os.path.exists(input_path):
        print(f"[!] Erreur : Le fichier source '{input_path}' n'a pas été trouvé.")
        sys.exit(1)

    print(f"[*] Lecture et nettoyage de '{INPUT_CSV}'...")

    entries_by_word = {}
    total_lignes = 0
    suppr_vides = 0
    suppr_hash = 0
    modifs_doubles = 0
    modifs_espaces = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.reader(f_in, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            print("[!] Le fichier CSV est vide.")
            return

        for row in reader:
            if not row:
                continue
            total_lignes += 1

            # 1. Règle : Supprimer les lignes sans contenu linguistique réel (ex: 'acquitter', 'aerostat')
            if not has_useful_content(row):
                suppr_vides += 1
                continue

            # Récupération du mot
            mot = row[0].strip()

            # 2. Règle : Supprimer les mots contenant '#' (ex: aaloke#mg.n)
            if "#" in mot:
                suppr_hash += 1
                continue

            # 3. Règle : Si le mot commence par 2 caractères successifs identiques (ex: 'aaloke' -> 'aloke')
            if len(mot) >= 2 and mot[0].lower() == mot[1].lower():
                mot = mot[1:]
                modifs_doubles += 1

            # 4. Règle : Corriger les mots collés (séparation après 'y' et mots de liaison)
            mot_corrige = unglue_words(mot)
            if mot_corrige != mot:
                modifs_espaces += 1
                mot = mot_corrige

            # Nettoyage des colonnes individuelles
            row_data = {
                "mot": mot,
                "radical": unglue_words(row[1].strip()) if len(row) > 1 else "",
                "formation_derive": row[2].strip() if len(row) > 2 else "",
                "partie_du_discours": row[3].strip() if len(row) > 3 else "",
                "origine_dialecte": clean_bracket_tails(row[4].strip()) if len(row) > 4 else "",
                "etymologie": clean_bracket_tails(row[5].strip()) if len(row) > 5 else "",
                "traductions_fr": clean_bracket_tails(row[6].strip()) if len(row) > 6 else "",
                "definitions_mg": clean_bracket_tails(row[7].strip()) if len(row) > 7 else "",
                "definitions_fr": clean_bracket_tails(row[8].strip()) if len(row) > 8 else "",
                "definitions_en": clean_bracket_tails(row[9].strip()) if len(row) > 9 else "",
                "synonymes_mg": clean_bracket_tails(row[10].strip()) if len(row) > 10 else "",
                "hyponymes_analogues": clean_bracket_tails(row[11].strip()) if len(row) > 11 else "",
                "exemples_mg": clean_bracket_tails(row[12].strip()) if len(row) > 12 else "",
                "morphologie": row[13].strip() if len(row) > 13 else "",
                "derives": row[14].strip() if len(row) > 14 else "",
                "domaines_vocabulaire": clean_bracket_tails(row[15].strip()) if len(row) > 15 else ""
            }

            # 5. Fusion et déduplication par mot
            key = mot.lower()
            if key not in entries_by_word:
                entries_by_word[key] = row_data
            else:
                existing = entries_by_word[key]
                for col in ["radical", "formation_derive", "partie_du_discours", "morphologie", "derives", "etymologie"]:
                    if not existing[col] and row_data[col]:
                        existing[col] = row_data[col]

                existing["origine_dialecte"] = merge_lists(existing["origine_dialecte"], row_data["origine_dialecte"], sep=", ")
                existing["traductions_fr"] = merge_lists(existing["traductions_fr"], row_data["traductions_fr"], sep=", ")
                existing["definitions_mg"] = merge_lists(existing["definitions_mg"], row_data["definitions_mg"], sep=" | ")
                existing["definitions_fr"] = merge_lists(existing["definitions_fr"], row_data["definitions_fr"], sep=" | ")
                existing["definitions_en"] = merge_lists(existing["definitions_en"], row_data["definitions_en"], sep=" | ")
                existing["synonymes_mg"] = merge_lists(existing["synonymes_mg"], row_data["synonymes_mg"], sep=", ")
                existing["hyponymes_analogues"] = merge_lists(existing["hyponymes_analogues"], row_data["hyponymes_analogues"], sep=", ")
                existing["exemples_mg"] = merge_lists(existing["exemples_mg"], row_data["exemples_mg"], sep=" | ")
                existing["domaines_vocabulaire"] = merge_lists(existing["domaines_vocabulaire"], row_data["domaines_vocabulaire"], sep=", ")

    print(f"[+] Résumé du traitement :")
    print(f"    - Total lignes d'origine : {total_lignes}")
    print(f"    - Lignes vides / sans contenu supprimées : {suppr_vides}")
    print(f"    - Lignes avec '#' (#mg, etc.) supprimées : {suppr_hash}")
    print(f"    - Mots avec double lettre initiale corrigés : {modifs_doubles}")
    print(f"    - Mots composés collés séparés : {modifs_espaces}")
    print(f"    - Mots uniques finaux enregistrés : {len(entries_by_word)}")

    # Écriture du nouveau fichier CSV propre
    print(f"[*] Écriture du fichier filtré : '{OUTPUT_CSV}'...")
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.writer(f_out, delimiter=";", quoting=csv.QUOTE_MINIMAL)
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
            writer.writerow([
                item["mot"],
                item["radical"],
                item["formation_derive"],
                item["partie_du_discours"],
                item["origine_dialecte"],
                item["etymologie"],
                item["traductions_fr"],
                item["definitions_mg"],
                item["definitions_fr"],
                item["definitions_en"],
                item["synonymes_mg"],
                item["hyponymes_analogues"],
                item["exemples_mg"],
                item["morphologie"],
                item["derives"],
                item["domaines_vocabulaire"]
            ])

    print(f"[✔] Fichier créé avec succès : {output_path}")


if __name__ == "__main__":
    main()
