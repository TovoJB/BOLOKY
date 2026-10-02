import re
import os

# --- Configuration ---
INPUT_FILE = "jw_corpus.txt"
OUTPUT_FILE = "jw_corpus_cleaned.txt"

# Regex pour détecter les liens (http, https, www)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Regex pour détecter les caractères arabes (Unicode blocks pour l'arabe)
# On peut aussi ajouter d'autres alphabets si besoin (Cyrillique, Chinois, etc.)
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

def clean_corpus(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Le fichier {input_path} n'existe pas encore.")
        return

    print(f"Début du nettoyage de {input_path}...")
    
    # Dictionnaire pour grouper les lignes par leur nombre de caractères
    # Format: { longueur: set(lignes) }
    lines_by_length = {}
    
    total_lines = 0
    removed_links = 0
    removed_arabic = 0
    removed_duplicates = 0
    
    unique_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            original_line = line.strip()
            
            if not original_line:
                continue

            # 1. Supprimer les liens
            line_no_url = URL_PATTERN.sub('', original_line).strip()
            if line_no_url != original_line:
                removed_links += 1
                
            if not line_no_url:
                continue

            # 2. Vérifier les caractères arabes (ou autres langues non désirées)
            if ARABIC_PATTERN.search(line_no_url):
                removed_arabic += 1
                continue

            # 3. Supprimer les doublons (en comparant d'abord la longueur)
            length = len(line_no_url)
            
            # Si cette longueur n'existe pas encore dans notre dictionnaire, on la crée
            if length not in lines_by_length:
                lines_by_length[length] = set()
            
            # On vérifie si la ligne existe déjà parmi les lignes de la même longueur
            if line_no_url in lines_by_length[length]:
                removed_duplicates += 1
            else:
                # C'est une nouvelle ligne unique
                lines_by_length[length].add(line_no_url)
                unique_lines.append(line_no_url)

    # Sauvegarde du résultat
    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in unique_lines:
            out_f.write(line + "\n")

    print("\n=== RAPPORT DE NETTOYAGE ===")
    print(f"Lignes totales analysées : {total_lines}")
    print(f"Lignes contenant des liens modifiées/supprimées : {removed_links}")
    print(f"Lignes avec caractères arabes supprimées : {removed_arabic}")
    print(f"Lignes doublons supprimées : {removed_duplicates}")
    print(f"-> Lignes finales conservées : {len(unique_lines)}")
    print(f"Fichier sauvegardé sous : {output_path}")

if __name__ == "__main__":
    clean_corpus(INPUT_FILE, OUTPUT_FILE)
