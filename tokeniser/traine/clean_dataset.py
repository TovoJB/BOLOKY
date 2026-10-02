import re
import os
import shutil

input_file = "./data/all_dataFinal.txt"
temp_output_file = "./data/all_dataFinal_clean.txt"

print(f"=== NETTOYAGE DU DATASET ===")
print(f"Fichier source : {input_file}")

if not os.path.exists(input_file):
    print("❌ Erreur: Le fichier source est introuvable.")
    exit(1)

# Cette expression régulière trouve tous les caractères qui NE SONT PAS :
# - Des lettres basiques (a-z, A-Z)
# - Des chiffres (0-9)
# - Des lettres accentuées courantes (À-ÿ, couvre le français et le malgache comme é, è, ô, ñ...)
# - Des espaces ou retours à la ligne (\s)
# - De la ponctuation standard
# Tout ce qui est emojis, caractères chinois, cyrilliques, arabes, etc. sera supprimé.
pattern = re.compile(r'[^a-zA-Z0-9\s\.,!\?:;\'"()\[\]{}<>\-/#&%*À-ÿ]')

total_lines = 0
removed_chars = set()

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(temp_output_file, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        total_lines += 1
        
        # Trouver les caractères bizarres pour les logger (optionnel, pour vérification)
        found_weird_chars = pattern.findall(line)
        if found_weird_chars:
            for c in found_weird_chars:
                if c.strip(): # On ignore les simples espaces
                    removed_chars.add(c)
        
        # Remplacer ces caractères bizarres par du vide
        cleaned_line = pattern.sub('', line)
        
        # On enlève les espaces multiples qui auraient pu se créer
        cleaned_line = re.sub(r' +', ' ', cleaned_line).strip()
        
        # Si la ligne n'est pas vide après le nettoyage, on la garde
        if cleaned_line:
            fout.write(cleaned_line + '\n')

print(f"\n✅ Nettoyage terminé sur {total_lines} lignes.")
if removed_chars:
    print(f"Exemples de caractères supprimés : {' '.join(list(removed_chars)[:50])} ...")

# On remplace l'ancien fichier par le nouveau fichier propre
print(f"\nRemplacement de l'ancien fichier...")
shutil.move(temp_output_file, input_file)

print(f"Terminé ! Le fichier {input_file} ne contient plus de caractères étrangers ni d'emojis.")
