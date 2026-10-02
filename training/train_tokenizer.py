import os
import pandas as pd
import sentencepiece as spm
from pathlib import Path

# Dossiers à analyser (ajoute d'autres dossiers si besoin, ex: "./AT" s'il existe)
folders_to_scan = ["./dataset", "./NT", "./AT" ,"./NT_betsileo","./NT_antakarana"]

# Fichiers spécifiques à inclure (fichiers nettoyés)
files_to_scan = [
    "./scrapping/jw_corpus_cleaned.txt",
    "./scrapping/vaovao_corpus_cleaned.txt"
]

txt_path = "./corpus_for_tokenizer.txt"
vocab_size = 16000 # J'ai augmenté la taille car il y aura plus de données
model_prefix = "marian_malagasy"

print("=== ENTRAÎNEMENT D'UN NOUVEAU TOKENIZER (DONNÉES GLOBALES) ===")

# 2. Préparation des données textuelles
textes = []

print("1. Recherche et lecture de tous les fichiers .txt et .csv...")

for folder in folders_to_scan:
    if not os.path.exists(folder):
        print(f"  ⚠️  Le dossier {folder} n'existe pas, on l'ignore.")
        continue
        
    for path in Path(folder).rglob('*'):
        if path.is_file():
            if path.suffix == '.csv':
                try:
                    df = pd.read_csv(path)
                    for col in df.columns:
                        textes.extend(df[col].dropna().astype(str).str.lower().tolist())
                    print(f"  ✅ Chargé : {path} ({len(df)} lignes csv)")
                except Exception as e:
                    print(f"  ❌ Erreur de lecture CSV {path}: {e}")
                    
            elif path.suffix == '.txt':
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lignes = f.readlines()
                        textes.extend([l.strip().lower() for l in lignes if l.strip()])
                    print(f"  ✅ Chargé : {path} ({len(lignes)} lignes txt)")
                except Exception as e:
                    print(f"  ❌ Erreur de lecture TXT {path}: {e}")

print("\n1.5 Recherche et lecture des fichiers spécifiques nettoyés...")
for file_path in files_to_scan:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lignes = f.readlines()
                textes.extend([l.strip().lower() for l in lignes if l.strip()])
            print(f"  ✅ Chargé : {file_path} ({len(lignes)} lignes txt)")
        except Exception as e:
            print(f"  ❌ Erreur de lecture TXT {file_path}: {e}")
    else:
        print(f"  ⚠️  Le fichier {file_path} n'existe pas, on l'ignore.")

print(f"\n2. Écriture de {len(textes)} phrases dans le fichier texte brut temporaire...")
with open(txt_path, "w", encoding="utf-8") as f:
    for texte in textes:
        f.write(texte + "\n")

# 3. Entraînement de SentencePiece
print(f"\n3. Entraînement du modèle SentencePiece (vocab_size={vocab_size})... Cela peut prendre quelques instants.")

spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=1.0, 
    model_type="unigram",   
    pad_id=0,
    unk_id=1,
    bos_id=-1,              
    eos_id=2,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>"
)

print(f"\n✅ Terminé ! Le modèle a été sauvegardé sous les noms :")
print(f"  - {model_prefix}.model (le modèle binaire)")
print(f"  - {model_prefix}.vocab (le vocabulaire lisible)")

# Nettoyage du fichier texte temporaire
if os.path.exists(txt_path):
    os.remove(txt_path)
    print("Fichier temporaire supprimé.")
