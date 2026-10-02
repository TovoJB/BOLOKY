import sys
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Calculer la similarité cosinus entre deux mots")
parser.add_argument("mot1", help="Premier mot")
parser.add_argument("mot2", help="Deuxième mot")
parser.add_argument("--model", default="models/mg_fasttext.bin", help="Chemin du modèle (.bin ou .model)")
args = parser.parse_args()

# Gérer le chemin relatif
model_path = args.model
if not os.path.exists(model_path):
    model_path = os.path.join("..", model_path)

if not os.path.exists(model_path):
    print(f"Erreur : Modèle introuvable à {model_path}")
    sys.exit(1)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# === GENSIM / WORD2VEC ===
if model_path.endswith(".model") or model_path.endswith(".kv"):
    from gensim.models import Word2Vec, KeyedVectors
    
    if model_path.endswith(".model"):
        model = Word2Vec.load(model_path).wv
    else:
        model = KeyedVectors.load(model_path)
        
    try:
        # Word2Vec propose une fonction native très efficace
        sim = model.similarity(args.mot1, args.mot2)
        print(f"\nScore de similarité [Word2Vec]")
        print(f"{'-'*30}")
        print(f"'{args.mot1}' ↔ '{args.mot2}' : {sim:.4f}")
    except KeyError as e:
        print(f"Erreur : Le mot {e} n'est pas dans le vocabulaire Word2Vec.")
        sys.exit(1)

# === FASTTEXT ===
else:
    import fasttext
    
    # FastText est silencieux et ne crash pas si le mot est OOV (il crée un vecteur via les sous-mots)
    model = fasttext.load_model(model_path)
    
    vec1 = model.get_word_vector(args.mot1)
    vec2 = model.get_word_vector(args.mot2)
    
    sim = cosine_similarity(vec1, vec2)
    
    print(f"\nScore de similarité [FastText]")
    print(f"{'-'*30}")
    print(f"'{args.mot1}' ↔ '{args.mot2}' : {sim:.4f}")
