import sys
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Faire des additions/soustractions sur les embeddings")
parser.add_argument("mot1", help="Premier mot")
parser.add_argument("operation", choices=['+', '-'], help="Opération (+ ou -)")
parser.add_argument("mot2", help="Deuxième mot")
parser.add_argument("--model", default="models/mg_fasttext.bin", help="Chemin du modèle")
args = parser.parse_args()

mot1 = args.mot1
operation = args.operation
mot2 = args.mot2

model_path = args.model
if not os.path.exists(model_path):
    model_path = os.path.join("..", model_path)

if not os.path.exists(model_path):
    print(f"Erreur : Modèle introuvable à {model_path}")
    sys.exit(1)

# === GENSIM / WORD2VEC ===
if model_path.endswith(".model") or model_path.endswith(".kv"):
    from gensim.models import Word2Vec, KeyedVectors
    
    if model_path.endswith(".model"):
        model = Word2Vec.load(model_path).wv
    else:
        model = KeyedVectors.load(model_path)
        
    # Gensim gère les analogies (addition/soustraction) nativement très rapidement !
    try:
        if operation == '+':
            # mot1 + mot2
            voisins = model.most_similar(positive=[mot1, mot2], topn=10)
        else:
            # mot1 - mot2
            voisins = model.most_similar(positive=[mot1], negative=[mot2], topn=10)
    except KeyError as e:
        print(f"Erreur : Un des mots n'est pas dans le vocabulaire Word2Vec. Détails: {e}")
        sys.exit(1)

    print(f"\nRésultat pour : {mot1} {operation} {mot2} [Gensim/Word2Vec]\n" + "-"*45)
    for mot, score in voisins:
        print(f"{mot:<20} {score:.4f}")

# === FASTTEXT ===
else:
    import fasttext
    
    try:
        print(f"Chargement du modèle... ({model_path})")
        model = fasttext.load_model(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    vec1 = model.get_word_vector(mot1)
    vec2 = model.get_word_vector(mot2)

    if operation == '+':
        vec_resultat = vec1 + vec2
    else:
        vec_resultat = vec1 - vec2

    vec_resultat = vec_resultat / np.linalg.norm(vec_resultat)

    print("Recherche des mots les plus proches (calcul matriciel en cours)...")
    mots = model.get_words()
    meilleurs = []
    
    for m in mots:
        if m == mot1 or m == mot2:
            continue
            
        v = model.get_word_vector(m)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        
        cos_sim = np.dot(vec_resultat, v) / norm
        meilleurs.append((cos_sim, m))

    meilleurs.sort(reverse=True)

    print(f"\nRésultat pour : {mot1} {operation} {mot2} [FastText]\n" + "-"*45)
    for i in range(10):
        score, m = meilleurs[i]
        print(f"{m:<20} {score:.4f}")
