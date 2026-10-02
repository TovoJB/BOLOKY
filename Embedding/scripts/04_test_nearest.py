import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Trouver les mots les plus proches")
parser.add_argument("mot", help="Le mot cible")
parser.add_argument("--model", default="models/mg_fasttext.bin", help="Chemin du modèle (.bin pour FastText, .model pour Word2Vec)")
args = parser.parse_args()

# Gérer les chemins relatifs
model_path = args.model
if not os.path.exists(model_path):
    model_path = os.path.join("..", model_path)

if not os.path.exists(model_path):
    print(f"Erreur : Modèle introuvable à {model_path}")
    sys.exit(1)

# Détection automatique du type de modèle selon l'extension
if model_path.endswith(".model") or model_path.endswith(".kv"):
    from gensim.models import Word2Vec, KeyedVectors
    
    # Chargement
    if model_path.endswith(".model"):
        model = Word2Vec.load(model_path).wv
    else:
        model = KeyedVectors.load(model_path)
        
    # Recherche
    try:
        voisins = model.most_similar(args.mot, topn=10)
    except KeyError:
        print(f"Le mot '{args.mot}' n'est pas dans le vocabulaire Word2Vec.")
        sys.exit(1)
        
    # Affichage
    print(f"\nLes 10 mots les plus proches de '{args.mot}' [Gensim/Word2Vec] :\n" + "-"*45)
    for mot, score in voisins:
        print(f"{mot:<20} {score:.4f}")

else:
    import fasttext
    
    # Chargement
    model = fasttext.load_model(model_path)
    
    # Recherche
    voisins = model.get_nearest_neighbors(args.mot, k=10)
    
    # Affichage
    print(f"\nLes 10 mots les plus proches de '{args.mot}' [FastText] :\n" + "-"*45)
    for score, mot in voisins:
        print(f"{mot:<20} {score:.4f}")
