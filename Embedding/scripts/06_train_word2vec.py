#!/usr/bin/env python3
"""
06_train_word2vec.py
====================
Entraînement d'un modèle Word2Vec classique en utilisant la librairie Gensim.
À la différence de FastText, Word2Vec n'utilise pas de sous-mots (n-grams). 
Chaque mot est considéré dans son intégralité.
"""

import argparse
import logging
from pathlib import Path

try:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
except ImportError:
    print("Erreur : La librairie 'gensim' n'est pas installée.")
    print("Veuillez l'installer avec : pip install gensim")
    exit(1)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Entraînement de Word2Vec avec Gensim")
    parser.add_argument("--input", default="corpus.train.txt", help="Corpus d'entraînement (texte brut)")
    parser.add_argument("--output", default="models/mg_word2vec.model", help="Chemin du modèle de sortie")
    parser.add_argument("--sg", type=int, default=1, help="1 pour Skip-gram, 0 pour CBOW")
    parser.add_argument("--dim", type=int, default=300, help="Dimension des vecteurs (ex: 300)")
    parser.add_argument("--window", type=int, default=5, help="Taille de la fenêtre de contexte")
    parser.add_argument("--min_count", type=int, default=5, help="Fréquence minimale d'un mot")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de threads CPU")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques (itérations sur le corpus)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Le fichier {input_path} n'existe pas. Veuillez d'abord lancer le prétraitement.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Préparation de la lecture ligne par ligne depuis {args.input}...")
    # LineSentence lit le fichier à la volée, évitant de charger tout le texte en RAM
    sentences = LineSentence(str(input_path))

    log.info("Démarrage de l'entraînement Word2Vec...")
    log.info(f"Paramètres : dim={args.dim}, sg={'Skip-gram' if args.sg==1 else 'CBOW'}, epochs={args.epochs}")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        workers=args.workers,
        epochs=args.epochs
    )

    log.info(f"Sauvegarde du modèle dans {output_path}...")
    model.save(str(output_path))
    
    # On sauvegarde également uniquement les vecteurs finaux au format txt classique (optionnel)
    word_vectors_path = output_path.with_suffix('.kv')
    model.wv.save(str(word_vectors_path))
    
    log.info("Entraînement Word2Vec terminé avec succès !")
    log.info(f"Taille du vocabulaire appris : {len(model.wv)} mots.")

if __name__ == "__main__":
    main()
