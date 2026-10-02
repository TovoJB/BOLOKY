import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_sentence_vectors(sentence, model_path):
    """
    Extrait les vecteurs de chaque mot de la phrase et les affiche 
    graphiquement en 2D (via PCA) dans un même repère.
    """
    mots = sentence.split()
    
    # Gérer les chemins relatifs
    if not os.path.exists(model_path):
        model_path_rel = os.path.join("..", model_path)
        if os.path.exists(model_path_rel):
            model_path = model_path_rel

    if not os.path.exists(model_path):
        print(f"Erreur : Modèle introuvable à {model_path}")
        sys.exit(1)

    vectors = []
    valid_words = []
    
    # Chargement du modèle et récupération des vecteurs
    print(f"Chargement du modèle : {model_path} ...")
    if model_path.endswith(".model") or model_path.endswith(".kv"):
        from gensim.models import Word2Vec, KeyedVectors
        if model_path.endswith(".model"):
            model = Word2Vec.load(model_path).wv
        else:
            model = KeyedVectors.load(model_path)
            
        for mot in mots:
            if mot in model:
                vectors.append(model[mot])
                valid_words.append(mot)
            else:
                print(f"Mot '{mot}' ignoré (non trouvé dans le vocabulaire Word2Vec).")
                
    else:
        import fasttext
        # Pour éviter l'affichage de l'avertissement de chargement
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(model_path)
        
        for mot in mots:
            vectors.append(model.get_word_vector(mot))
            valid_words.append(mot)
            
    if len(valid_words) < 2:
        print("Erreur : Il faut au moins 2 mots valides dans la phrase pour générer un graphique 2D.")
        sys.exit(1)
        
    # Réduction de dimension à 2 dimensions avec PCA
    vectors = np.array(vectors)
    
    # On ajuste les dimensions pour PCA
    n_components = min(2, len(valid_words))
    pca = PCA(n_components=n_components)
    vectors_2d = pca.fit_transform(vectors)
    
    # Si on n'a que 2 mots, le PCA donnera peut-être 1 seule dimension avec de la variance
    # On force une forme (N, 2)
    if vectors_2d.shape[1] == 1:
        vectors_2d = np.hstack((vectors_2d, np.zeros((vectors_2d.shape[0], 1))))
        
    # Création du graphique
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], color='coral', s=100, edgecolor='red', zorder=5)
    
    # Ajouter les labels pour chaque mot
    for i, mot in enumerate(valid_words):
        plt.annotate(mot,
                     xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                     xytext=(7, 7),
                     textcoords='offset points',
                     ha='left',
                     va='bottom',
                     fontsize=12,
                     fontweight='bold',
                     color='darkblue')
                     
    # Ajouter des lignes repères (axes X et Y passant par l'origine)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
    
    plt.title(f"Représentation vectorielle des mots (PCA 2D)\nPhrase : \"{sentence}\"")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.grid(True, linestyle=':', alpha=0.6, zorder=0)
    
    # Sauvegarde ou affichage
    output_filename = "sentence_vectors.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n=> Graphique sauvegardé avec succès sous le nom : {output_filename}")
    
    # Si vous exécutez ceci dans un environnement avec écran graphique, vous pouvez décommenter :
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Afficher graphiquement les vecteurs des mots d'une phrase")
    parser.add_argument("phrase", help="La phrase à analyser (entre guillemets, ex: 'Manao ahoana ianao')")
    parser.add_argument("--model", default="models/mg_fasttext.bin", help="Chemin du modèle (.bin pour FastText, .model pour Word2Vec)")
    args = parser.parse_args()
    
    visualize_sentence_vectors(args.phrase, args.model)
