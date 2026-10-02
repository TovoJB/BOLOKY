#!/usr/bin/env python3
"""
03_evaluate.py
==============
Évaluation qualitative et quantitative du modèle FastText entraîné.

Évaluations proposées :
  1. Analogies morphologiques (malgache)
  2. Similarités de paires de mots
  3. Clustering des 500 mots les plus fréquents (t-SNE 2D)
  4. Couverture OOV (hors-vocabulaire) via sous-mots

Usage :
  python3 scripts/03_evaluate.py \
      --model   models/mg_fasttext.bin \
      --output_dir eval/
"""

import argparse
import json
import logging
from pathlib import Path

import fasttext
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Analogies malgaches : a est à b ce que c est à ?
# Exemples exploitant la morphologie par préfixes/suffixes
# ──────────────────────────────────────────────
ANALOGIES = [
    # (a, b, c, attendu)
    ("misotro", "mihinana", "manasotro", "mananina"),    # mi- → man-
    ("mianatra", "miasa",    "fianaran",  "fiasana"),    # mi- → fi-...-ana
    ("lehibe",   "kely",     "lava",      "fohy"),       # antonymes taille
    ("ray",      "reny",     "zanaky",    "zanaka"),
]

# Paires de mots pour test de similarité cosinus
WORD_PAIRS = [
    ("vady", "lehilahy"),      # mari – homme
    ("rano", "orana"),         # eau – pluie
    ("mianatra", "mahay"),     # apprendre – savoir
    ("maty", "velona"),        # mort – vivant
    ("tsara", "ratsy"),        # bon – mauvais
    ("loha", "tongotra"),      # tête – pied
    ("afo",  "rano"),          # feu – eau
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate_analogies(model, output_dir: Path) -> dict:
    log.info("\n── Analogies morphologiques ──")
    results = []
    correct = 0
    for a, b, c, expected in ANALOGIES:
        # vecteur(b) - vecteur(a) + vecteur(c)
        va, vb, vc = (
            model.get_word_vector(a),
            model.get_word_vector(b),
            model.get_word_vector(c),
        )
        query = vb - va + vc
        neighbors = model.get_nearest_neighbors_by_vector(query, k=5) \
                    if hasattr(model, "get_nearest_neighbors_by_vector") \
                    else model.get_nearest_neighbors(expected, k=1)
        top1 = neighbors[0][1] if neighbors else "?"
        ok = top1 == expected
        correct += int(ok)
        mark = "✓" if ok else "✗"
        log.info(f"  {mark} {a}:{b} :: {c}:{expected}  → prédit={top1}")
        results.append({"a": a, "b": b, "c": c, "expected": expected,
                        "predicted": top1, "correct": ok})

    acc = correct / len(ANALOGIES) * 100
    log.info(f"  Précision analogies : {acc:.1f}% ({correct}/{len(ANALOGIES)})")
    (output_dir / "analogies.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"analogy_accuracy": acc}


def evaluate_similarities(model, output_dir: Path) -> dict:
    log.info("\n── Similarités de paires ──")
    results = []
    for w1, w2 in WORD_PAIRS:
        v1, v2 = model.get_word_vector(w1), model.get_word_vector(w2)
        sim = cosine(v1, v2)
        log.info(f"  {w1:20s} ↔ {w2:20s}  sim={sim:.4f}")
        results.append({"w1": w1, "w2": w2, "cosine": round(sim, 4)})

    (output_dir / "similarities.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"pairs_evaluated": len(results)}


def evaluate_oov(model, output_dir: Path) -> dict:
    """Vérifie la capacité FastText à représenter des mots OOV via sous-mots."""
    log.info("\n── Couverture OOV (sous-mots) ──")
    # Mots dérivés qui pourraient ne pas être vus pendant l'entraînement
    oov_candidates = [
        "mihinanina",   # mi- + hinan- + ina
        "fampianarana", # famp- + ianaran + a
        "tsy mahavita",
        "mpampianatra",
        "masoandrotsika",
    ]
    results = []
    for w in oov_candidates:
        vec = model.get_word_vector(w)
        norm = float(np.linalg.norm(vec))
        in_vocab = w in model.get_words()
        log.info(f"  {'[VOCAB]' if in_vocab else '[OOV]  ':8s} {w:25s}  ||v||={norm:.3f}")
        results.append({"word": w, "in_vocab": in_vocab, "vector_norm": round(norm, 3)})

    (output_dir / "oov_coverage.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"oov_tested": len(oov_candidates)}


def nearest_neighbors_report(model, output_dir: Path) -> None:
    """Top-10 voisins pour un ensemble de mots-clés."""
    log.info("\n── Voisins proches ──")
    keywords = [
        "mianatra", "teny", "firenena", "trano", "rano",
        "misotro", "mandeha", "masoandro", "lehibe", "maty",
        "ianao"
    ]
    results = {}
    for w in keywords:
        neighbors = model.get_nearest_neighbors(w, k=10)
        fmt = [(round(sc, 4), nw) for sc, nw in neighbors]
        log.info(f"  {w:15s} → {fmt[:5]}")
        results[w] = fmt

    (output_dir / "nearest_neighbors.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_tsne(model, output_dir: Path, top_n: int = 500) -> None:
    """Génère une visualisation t-SNE des top_n mots les plus fréquents."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("sklearn / matplotlib non disponible – skip t-SNE")
        return

    log.info(f"\n── t-SNE ({top_n} mots) ──")
    words = model.get_words(include_freq=True)
    # Tri par fréquence décroissante
    words_sorted = sorted(words, key=lambda x: x[1], reverse=True)[:top_n]
    tokens = [w for w, _ in words_sorted]
    vectors = np.array([model.get_word_vector(w) for w in tokens])

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.6, color="#3d85c8")
    for i, tok in enumerate(tokens[:200]):   # étiquettes pour les 200 premiers
        ax.annotate(tok, coords[i], fontsize=5, alpha=0.75)
    ax.set_title(f"t-SNE – {top_n} mots les plus fréquents (Malgache FastText)")
    ax.axis("off")
    fig.tight_layout()
    out_png = output_dir / "tsne.png"
    fig.savefig(out_png, dpi=150)
    log.info(f"  Visualisation sauvegardée : {out_png}")
    plt.close(fig)


# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Évaluation FastText Malgache")
    parser.add_argument("--model",      default="models/mg_fasttext.bin")
    parser.add_argument("--output_dir", default="eval/")
    parser.add_argument("--tsne",       action="store_true", default=False,
                        help="Générer la visualisation t-SNE (lente)")
    parser.add_argument("--tsne_top",   type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Chargement du modèle : {args.model}")
    model = fasttext.load_model(args.model)
    log.info(f"Vocabulaire : {len(model.get_words()):,} mots | dim={model.get_dimension()}")

    metrics = {}
    metrics.update(evaluate_similarities(model, output_dir))
    metrics.update(evaluate_oov(model, output_dir))
    nearest_neighbors_report(model, output_dir)
    # Les analogies nécessitent get_nearest_neighbors_by_vector (fasttext >= 0.9.2)
    try:
        metrics.update(evaluate_analogies(model, output_dir))
    except AttributeError:
        log.warning("Analogies non disponibles (version fasttext trop ancienne)")

    if args.tsne:
        run_tsne(model, output_dir, top_n=args.tsne_top)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"\nRésumé sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()
