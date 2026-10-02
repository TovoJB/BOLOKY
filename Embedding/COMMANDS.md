# Commandes – FastText Malgache

## 🔄 Pipeline complet

Exécute les 3 étapes automatiquement (prétraitement → entraînement → évaluation) :

```bash
bash run_pipeline.sh
```

Mode **rapide** pour tester (dim=100, 3 époques, résultats moins bons mais rapide) :

```bash
bash run_pipeline.sh --quick
```

---

## Étape par étape

---

### 1. Prétraitement — `01_preprocess.py`

Nettoie le corpus brut avant l'entraînement : normalisation, détachement ponctuation,
déduplication, puis division en corpus d'entraînement et de validation.

```bash
python3 scripts/01_preprocess.py \
    --input       all_dataFinal.txt \
    --output_dir  . \
    --min_len     3 \
    --max_len     512 \
    --deduplicate \
    --valid_ratio 0.05
```

**Fichiers générés :** `corpus.train.txt` et `corpus.valid.txt`

#### Paramètres

| Paramètre       | Valeur par défaut | Description |
|-----------------|-------------------|-------------|
| `--input`       | `all_dataFinal.txt` | Fichier texte brut source (une phrase par ligne) |
| `--output_dir`  | `.` | Dossier où écrire les corpus train/valid |
| `--min_len`     | `3` | Nombre **minimum** de tokens par ligne — élimine les phrases trop courtes (ex. titres d'une seule ligne) |
| `--max_len`     | `512` | Nombre **maximum** de tokens par ligne — élimine les très longs blocs de texte |
| `--deduplicate` | activé | Supprime les phrases en double exact dans tout le corpus |
| `--valid_ratio` | `0.05` | Proportion réservée à la validation (5 % → 1 ligne sur 20 va dans `corpus.valid.txt`) |

#### Ce que fait le nettoyage

1. **Normalisation Unicode NFC** — unifie les caractères accentués (é = e + accent → é)
2. **Suppression URLs** — retire `http://...` et `www....`
3. **Minuscules** — tout le texte passe en bas de casse
4. **Détachement ponctuation** — `mianatra,` → `mianatra` · `rano;` → `rano` · `trano!` → `trano`
5. **Suppression tokens parasites** — chiffres seuls (`1234`), ponctuation seule (`,` `;`)

---

### 2. Entraînement — `02_train_fasttext.py`

Entraîne un modèle FastText (skipgram) sur le corpus nettoyé.
FastText apprend des vecteurs de mots **et** de sous-mots (n-grammes de caractères),
ce qui est idéal pour le malgache, langue agglutinante riche en affixes
(`mi-`, `ma-`, `man-`, `-ana`, `-ina`...).

```bash
python3 scripts/02_train_fasttext.py \
    --input    corpus.train.txt \
    --output   models/mg_fasttext \
    --model    skipgram \
    --dim      300 \
    --epoch    10 \
    --lr       0.05 \
    --minCount 5 \
    --minn     2 \
    --maxn     6 \
    --thread   4 \
    2>&1 | tee logs/training.log
```

**Modèle sauvegardé :** `models/mg_fasttext.bin`

Suivre la progression en temps réel :

```bash
tail -f logs/training.log
```

#### Paramètres

| Paramètre     | Valeur | Description |
|---------------|--------|-------------|
| `--input`     | `corpus.train.txt` | Corpus d'entraînement (une phrase par ligne) |
| `--output`    | `models/mg_fasttext` | Préfixe du fichier de sortie (`.bin` ajouté automatiquement) |
| `--model`     | `skipgram` | Algorithme : **skipgram** prédit le contexte à partir d'un mot (meilleur pour les mots rares) · `cbow` prédit un mot à partir du contexte (plus rapide) |
| `--dim`       | `300` | Taille des vecteurs de mots — 300 est un standard (plus grand = plus précis mais plus lent et lourd) |
| `--epoch`     | `10` | Nombre de passes complètes sur le corpus — plus d'époques = meilleure qualité, mais plus long |
| `--lr`        | `0.05` | Taux d'apprentissage — contrôle la vitesse de mise à jour des poids (0.05 est la valeur recommandée pour skipgram) |
| `--minCount`  | `5` | Fréquence minimale d'un mot pour être inclus dans le vocabulaire — les mots rares (< 5 occurrences) sont ignorés |
| `--minn`      | `2` | Taille minimale des n-grammes de caractères — capture les préfixes courts `mi-`, `ma-` |
| `--maxn`      | `6` | Taille maximale des n-grammes de caractères — capture les suffixes longs `-ana`, `-ina`, `-itra` |
| `--thread`    | `4` | Nombre de threads CPU parallèles — adapter au nombre de cœurs disponibles |
| `2>&1 \| tee` | `logs/training.log` | Redirige toute la sortie vers l'écran **et** vers le fichier log simultanément |

#### Comprendre la progression affichée

```
Progress:  45.2%  words/sec/thread: 19500  lr: 0.027  avg.loss: 1.82  ETA: 0h12m30s
```

| Champ | Signification |
|-------|---------------|
| `Progress` | % du corpus traité sur toutes les époques |
| `words/sec/thread` | Vitesse de traitement par thread — ~18 000–20 000 est normal |
| `lr` | Taux d'apprentissage actuel (décroît linéairement vers 0) |
| `avg.loss` | Perte moyenne — doit **diminuer** au fil du temps (4 → 2 → 1.5...) |
| `ETA` | Temps restant estimé |

---

---

### 2.b. Entraînement Alternatif (Word2Vec) — `06_train_word2vec.py`

Entraîne un modèle Word2Vec classique (via Gensim) sur le corpus nettoyé. 
Contrairement à FastText, Word2Vec n'utilise pas les n-grammes (sous-mots), chaque mot est traité dans sa globalité.

```bash
python3 scripts/06_train_word2vec.py \
    --input corpus.train.txt \
    --output models/mg_word2vec.model \
    --sg 1 \
    --dim 300 \
    --epochs 10 \
    --workers 4
```

**Modèles sauvegardés :** `models/mg_word2vec.model` et `models/mg_word2vec.kv`

---

### 3. Évaluation — `03_evaluate.py`

Évalue la qualité du modèle entraîné sur plusieurs critères linguistiques.

```bash
python3 scripts/03_evaluate.py --model models/mg_fasttext.bin
```

**Résultats sauvegardés :** `eval/summary.json`

#### Ce que l'évaluation mesure

| Test | Description |
|------|-------------|
| **Similarités de paires** | Mesure si des mots sémantiquement proches ont des vecteurs proches (`rano` ↔ `orana`) |
| **Couverture OOV** | Vérifie que les mots hors-vocabulaire sont quand même représentés grâce aux sous-mots |
| **Voisins proches** | Affiche les 5 mots les plus proches de mots de test |
| **Analogies morphologiques** | Teste si le modèle a appris la structure : `misotro:mihinana :: manasotro:?` |

---

## Résumé des fichiers produits

```
Embedding/
├── corpus.train.txt      ← corpus nettoyé (95 %)
├── corpus.valid.txt      ← corpus de validation (5 %)
├── models/
│   └── mg_fasttext.bin   ← modèle entraîné (chargeable avec fasttext.load_model)
├── logs/
│   └── training.log      ← logs complets de l'entraînement
└── eval/
    └── summary.json      ← résultats de l'évaluation
```
---

## 🛠️ Tests Interactifs

Vous pouvez explorer les embeddings (FastText ou Word2Vec) directement en ligne de commande. Par défaut, les scripts utilisent le modèle FastText. Ajoutez l'argument `--model` pour utiliser Word2Vec.

**Trouver les mots les plus proches :**
```bash
# Avec FastText (par défaut)
python3 scripts/04_test_nearest.py ianao

# Avec Word2Vec
python3 scripts/04_test_nearest.py ianao --model models/mg_word2vec.model
```

**Faire des mathématiques sur les mots (Analogies) :**
```bash
# Addition (ex: humain + mâle)
python3 scripts/05_test_maths.py olona + lahy
python3 scripts/05_test_maths.py olona + lahy --model models/mg_word2vec.model

# Soustraction (ex: roi - mâle)
python3 scripts/05_test_maths.py mpanjaka - lahy
```

python3 scripts/04_test_nearest.py ianao --model models/mg_fasttext.bin

**Voir la relation (similarité) entre deux mots :**
```bash
# Avec FastText (par défaut)
python3 scripts/07_test_similarity.py rano orana

# Avec Word2Vec
python3 scripts/07_test_similarity.py rano orana --model models/mg_word2vec.model
```
python3 scripts/04_test_nearest.py ianao --model models/mg_word2vec.model



