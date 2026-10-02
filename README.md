# 🇲🇬 BOLOKY — Écosystème de Traduction Vocale Multidirectionnelle et Traitement des Dialectes Malgaches

---

## 🎥 Démonstration Vidéo

<!-- EMPLACEMENT VIDÉO DE DÉMONSTRATION -->
<div align="center">

[![Démonstration Vidéo du Projet BOLOKY](https://img.youtube.com/vi/VOTRE_ID_VIDEO/maxresdefault.jpg)](https://www.youtube.com/watch?v=VOTRE_ID_VIDEO)

> 📹 **Lien de la vidéo démo :** `[Insérer ici le lien YouTube / Google Drive / Loom de la démo vidéo]`  
> *(Remplacez l'URL ci-dessus par le lien final de votre démonstration)*

</div>

---

## 📑 Table des Matières
1. [Vue d'Ensemble & Emplacement des Composants](#-vue-densemble--emplacement-des-composants)
2. [Cadre Scientifique & Analyse Lexicostatistique des Dialectes](#-cadre-scientifique--analyse-lexicostatistique-des-dialectes)
   - [1. Cartographie Géolinguistique (60 points de collecte)](#1-cartographie-géolinguistique-60-points-de-collecte)
   - [2. Corrélation Distance Géographique vs Distance Lexicale](#2-corrélation-distance-géographique-vs-distance-lexicale)
   - [3. Hiérarchie de Centralité & Intercompréhension](#3-hiérarchie-de-centralité--intercompréhension)
3. [Création du Tokenizer Dédié & Morphologie Malgache](#-création-du-tokenizer-dédié--morphologie-malgache)
   - [Algorithme SentencePiece Unigram & Hyperparamètres](#1-algorithme-sentencepiece-unigram--hyperparamètres)
   - [Gestion Morphologique et Caractères Spéciaux](#2-gestion-morphologique-et-caractères-spéciaux)
   - [Visualisation du Tokenizer & Token Cloud](#3-visualisation-du-tokenizer--token-cloud)
4. [Espace d'Embedding & Représentation Vectorielle](#-espace-dembedding--représentation-vectorielle)
   - [Entraînement FastText & Word2Vec](#1-entraînement-fasttext--word2vec)
   - [Visualisation PCA 2D des Projections Vectorielles](#2-visualisation-pca-2d-des-projections-vectorielles)
   - [Évaluation Sémantique et Analogies Linguistiques](#3-évaluation-sémantique-et-analogies-linguistiques)
5. [Acquisition de Données & Génération Synthétique](#-acquisition-de-données--génération-synthétique)
   - [Web Scraping Ciblé](#1-web-scraping-ciblé)
   - [Génération par Alignement de Patterns Linguistiques & Révision Humaine](#2-génération-par-alignement-de-patterns-linguistiques--révision-humaine)
6. [Modèles de Traduction & Méthodologie d'Entraînement](#-modèles-de-traduction--méthodologie-dentraînement)
   - [Architecture MarianMT & Tokenizer Personnalisé](#1-architecture-marianmt--tokenizer-personnalisé)
   - [Intégration Meta NLLB-200](#2-intégration-meta-nllb-200)
   - [Stratégie de Fine-Tuning & Hyperparamètres](#3-stratégie-de-fine-tuning--hyperparamètres)
7. [Synthèse Vocale (TTS) & Reconnaissance Vocale (STT)](#-synthèse-vocale-tts--reconnaissance-vocale-stt)
   - [Vezo MMS-TTS Fine-Tuned & Traitement du Signal (DSP)](#1-vezo-mms-tts-fine-tuned--traitement-du-signal-dsp)
   - [Malagasy Officiel TTS (Meta MMS-1B)](#2-malagasy-officiel-tts-meta-mms-1b)
   - [English TTS (Kokoro ONNX)](#3-english-tts-kokoro-onnx)
   - [Reconnaissance Vocale Multilingue (MMS STT & Whisper)](#4-reconnaissance-vocale-multilingue-mms-stt--whisper)
8. [Architecture Logicielle & Déploiement](#-architecture-logicielle--déploiement)
9. [Structure du Dépôt](#-structure-du-dépôt)
10. [Guide de Démarrage](#-guide-de-démarrage)

---

## 🧭 Vue d'Ensemble & Emplacement des Composants

L'écosystème de recherche et de développement de **BOLOKY** s'articule autour des modules sources suivants sur la station de travail :

| Composant | Emplacement Source Local | Rôle & Fonctionnalité |
| :--- | :--- | :--- |
| **Backend API** | `/home/tovo/Bureau/crappingSianaka/backend` | Serveur FastAPI orchestrant l'inférence NMT, STT, TTS et la chaîne DSP |
| **Entraînement TTS & Alignement** | `/home/tovo/Bureau/scrap` | Pipelines de fine-tuning VITS/MMS, conversion discriminateur, scraping audio |
| **Création du Tokenizer** | `/home/tovo/Bureau/crappingSianaka/tokeniser` | Entraînement SentencePiece Unigram (32k vocab) et interface de tokenisation |
| **Embeddings & Évaluation** | `/home/tovo/Bureau/crappingSianaka/Embedding` | Entraînement FastText / Word2Vec, évaluation intrinsèque et projection PCA 2D |
| **Application Mobile** | `/home/tovo/Bureau/crappingSianaka/mobile_app` | Client Flutter Android (enregistrement, synthèse audio, DSP, 3 directions) |
| **Interface Web Démo** | `/home/tovo/Bureau/crappingSianaka/translation-ui` | Frontend Next.js / TailwindCSS avec visualisation 3D des tokens |

```
       ┌───────────────────────┐
       │   Mobile App Flutter  │ (Micro / Interface / DSP / Audio)
       └───────────┬───────────┘
                   │ HTTP / JSON / Audio WAV
                   ▼
       ┌───────────────────────┐
       │  FastAPI Backend :8000│
       ├───────────────────────┤
       │ • STT : MMS / Whisper │
       │ • NMT : NLLB / Marian │
       │ • TTS : MMS / Kokoro  │
       │ • DSP : Noise / Warmth│
       └───────────────────────┘
```

---

## 🔬 Cadre Scientifique & Analyse Lexicostatistique des Dialectes

Avant d'aborder la modélisation neuronale, une **étude lexicostatistique computationnelle** a été réalisée à l'échelle nationale pour quantifier scientifiquement les distances morphologiques, phonologiques et lexicales entre les dialectes malgaches.

---

### 1. Cartographie Géolinguistique (60 points de collecte)

<div align="center">
  <img src="photoAnalyse/dialect_map.png" alt="Cartographie des Dialectes Malgaches" width="85%"/>
</div>

#### Analyse & Interprétation :
- **Couverture géographique :** 60 points géolocalisés à travers Madagascar, couvrant 20 groupes ethnolinguistiques (*Merina, Betsileo, Vezo, Sakalava, Bara, Antanosy, Antandroy, Mahafaly, Masikoro, Mikea, Tsimihety, Sihanaka, Betsimisaraka, Antaimoro, Antaisaka, Antambahoaka, Antanalana, Tanala, Zafisoro, Nosy Boraha*).
- **Continuité linguistique :** L'analyse met en évidence un continuum linguistique austronésien, ponctué de variations phonologiques marquées le long du littoral ouest et sud (Vezo, Sakalava, Mahafaly, Tandroy).

---

### 2. Corrélation Distance Géographique vs Distance Lexicale

<div align="center">
  <img src="photoAnalyse/lexicostat_vs_geographic.png" alt="Distance Lexicostatistique vs Distance Géographique" width="85%"/>
</div>

#### Analyse & Interprétation :
- **Métrique computationnelle :** Distance de Levenshtein normalisée moyenne calculée sur des listes diagnostiques de Swadesh.
- **Résultats statistiques :**
  - Coefficient de corrélation linéaire robuste : **$r = 0.64$** ($p < 0.001$).
  - La divergence lexicale augmente régulièrement avec la distance kilométrique (de $0.26$ à courte distance jusqu'à $>0.50$ au-delà de 1 400 km).
- **Séparation des branches :**
  - **Points fuchsia (*Même branche*) :** Distance lexicostatistique faible ($0.10$ à $0.25$) même à des distances intermédiaires (100 à 400 km).
  - **Points cyan (*Branches différentes*) :** Forte dispersion ($0.30$ à $0.55$), matérialisant les écarts morphosyntaxiques entre les Hauts-Plateaux (Merina, Betsileo) et les dialectes côtiers (Vezo, Antandroy).

---

### 3. Hiérarchie de Centralité & Intercompréhension

<div align="center">
  <img src="photoAnalyse/ranking_barplot.png" alt="Classement de Centralité Lexicostatistique" width="85%"/>
</div>

#### Analyse & Interprétation :
- **Indicateur de centralité :** Distance moyenne de chaque dialecte par rapport à tous les autres (plus la note est basse, plus le dialecte est central et mutuellement intelligible).
- **Constats clés :**
  1. **Antananarivo (Merina) — $\mathbf{0.3037}$ :** Position la plus centrale de l'île, validant le choix du Malagasy Officiel comme langue pivot de traduction (*pivot language*).
  2. **Ambohimahasoa / Fianarantsoa / Ambositra (Betsileo) — $\mathbf{0.3149 - 0.3162}$ :** Forte proximité structurelle avec le Merina.
  3. **Vezo & Dialectes du Sud (Morombe, Toliara, Ambovombe) — $\mathbf{0.36 - 0.43}$ :** Éloignement maximal justifiant le développement de modèles neuronaux dédiés pour le dialecte Vezo.

---

## 🔤 Création du Tokenizer Dédié & Morphologie Malgache

La morphologie agglutinante du malgache (avec de multiples préfixes `m-`, `man-`, `mamp-`, infixes `-in-`, `-if-`, et suffixes `-ana`, `-ina`, `-y`) rend les tokenizers génériques inefficaces. Un tokenizer sur mesure a été conçu dans `/tokeniser`.

### 1. Algorithme SentencePiece Unigram & Hyperparamètres
Le script `tokeniser/traine/train_tokenizer.py` implémente un entraînement **SentencePiece Unigram** :
- **Fichier source :** Corpus nettoyé multivariétés `all_dataFinal.txt` (~366 Mo de texte malgache).
- **Taille de vocabulaire :** `vocab_size = 32000` sous-mots optimaux.
- **Couverture de caractères :** `character_coverage = 1.0` (100% des caractères et diacritiques malgaches préservés sans troncature).
- **Tokens spéciaux réservés :**
  - `<pad>` (ID: 0) : Padding pour batching uniforme.
  - `<unk>` (ID: 1) : Token inconnu de secours.
  - `<s>` (BOS) : Début de séquence.
  - `</s>` (EOS, ID: 2) : Fin de séquence.

```python
# Extrait du script train_tokenizer.py
spm.SentencePieceTrainer.train(
    input="./data/all_dataFinal.txt",
    model_prefix="boloky_tokenizer",
    vocab_size=32000,
    character_coverage=1.0,
    model_type="unigram",
    pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
    pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
)
```

### 2. Gestion Morphologique et Caractères Spéciaux
Le tokenizer décompose fidèlement les mots complexes en unités sous-lexicales linguistiquement pertinentes (ex: `fampianarana` $\rightarrow$ `_fampi`, `anarana`), facilitant le transfert d'apprentissage vers les dialectes où les racines sont conservées mais les affixes modifiés.

### 3. Visualisation du Tokenizer & Token Cloud

<!-- EMPLACEMENT PHOTO VISUALISATION TOKENIZER / NUAGE DE TOKENS -->
<div align="center">

![Nuage de Tokens 3D](web_ui/public/logo.png)

> 📸 **Emplacement Photo : Interface de Tokenisation & Nuage de Tokens 3D**  
> *(Interface interactive disponible dans `web_ui/src/app/components/TokenCloud3D.tsx` et `tokeniser/ui/app.py`)*

</div>

---

## 🌐 Espace d'Embedding & Représentation Vectorielle

Le dossier `/Embedding` contient le pipeline de vectorisation sémantique (`01_preprocess.py` à `08_visualize_sentence.py`).

### 1. Entraînement FastText & Word2Vec
- **FastText (Skipgram avec sous-mots n-grammes) :** Dimension vectorielle de 200/300 dimensions, capturant la sémantique des morphèmes malgaches même pour les mots rares ou dialectaux absents du vocabulaire principal.
- **Word2Vec (CBOW & Skip-gram) :** Modélisation des cooccurrences syntaxiques et contextuelles.

### 2. Visualisation PCA 2D des Projections Vectorielles

<div align="center">
  <img src="photoAnalyse/sentence_vectors.png" alt="Représentation vectorielle des mots (PCA 2D)" width="85%"/>
</div>

#### Interprétation de la Projection Vectorielle :
Sur la phrase malgache illustrative : *"mino aho fa mbola ho tsarany ny hoaviko"* :
- **Regroupement grammatical :** Les marqueurs temporels, de liaison et déterminants (`fa`, `ny`, `ho`, `mbola`) se projettent dans un cluster dense en haut à droite du repère PCA.
- **Séparation sémantique des actants et prédicats :** Les notions qualitatives et existentielles (`tsarany`, `hoaviko`, `mino`, `aho`) occupent des quadrants vectoriels distincts, reflétant leur divergence distributionnelle dans la grammaire VOS malgache (Verbe-Objet-Sujet).

### 3. Évaluation Sémantique et Analogies Linguistiques

<!-- EMPLACEMENT PHOTO EVALUATION VECTORIELLE -->
<div align="center">

> 📸 **Emplacement Photo : Matrice de Similarité Cosinus & Analogies Vectorielles Malgaches**  
> `[Insérer ici la photo des résultats d'évaluation intrinsèque générés par 03_evaluate.py]`

</div>

---

## 📊 Acquisition de Données & Génération Synthétique

Face au manque critique de données numériques pour les dialectes régionaux (*low-resource NLP*), deux piliers ont été déployés :

### 1. Web Scraping Ciblé (`scrapping/` & `scrap/`)
- `scraper_bible.py` & `scraper.py` : Extraction de corpus bibliques parallèles alignés verset par verset en Malagasy Officiel, Betsileo et Vezo (`ScriptureEarth`, `JW.org`).
- `scraper_motmalgache.py` : Scraping lexicographique et dictionnairique (~116 Mo SQLite `motmalgache.db`).
- `scraper_vaovao.py` : Extraction d'articles d'actualité en malgache pour l'entraînement non-supervisé.
- `clean_corpus.py` : Normalisation orthographique, suppression des balises et harmonisation des accents.

### 2. Génération par Alignement de Patterns Linguistiques & Révision Humaine
- **Moteur d'alignement de patterns :** Un algorithme d'alignement phonologique et syntaxique analyse les règles de transformation systématiques entre le Merina officiel et le dialecte Vezo (mutations consonantiques, pronoms spécifiques, marqueurs d'aspect).
- **Génération synthétique semi-supervisée :** Projection algorithmique de phrases officielles vers des candidats d'énoncés Vezo.
- **Validation humaine rigoureuse :** Correction manuelle intégrale par des locuteurs natifs pour certifier la conformité idiomatique et éliminer les faux positifs avant injection dans le corpus d'entraînement.

---

## 🧠 Modèles de Traduction & Méthodologie d'Entraînement

```
                  ┌───────────────────────────────┐
                  │           ENTRÉE              │
                  └──────┬─────────────────┬──────┘
                         │                 │
             [Malagasy / Vezo]        [English / French]
                         │                 │
                         ▼                 ▼
             ┌──────────────────┐  ┌──────────────────┐
             │ Meta NLLB-200    │  │ Helsinki MarianMT│
             │ (Distilled 600M) │  │ (opus_mt_en_mg)  │
             └─────────┬────────┘  └────────┬─────────┘
                       │                    │
                       └──────────┬─────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Modèle MarianMT Fine-Tuné│
                     │ (Merina ➔ Vezo / Betsileo│
                     └──────────────────────────┘
```

### 1. Architecture MarianMT & Tokenizer Personnalisé
- **Base :** `Helsinki-NLP/opus-mt` adapté au transfert de dialectes.
- **Fine-Tuning :** Entraîné sur les paires Merina $\rightarrow$ Vezo et Merina $\rightarrow$ Betsileo.

### 2. Intégration Meta NLLB-200
- **Modèle :** `facebook/nllb-200-distilled-600M`.
- **Rôle :** Traduction haute précision **Malagasy Officiel $\rightarrow$ Anglais** (`plt_Latn` / `mlg` vers `eng_Latn`), surpassant largement les traducteurs traditionnels grâce à son espace latent multilingue.

### 3. Stratégie de Fine-Tuning (`training/train_merina_to_vezo.py`)
- **Précision :** FP16 (Mixed Precision).
- **Optimisation VRAM :** Gradient Checkpointing activé.
- **Batching :** Batch size de 4 avec `gradient_accumulation_steps = 4` (taille effective de 16).
- **Taux d'apprentissage :** $5 \times 10^{-5}$ avec `weight_decay = 0.01` et early stopping sur `eval_loss`.

---

## 🔊 Synthèse Vocale (TTS) & Reconnaissance Vocale (STT)

### 1. Vezo MMS-TTS Fine-Tuned & Traitement du Signal (DSP)
- **Modèle :** VITS `facebook/mms-tts` fine-tuné sur le dialecte Vezo (`mms-tts-vezo-finetuned-v4` dans `/home/tovo/Bureau/scrap/models/`).
- **Pipeline DSP Audio en temps réel :**
  - **Filtre Passe-Haut Butterworth (150 Hz) :** Élimine les bruits de souffle et ronflements subsoniques.
  - **Noise Gate Adaptative :** Silence absolu entre les énoncés parlés.
  - **Égalisation Harmonique (Vocal Warmth) :** Présence vocale naturelle renforcée sur la bande 200 Hz – 3 kHz.
  - **Normalisation Peak/RMS :** Volume uniforme ($1.15\times$, 22.05 kHz).

### 2. Malagasy Officiel TTS (Meta MMS-1B)
- **Modèle :** `facebook/mms-1b-all` avec `target_lang="mlg"`.

### 3. English TTS (Kokoro ONNX)
- **Modèle :** `Kokoro-82M` ONNX haute fidélité (`af_sarah`), latence $<100$ ms.

### 4. Reconnaissance Vocale Multilingue (MMS STT & Whisper)
- **Malagasy & Dialectes :** `facebook/mms-1b-all` (Wav2Vec2ForCTC) avec chunking de 30s (`utils/mms_transcribeCPU.py`).
- **Anglais & Français :** OpenAI Whisper.

---

## 📱 Architecture Logicielle & Déploiement

### Backend (`/home/tovo/Bureau/crappingSianaka/backend`)
FastAPI asynchrone exposant :
- `POST /translate` : Traduction multi-directionnelle texte.
- `POST /speech-to-vezo` : Pipeline vocal unifié (STT $\rightarrow$ NMT $\rightarrow$ TTS).
- `POST /tts-vezo` : Synthèse vocale Vezo avec filtrage DSP.
- `POST /tts-english` : Synthèse vocale anglaise Kokoro.
- `POST /transcribe` : Transcription audio brute.

### Application Mobile (`/home/tovo/Bureau/crappingSianaka/mobile_app`)
- **Framework :** Flutter (Android).
- **Fonctionnalités :** 3 modes de traduction (`EN ➔ Vezo`, `EN ➔ Malagasy`, `Malagasy ➔ EN`), sélecteur d'IP dynamique, lecture audio `BytesSource`, contrôle de vitesse et activation/désactivation des filtres DSP.

---

## 📂 Structure du Dépôt `BOLOKY`

```
BOLOKY/
├── backend/                       # API FastAPI (main.py, filtres DSP, endpoints)
├── mobile_app/                    # Application mobile Flutter complète
├── training/                      # Scripts d'entraînement NMT MarianMT & runner GPU
├── tokeniser/                     # Scripts d'entraînement SentencePiece & UI
│   ├── traine/train_tokenizer.py  # Entraînement tokenizer 32k
│   └── ui/app.py                  # Interface de test de tokenisation
├── Embedding/                     # Pipeline d'embedding FastText & Word2Vec
│   ├── scripts/                   # Scripts de preprocessing, training et PCA 2D
│   └── eval/                      # Évaluation des similarités sémantiques
├── scrap_and_tts_training/        # Scripts de scraping et fine-tuning TTS Vezo VITS
├── scrapping/                     # Scrapers de corpus bilingues et nettoyage
├── utils/                         # Modules de transcription MMS STT
├── photoAnalyse/                  # Graphiques d'analyse lexicostatistique & embeddings
│   ├── dialect_map.png            # Carte des 60 points de collecte
│   ├── lexicostat_vs_geographic.png # Régression distance lexicale vs km (r=0.64)
│   ├── ranking_barplot.png        # Barplot de centralité dialectale
│   └── sentence_vectors.png       # Projection PCA 2D des mots malgaches
├── web_ui/                        # Interface web de démonstration Next.js
├── .gitignore                     # Exclusion des modèles lourds et gros datasets
└── README.md                      # Documentation scientifique et technique complète
```

---

## 🚀 Guide de Démarrage

### 1. Lancement du Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Lancement de l'Application Mobile
```bash
cd mobile_app
flutter pub get
flutter run
```

### 3. Entraînement du Tokenizer
```bash
cd tokeniser/traine
python3 train_tokenizer.py
```

### 4. Entraînement du Traducteur Merina ➔ Vezo
```bash
cd training
bash run_train_vezo.sh
```

---

<div align="center">
  <sub>Développé avec passion pour la préservation et la valorisation des langues et dialectes de Madagascar. 🇲🇬</sub>
</div>
