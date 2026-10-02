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
1. [Vue d'Ensemble du Projet](#-vue-densemble-du-projet)
2. [Cadre Scientifique & Analyse Lexicostatistique des Dialectes](#-cadre-scientifique--analyse-lexicostatistique-des-dialectes)
   - [Cartographie Géolinguistique (60 points de collecte)](#1-cartographie-géolinguistique-60-points-de-collecte)
   - [Corrélation Distance Géographique vs Distance Lexicale](#2-corrélation-distance-géographique-vs-distance-lexicale)
   - [Hiérarchie de Centralité & Intercompréhension](#3-hiérarchie-de-centralité--intercompréhension)
3. [Acquisition de Données & Génération Synthétique](#-acquisition-de-données--génération-synthétique)
   - [Web Scraping Ciblé](#1-web-scraping-ciblé)
   - [Génération par Alignement de Patterns Linguistiques & Révision Humaine](#2-génération-par-alignement-de-patterns-linguistiques--révision-humaine)
4. [Modèles de Traduction & Méthodologie d'Entraînement](#-modèles-de-traduction--méthodologie-dentraînement)
   - [Architecture MarianMT & Tokenizer Personnalisé](#1-architecture-marianmt--tokenizer-personnalisé)
   - [Intégration Meta NLLB-200](#2-intégration-meta-nllb-200)
   - [Stratégie de Fine-Tuning & Hyperparamètres](#3-stratégie-de-fine-tuning--hyperparamètres)
5. [Synthèse Vocale (TTS) & Reconnaissance Vocale (STT)](#-synthèse-vocale-tts--reconnaissance-vocale-stt)
   - [Vezo MMS-TTS Fine-Tuned & Traitement du Signal (DSP)](#1-vezo-mms-tts-fine-tuned--traitement-du-signal-dsp)
   - [Malagasy Officiel TTS (Meta MMS-1B)](#2-malagasy-officiel-tts-meta-mms-1b)
   - [English TTS (Kokoro ONNX)](#3-english-tts-kokoro-onnx)
   - [Reconnaissance Vocale Multilingue (MMS STT & Whisper)](#4-reconnaissance-vocale-multilingue-mms-stt--whisper)
6. [Architecture Logicielle](#-architecture-logicielle)
7. [Structure du Dépôt](#-structure-du-dépôt)
8. [Guide de Démarrage](#-guide-de-démarrage)

---

## 🌍 Vue d'Ensemble du Projet

**BOLOKY** est une plateforme complète d'intelligence artificielle dédiée au traitement automatique des langues malgaches, de leurs dialectes régionaux (notamment le **Vezo**, le **Betsileo**, le **Sihanaka**) et de leur interconnexion avec l'**Anglais** et le **Français**.

Le système intègre :
- Une **application mobile Flutter** avec enregistrement micro, synthèse audio temps-réel, réglages DSP personnalisés et sélection dynamique de direction.
- Un **serveur FastAPI asynchrone** orchestrant les pipelines de Traduction Automatique (NMT), Reconnaissance Vocale (STT) et Synthèse Vocale (TTS).
- Une **interface web de recherche & démonstration** Next.js / TailwindCSS.

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

Avant d'aborder la phase d'apprentissage automatique, une **étude lexicostatistique computationnelle** rigoureuse a été menée sur l'ensemble du territoire de Madagascar afin de quantifier les distances morphologiques, phonologiques et lexicales entre les différents groupes ethniques et dialectes.

Les résultats de cette étude sont consignés dans le dossier `photoAnalyse/`.

---

### 1. Cartographie Géolinguistique (60 points de collecte)

![Cartographie des Dialectes Malgaches](photoAnalyse/dialect_map.png)

#### Description & Analyse :
- **Échantillonnage :** 60 points géographiques de collecte répartis sur les 22 régions de la Grande Île (de l'Antakarana au nord jusqu'à l'Antandroy/Tandroy au sud, du Vezo sur la côte ouest au Betsimisaraka sur la côte est).
- **Couverture ethnolinguistique :** 20 groupes distincts identifiés (*Merina, Betsileo, Vezo, Sakalava, Bara, Antanosy, Antandroy, Mahafaly, Masikoro, Mikea, Tsimihety, Sihanaka, Betsimisaraka, Antaimoro, Antaisaka, Antambahoaka, Antanalana, Tanala, Zafisoro, Nosy Boraha*).
- **Constat géolinguistique :** Madagascar présente une continuité dialectale remarquable (continuum linguistique austronésien), mais avec de fortes variations phonologiques locales le long des axes côtiers et des barrières topographiques centrales (Hauts-Plateaux).

---

### 2. Corrélation Distance Géographique vs Distance Lexicale

![Distance Lexicostatistique vs Distance Géographique](photoAnalyse/lexicostat_vs_geographic.png)

#### Description & Analyse :
- **Métrique utilisée :** Distance de Levenshtein normalisée moyenne calculée sur des listes de vocabulaire diagnostique (listes de Swadesh adaptées au malgache).
- **Résultats statistiques :**
  - Coefficient de corrélation linéaire : **$r = 0.64$** ($p < 0.001$).
  - La droite de régression montre une augmentation proportionnelle de la divergence lexicale avec l'éloignement kilométrique (de $0.26$ à proximité immédiate jusqu'à $>0.50$ à plus de 1 400 km).
- **Différenciation de branche :**
  - Les points **rose fuchsia** (*Même branche linguistique*) présentent une distance lexicostatistique systématiquement plus faible ($0.10$ à $0.25$) même à des distances géographiques modérées (100 à 400 km).
  - Les points **cyan** (*Branches différentes*) manifestent une dispersion plus importante ($0.30$ à $0.55$), soulignant les divergences morphologiques et structurales entre dialectes des plateaux (Merina, Betsileo) et dialectes côtiers (Vezo, Antandroy, Sakalava).

---

### 3. Hiérarchie de Centralité & Intercompréhension

![Classement de Centralité Lexicostatistique](photoAnalyse/ranking_barplot.png)

#### Description & Analyse :
- **Indicateur :** Distance lexicostatistique moyenne de chaque dialecte par rapport à l'ensemble des autres dialectes de l'île (plus la valeur est basse, plus le dialecte est central et proche de tous les autres).
- **Résultats clés :**
  1. **Antananarivo (Merina) — $\mathbf{0.3037}$ :** Position la plus centrale de l'île, confirmant scientifiquement la pertinence du Malagasy Officiel (standardisé sur la base du Merina) comme langue pivot de traduction (*pivot language*).
  2. **Ambohimahasoa / Fianarantsoa / Ambositra (Betsileo) — $\mathbf{0.3149 - 0.3162}$ :** Très forte proximité avec le Merina, justifiant une convergence linguistique rapide lors des entraînements neuronaux.
  3. **Côte Ouest & Sud (Vezo, Sakalava, Mahafaly, Antandroy) — $\mathbf{0.36 - 0.43}$ :** Les dialectes comme le Vezo (Morombe, Toliara, Maintirano) et l'Antandroy (Ambovombe, Tsihombe) présentent les plus fortes distances par rapport au standard Merina, démontrant la nécessité absolue de modèles de traduction et de synthèse vocale dédiés plutôt qu'une simple règle de substitution lexicale.

---

## 📊 Acquisition de Données & Génération Synthétique

Face à la pénurie critique de ressources textuelles numériques (*low-resource NLP*) pour les dialectes malgaches, une approche hybride d'acquisition a été mise en œuvre.

### 1. Web Scraping Ciblé
Les scripts situés dans le dossier `scrapping/` (`scraper.py`, `scraper_bts.py`, `scraper_jw.py`, `scraper_vaovao.py`) ont extrait des corpus textuels bilingues et multivariétés :
- **Textes bibliques parallèles :** Extraction des livres du Nouveau Testament en Merina standard, Betsileo et dialectes régionaux (`ScriptureEarth`, `JW.org`).
- **Articles d'actualités et publications locales :** Extraction et nettoyage automatisé du texte (`clean_corpus.py`) éliminant le balisage HTML, normalisant la ponctuation et harmonisant les diacritiques malgaches.

### 2. Génération par Alignement de Patterns Linguistiques & Révision Humaine
Pour enrichir les paires d'entraînement du dialecte **Vezo** :
- **Algorithme d'alignement de patterns :** Un moteur algorithmique propriétaire analyse les correspondances morphophonologiques systématiques entre le Merina officiel et le Vezo (ex: mutations consonantiques récurrentes, marqueurs aspectuels spécifiques, pronoms personnels et possessifs vezo).
- **Génération semi-supervisée :** L'algorithme génère des candidats de phrases en dialecte Vezo à partir d'énoncés en Malagasy officiel par projection de règles apprises.
- **Correction et validation humaine :** L'intégralité du corpus généré a été revue, annotée et corrigée manuellement par des locuteurs natifs pour éliminer les artéfacts de traduction et garantir l'authenticité idiomatique du dialecte Vezo.

---

## 🧠 Modèles de Traduction & Méthodologie d'Entraînement

Le sous-système de traduction repose sur une architecture multi-modèles adaptée aux différentes paires de langues.

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
- **Modèle de base :** `Helsinki-NLP/opus-mt` (architecture Transformer Seq2Seq).
- **Tokenisation SentencePiece :** Entraînement d'un tokenizer SentencePiece personnalisé (`train_tokenizer.py`, `train_with_new_tokenizer.py`) avec un vocabulaire étendu aux phonèmes et morphèmes spécifiques des dialectes malgaches (`marian_malagasy.vocab`, `marian_malagasy.model`).

### 2. Intégration Meta NLLB-200
- **Modèle :** `facebook/nllb-200-distilled-600M`.
- **Rôle :** Traduction de haute précision pour les directions complexes comme **Malagasy Officiel ➔ Anglais** (`plt_Latn` / `mlg` vers `eng_Latn`), surpassant largement les modèles bilingues standards grâce à son espace sémantique multilingue partagé.

### 3. Stratégie de Fine-Tuning & Hyperparamètres (`train_merina_to_vezo.py`)
L'entraînement du traducteur **Merina ➔ Vezo** a été optimisé pour un environnement GPU à mémoire limitée :
- **Précision :** FP16 (Mixed Precision).
- **Gradient Checkpointing :** Activé pour minimiser l'empreinte VRAM.
- **Taille de lot effective :** Batch size de 4 par GPU avec `gradient_accumulation_steps = 4` (équivalent à un batch effectif de 16).
- **Taux d'apprentissage :** $5 \times 10^{-5}$ avec décroissance de poids (*weight decay*) de 0.01.
- **Critère de sélection :** Évaluation par époque avec conservation du meilleur checkpoint sur la perte d'évaluation (`eval_loss`).

---

## 🔊 Synthèse Vocale (TTS) & Reconnaissance Vocale (STT)

Le pipeline audio assure une conversion bidirectionnelle voix $\leftrightarrow$ texte.

### 1. Vezo MMS-TTS Fine-Tuned & Traitement du Signal (DSP)
- **Modèle :** Modèle VITS `facebook/mms-tts` ré-entraîné et fine-tuné sur un dataset vocal Vezo annoté (`mms-tts-vezo-finetuned-v4`).
- **Chaîne de Traitement Audio Numérique (DSP Pipeline) :**
  Pour compenser les artéfacts acoustiques et restituer le timbre naturel :
  - **Filtre Passe-Haut Butterworth (150 Hz) :** Élimination des bruits de fond subsoniques et des bruits de souffle micro.
  - **Porte de Bruit Adaptative (*Noise Gate*) :** Suppression des bruits résiduels entre les segments parlés.
  - **Égalisation Harmonique (*Vocal Warmth*) :** Renforcement de la présence vocale (plage 200 Hz - 3 kHz).
  - **Normalisation Peak/RMS :** Harmonisation de la dynamique sonore avec vitesse d'élocution configurable ($1.15\times$, 22.05 kHz).

### 2. Malagasy Officiel TTS (Meta MMS-1B)
- **Modèle :** `facebook/mms-1b-all` configuré sur la langue cible `target_lang="mlg"`.
- Fournit une synthèse vocale fluide et naturelle pour le Malagasy standard.

### 3. English TTS (Kokoro ONNX)
- **Modèle :** `Kokoro-82M` au format ONNX avec quantification pour une latence inférieure à 100 ms sur CPU/GPU.
- Génère une voix anglaise haute fidélité (`af_sarah`).

### 4. Reconnaissance Vocale Multilingue (MMS STT & Whisper)
- **Malagasy & Dialectes :** `facebook/mms-1b-all` (Wav2Vec2ForCTC) avec traitement par fenêtrage de 30 secondes (`mms_transcribeCPU.py`).
- **Anglais & Français :** OpenAI Whisper pour une transcription robuste face aux accents et au bruit ambiant.

---

## 📱 Architecture Logicielle

### Backend (FastAPI — `backend/main.py`)
- **Endpoints RESTful :**
  - `POST /translate` : Traduction textuelle multi-directionnelle (EN $\rightarrow$ Vezo, EN $\rightarrow$ Merina, Merina $\rightarrow$ EN, etc.).
  - `POST /speech-to-vezo` : Pipeline unifié STT $\rightarrow$ NMT $\rightarrow$ TTS.
  - `POST /tts-vezo` : Synthèse vocale Vezo avec filtres DSP.
  - `POST /tts-english` : Synthèse vocale anglaise Kokoro.
  - `POST /transcribe` : Transcription audio pure (WAV/MP3).

### Mobile App (Flutter — `mobile_app/`)
- Architecture modulaire :
  - `lib/screens/ip_connect_screen.dart` : Découverte et connexion dynamique au serveur local.
  - `lib/screens/translation_screen.dart` : Interface 3 modes avec enregistreur audio intégré, lecteur audio `BytesSource`, contrôle de vitesse, switch de filtres DSP.
  - `lib/services/api_service.dart` : Client HTTP asynchrone avec gestion des timeouts et téléversement multipart.

---

## 📂 Structure du Dépôt

```
BOLOKY/
├── backend/                  # Serveur FastAPI
│   └── main.py              # Endpoints API, chargement des modèles, filtres DSP
├── mobile_app/               # Application mobile Flutter
│   ├── lib/
│   │   ├── main.dart        # Point d'entrée Flutter
│   │   ├── screens/         # Écrans de traduction et de connexion
│   │   ├── services/        # Service API client
│   │   └── utils/           # Thème et constantes
│   └── pubspec.yaml         # Dépendances Flutter
├── training/                 # Scripts d'entraînement
│   ├── train_merina_to_vezo.py       # Fine-tuning MarianMT Merina -> Vezo
│   ├── train_tokenizer.py            # Création du tokenizer SentencePiece
│   ├── train_with_new_tokenizer.py   # Entraînement avec nouveau vocabulaire
│   ├── run_train_vezo.sh             # Script de lancement GPU
│   └── start.sh                      # Script de démarrage des serveurs
├── scrapping/                # Pipeline de collecte et nettoyage de données
│   ├── scraper.py           # Scraper de textes dialectaux
│   ├── scraper_bts.py       # Scraper Betsileo
│   ├── scraper_jw.py        # Scraper de corpus parallèles
│   ├── scraper_vaovao.py    # Scraper d'actualités
│   └── clean_corpus.py      # Nettoyage et normalisation de corpus
├── utils/                    # Utilitaires STT & audio
│   ├── mms_transcribe.py    # Transcription MMS GPU
│   └── mms_transcribeCPU.py # Transcription MMS CPU fenêtrée
├── photoAnalyse/             # Résultats de l'étude lexicostatistique
│   ├── dialect_map.png                 # Carte des 60 points de collecte
│   ├── lexicostat_vs_geographic.png    # Régression distance lexicale vs géo
│   └── ranking_barplot.png             # Barplot de centralité dialectale
├── web_ui/                   # Interface web de démonstration (Next.js)
├── .gitignore                # Exclusion des modèles lourds et datasets
└── README.md                 # Documentation scientifique et technique
```

---

## 🚀 Guide de Démarrage

### 1. Prérequis
- Python 3.10+ (PyTorch, Transformers, FastAPI, Uvicorn, SoundFile, Librosa)
- Flutter SDK 3.0+
- Node.js 18+ (pour l'interface web)

### 2. Lancement du Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Lancement de l'Application Mobile
```bash
cd mobile_app
flutter pub get
flutter run
```

### 4. Lancement de l'Entraînement Traducteur
```bash
cd training
bash run_train_vezo.sh
```

---

<div align="center">
  <sub>Développé avec passion pour la préservation et la valorisation des langues et dialectes de Madagascar. 🇲🇬</sub>
</div>
