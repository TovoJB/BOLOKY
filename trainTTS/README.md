# Dataset Scraper & Audio Segmenter : Dialecte Vezo (`skg-x-vz`) <-> Malagasy Officiel (`mg`)

Ce projet permet d'extraire, d'aligner et de formater automatiquement les données textuelles et audios en dialecte **Vezo** et **Malagasy officiel** depuis JW.org pour l'entraînement de modèles d'IA :
1. **Transcription Vocale (ASR)** : Whisper, MMS (Meta), Wav2Vec2.
2. **Synthèse Vocale (TTS)** : Piper, VITS, XTTS, Coqui.
3. **Traduction Automatique & LLM (MT)** : NMT, MarianMT, Llama/Gemma fine-tuning.

---

## 📁 Structure du Dataset généré (`dataset_vezo/`)

```text
dataset_vezo/
├── audio/
│   ├── raw/                 # Fichiers MP3 complets par chapitre / livre
│   └── segments/            # Clips WAV (22.05kHz 16-bit mono) découpés par verset / phrase
├── metadata/
│   ├── metadata_asr_bible.csv   # Format standard ASR (audio_path, duration, transcript, language)
│   └── metadata_tts_bible.csv   # Format standard LJSpeech / TTS (id | text | normalized_text)
├── parallel_text/
│   └── parallel_bible.csv       # Texte bilingue aligné (id, book, chapter, verse, malagasy_officiel, vezo)
└── raw_markers/                 # Métadonnées brutes avec timestamps millisecondes
```

---

## 🚀 Utilisation

### 1. Prérequis
- Python 3.8+
- `ffmpeg` (installé pour le découpage audio en WAV)
- `beautifulsoup4`, `requests`

### 2. Commandes disponibles

#### Télécharger et découper la Bible complète (26 livres, ~8000 versets + audio) :
```bash
python run_scraper.py --bible
```

#### Faire un test rapide (sur 1 seul livre biblique, ex: Matio) :
```bash
python run_scraper.py --bible --limit-books 1
```

#### Télécharger les livres d'étude et brochures audio (`lfb`, `ll`, `bh`, `bhs`, etc.) :
```bash
python run_scraper.py --books
```

#### Télécharger tout le dataset (Bible + Livres + Audio + Découpage WAV) :
```bash
python run_scraper.py --all
```

#### Télécharger uniquement les textes parallèles (rapide, sans télécharger les audios MP3) :
```bash
python run_scraper.py --all --no-audio
```

---

## 🎯 Formats des Données

### 1. Parallèle (`parallel_bible.csv`)
| id | book | chapter | verse | malagasy_officiel | vezo | has_audio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nwt_40_Mt_01_v001` | Matio | 1 | 1 | Ny boky mirakitra ny tantaran'i Jesosy Kristy... | Ty boky misy ty tantaran'i Jesosy Kristy... | True |

### 2. ASR (`metadata_asr_bible.csv`)
| audio_path | duration | transcript | language | source |
| :--- | :--- | :--- | :--- | :--- |
| `audio/segments/nwt_40_Mt_01_v001.wav` | 5.548 | Ty boky misy ty tantaran'i Jesosy Kristy... | skg-x-vz | nwt_Mt_1_1 |

### 3. TTS (`metadata_tts_bible.csv`)
```text
nwt_40_Mt_01_v001|Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida...|Ty boky misy ty tantaran'i Jesosy Kristy, anan'i Davida...
```
