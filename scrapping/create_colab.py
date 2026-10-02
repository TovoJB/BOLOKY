import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Entraînement du modèle de traduction Merina -> Betsileo\n",
                "Bienvenue sur Google Colab ! Ce notebook est configuré pour entraîner rapidement votre modèle de traduction avec la puissance des GPUs de Google.\n",
                "\n",
                "### Étape 1 : Activer le GPU\n",
                "Allez dans le menu **Exécution > Modifier le type d'exécution**, et choisissez **T4 GPU** (ou GPU) comme accélérateur matériel, puis cliquez sur Enregistrer."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Étape 2 : Importer votre dataset\n",
                "Exécutez la cellule ci-dessous pour uploader votre fichier `parallel_dataset.csv` que vous avez généré sur votre ordinateur."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import files\n",
                "uploaded = files.upload()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Étape 3 : Installer les librairies nécessaires"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install transformers datasets pandas accelerate"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Étape 4 : Lancer l'entraînement\n",
                "Le code ci-dessous est le même que sur votre ordinateur, mais il utilisera automatiquement le GPU de Colab pour tourner beaucoup plus vite."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "from datasets import Dataset\n",
                "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq\n",
                "\n",
                "print(\"Chargement du dataset...\")\n",
                "df = pd.read_csv(\"parallel_dataset.csv\")\n",
                "df = df.dropna()\n",
                "\n",
                "dataset = Dataset.from_pandas(df)\n",
                "dataset = dataset.train_test_split(test_size=0.1)\n",
                "\n",
                "model_checkpoint = \"Helsinki-NLP/opus-mt-en-fr\" # Modèle de base\n",
                "print(f\"Chargement du tokenizer et du modèle : {model_checkpoint}\")\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)\n",
                "model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)\n",
                "\n",
                "max_input_length = 128\n",
                "max_target_length = 128\n",
                "\n",
                "def preprocess_function(examples):\n",
                "    inputs = examples[\"merina\"]\n",
                "    targets = examples[\"betsileo\"]\n",
                "    model_inputs = tokenizer(\n",
                "        text=inputs, \n",
                "        text_target=targets, \n",
                "        max_length=max_input_length, \n",
                "        truncation=True\n",
                "    )\n",
                "    return model_inputs\n",
                "\n",
                "print(\"Tokenisation du dataset...\")\n",
                "tokenized_datasets = dataset.map(preprocess_function, batched=True)\n",
                "\n",
                "batch_size = 32 # On peut augmenter la taille du batch sur un GPU Colab\n",
                "args = Seq2SeqTrainingArguments(\n",
                "    \"merina-betsileo-model\",\n",
                "    eval_strategy=\"epoch\",\n",
                "    learning_rate=2e-5,\n",
                "    per_device_train_batch_size=batch_size,\n",
                "    per_device_eval_batch_size=batch_size,\n",
                "    weight_decay=0.01,\n",
                "    save_total_limit=3,\n",
                "    num_train_epochs=5, # Entraînement un peu plus long car c'est rapide sur GPU\n",
                "    predict_with_generate=True,\n",
                "    fp16=True, # Accélération de l'entraînement sur carte graphique\n",
                ")\n",
                "\n",
                "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)\n",
                "\n",
                "trainer = Seq2SeqTrainer(\n",
                "    model=model,\n",
                "    args=args,\n",
                "    train_dataset=tokenized_datasets[\"train\"],\n",
                "    eval_dataset=tokenized_datasets[\"test\"],\n",
                "    data_collator=data_collator,\n",
                "    processing_class=tokenizer,\n",
                ")\n",
                "\n",
                "print(\"Démarrage de l'entraînement...\")\n",
                "trainer.train()\n",
                "\n",
                "print(\"Sauvegarde du modèle final...\")\n",
                "trainer.save_model(\"merina-betsileo-model-final\")\n",
                "print(\"Entraînement terminé !\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Étape 5 : Télécharger le modèle final\n",
                "Exécutez cette cellule pour télécharger le modèle fini sur votre ordinateur sous forme d'archive zip."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "from google.colab import files\n",
                "\n",
                "shutil.make_archive(\"merina-betsileo-model-final\", 'zip', \"merina-betsileo-model-final\")\n",
                "files.download(\"merina-betsileo-model-final.zip\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "name": "Training_Merina_Betsileo.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("Training_Merina_Betsileo.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)
