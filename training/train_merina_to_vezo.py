import os
import sys
import glob
import logging

# Ensure CUDA and Nvidia libraries are in LD_LIBRARY_PATH if running with virtual environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
venv_nvidia_lib = "/home/tovo/Bureau/scrap/venv/lib/python3.12/site-packages/nvidia"
if os.path.exists(venv_nvidia_lib):
    nvidia_libs = glob.glob(f"{venv_nvidia_lib}/*/lib")
    if nvidia_libs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(nvidia_libs) + (":" + current_ld if current_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

# Configuration
BASE_MODEL_DIR = "./merina_to_vezo"
OUTPUT_DIR = "./merina-vezo-checkpoints"
FINAL_MODEL_DIR = "./merina_to_vezo_final"
BIBLE_DATASET = "./dataset_vezo/parallel_text/parallel_bible.csv"
BOOKS_DATASET = "./dataset_vezo/parallel_text/parallel_books.csv"

def load_and_prepare_dataset():
    """Charge et nettoie les données parallèles Merina (malagasy_officiel) -> Vezo."""
    print("=" * 60)
    print("1. Chargement et préparation des données...")
    print("=" * 60)
    
    dfs = []
    if os.path.exists(BIBLE_DATASET):
        df_bible = pd.read_csv(BIBLE_DATASET)
        print(f"📖 Dataset Bible chargé : {len(df_bible)} lignes")
        dfs.append(df_bible[['malagasy_officiel', 'vezo']])
    else:
        print(f"⚠️ Fichier introuvable : {BIBLE_DATASET}")
        
    if os.path.exists(BOOKS_DATASET):
        df_books = pd.read_csv(BOOKS_DATASET)
        print(f"📚 Dataset Livres chargé : {len(df_books)} lignes")
        dfs.append(df_books[['malagasy_officiel', 'vezo']])
    else:
        print(f"⚠️ Fichier introuvable : {BOOKS_DATASET}")
        
    if not dfs:
        raise FileNotFoundError("Aucun jeu de données trouvé dans dataset_vezo/parallel_text/")
        
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Nettoyage
    df_combined = df_combined.dropna(subset=['malagasy_officiel', 'vezo'])
    df_combined['source'] = df_combined['malagasy_officiel'].astype(str).str.strip()
    df_combined['target'] = df_combined['vezo'].astype(str).str.strip()
    
    # Filtrer les textes vides ou trop courts
    df_combined = df_combined[(df_combined['source'].str.len() > 1) & (df_combined['target'].str.len() > 1)]
    
    # Suppression des doublons exacts
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['source', 'target'])
    print(f"📊 Total paires nettoyées : {len(df_combined)} (doublons supprimés : {before_dedup - len(df_combined)})")
    
    hf_dataset = Dataset.from_pandas(df_combined[['source', 'target']])
    dataset_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"🔹 Train set : {len(dataset_split['train'])} exemples")
    print(f"🔹 Eval/Test set : {len(dataset_split['test'])} exemples")
    
    return dataset_split

class ProgressLoggerCallback(TrainerCallback):
    """Callback pour suivre la perte d'entraînement et d'évaluation."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = round(state.epoch, 2) if state.epoch is not None else "?"
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")
            msg = f"[Epoch {epoch} | Step {state.global_step}]"
            if loss is not None:
                msg += f" Train Loss: {loss:.4f}"
            if eval_loss is not None:
                msg += f" Eval Loss: {eval_loss:.4f} ⭐"
            if lr is not None:
                msg += f" LR: {lr:.2e}"
            print(msg)

def main():
    # 1. Vérification GPU
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"🖥️ Périphérique d'entraînement : {device_name} (CUDA={has_cuda})")
    if has_cuda:
        print(f"⚡ Précision FP16 activée pour GPU NVIDIA")

    # 2. Chargement du Tokenizer et Modèle copié
    print("\n" + "=" * 60)
    print("2. Chargement du modèle source (Merina -> Betsileo2)...")
    print("=" * 60)
    print(f"📂 Répertoire de base : {BASE_MODEL_DIR}")
    
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
    
    # 3. Préparation des données
    raw_dataset = load_and_prepare_dataset()
    
    max_length = 128
    def preprocess_function(examples):
        inputs = [text for text in examples["source"]]
        targets = [text for text in examples["target"]]
        return tokenizer(text=inputs, text_target=targets, max_length=max_length, truncation=True)

    print("\n🔤 Tokenisation des données d'entraînement et d'évaluation...")
    tokenized_datasets = raw_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_dataset["train"].column_names
    )

    # 4. Configuration d'entraînement optimisée pour GPU 4GB VRAM (GTX 1650 Ti)
    print("\n" + "=" * 60)
    print("3. Configuration des hyperparamètres d'entraînement...")
    print("=" * 60)
    
    # Activer le gradient checkpointing sur le modèle pour réduire considérablement la mémoire GPU
    model.gradient_checkpointing_enable()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Équivaut à un batch size effectif de 16
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=8,
        predict_with_generate=False,    # Économise la mémoire GPU lors de l'évaluation
        fp16=has_cuda,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[ProgressLoggerCallback()],
    )

    print("\n🚀 Démarrage de l'entraînement / fine-tuning Merina -> Vezo...")
    trainer.train()

    print("\n" + "=" * 60)
    print("4. Sauvegarde du modèle final...")
    print("=" * 60)
    
    # Sauvegarde dans le dossier final et mise à jour du dossier merina_to_vezo
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    trainer.save_model(BASE_MODEL_DIR)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    
    print(f"✅ Modèle sauvegardé avec succès dans :")
    print(f"   - {FINAL_MODEL_DIR}")
    print(f"   - {BASE_MODEL_DIR}")
    print("🎉 Entraînement terminé !")

if __name__ == "__main__":
    main()
