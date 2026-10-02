import os
import json
import shutil
import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback

def prepare_tokenizer():
    """Prépare les fichiers pour que MarianTokenizer puisse les lire."""
    print("1. Préparation des fichiers du Tokenizer pour MarianMT...")
    os.makedirs("custom_marian_tokenizer", exist_ok=True)
    
    # MarianMT s'attend à deux fichiers SentencePiece (source et target)
    shutil.copy("marian_malagasy.model", "custom_marian_tokenizer/source.spm")
    shutil.copy("marian_malagasy.model", "custom_marian_tokenizer/target.spm")

    # Il a aussi besoin d'un vocab.json qui mappe les tokens à leurs IDs
    vocab = {}
    with open("marian_malagasy.vocab", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            token = line.strip().split('\t')[0]
            vocab[token] = i

    with open("custom_marian_tokenizer/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("✅ Fichiers tokenizer prêts.")

def main():
    prepare_tokenizer()

    print("\n2. Chargement du nouveau Tokenizer et du Modèle de base...")
    # On charge notre nouveau tokenizer
    tokenizer = MarianTokenizer.from_pretrained("custom_marian_tokenizer")
    
    # On charge le modèle existant (ou un modèle pré-entraîné basique)
    model_checkpoint = "./merina_to_betsileo" if os.path.exists("./merina_to_betsileo") else "Helsinki-NLP/opus-mt-en-fr"
    print(f"Modèle de base : {model_checkpoint}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # ÉTAPE CRUCIALE : On doit redimensionner la couche d'embeddings du modèle 
    # car la taille de notre nouveau vocabulaire (16 000) est différente de l'ancien.
    print(f"Redimensionnement des embeddings du modèle à {len(tokenizer)} tokens...")
    model.resize_token_embeddings(len(tokenizer))

    # Mise à jour des identifiants spéciaux du modèle pour correspondre au nouveau tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.pad_token_id # MarianMT utilise souvent le pad comme début de phrase
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)

    # Mise à jour également de la configuration de génération
    if model.generation_config:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
        model.generation_config.bad_words_ids = [[tokenizer.pad_token_id]]
        model.generation_config.forced_eos_token_id = tokenizer.eos_token_id

    print("\n3. Préparation du Dataset...")
    df = pd.read_csv("./dataset/parallel_dataset.csv")
    df = df.dropna()
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)
    
    max_length = 128
    def preprocess_function(examples):
        # On met tout en minuscules pour l'entraînement du modèle aussi !
        inputs = [str(text).lower() for text in examples["merina"]]
        targets = [str(text).lower() for text in examples["betsileo"]]
        return tokenizer(text=inputs, text_target=targets, max_length=max_length, truncation=True)
    
    print("Tokenisation des phrases d'entraînement...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    print("\n4. Configuration de l'entraînement...")
    # Ajout d'un callback pour afficher les métriques après chaque epoch
    class SimpleLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # logs contient loss, eval_loss, etc.
            if logs is not None:
                epoch = round(state.epoch, 2) if hasattr(state, "epoch") else "?"
                loss = logs.get("loss")
                eval_loss = logs.get("eval_loss")
                lr = logs.get("learning_rate")
                msg = f"[Epoch {epoch}]"
                if loss is not None:
                    msg += f" loss={loss:.4f}"
                if eval_loss is not None:
                    msg += f" eval_loss={eval_loss:.4f}"
                if lr is not None:
                    msg += f" lr={lr:.2e}"
                print(msg)
    
    # Configuration des arguments d'entraînement avec sauvegarde du meilleur modèle
    args = Seq2SeqTrainingArguments(
        output_dir="merina-betsileo-new-tokenizer",
        eval_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="logs",
        learning_rate=1e-4, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=9, # 10 époques suffisent largement
        predict_with_generate=True,
        fp16=True, # Accélération GPU

        # --- PARAMÈTRES ANTI-OVERFITTING ---
        save_strategy="epoch",             # Sauvegarder à chaque époque
        load_best_model_at_end=True,       # Charger le meilleur modèle à la fin
        metric_for_best_model="eval_loss", # Se baser sur la perte de validation
        greater_is_better=False,           # Plus la loss est basse, mieux c'est
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[SimpleLogger()],
    )
    
    print("🚀 Démarrage de l'entraînement (cela peut prendre du temps)...")
    trainer.train()
    
    print("💾 Sauvegarde du modèle final...")
    trainer.save_model("merina-betsileo-new-tokenizer-final")
    tokenizer.save_pretrained("merina-betsileo-new-tokenizer-final")
    print("✅ Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()
