import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv("parallel_dataset.csv")
    
    # Drop any nulls just in case
    df = df.dropna()
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Using a multilingual model as base, for example facebook/nllb-200-distilled-600M or an opus-mt model.
    # Since it's Malagasy (Merina) to Betsileo, let's use MarianMT or mBART. Here we use an opus-mt architecture.
    # If no specific Malagasy model is available locally, we can initialize a small model or fine-tune an existing one.
    # For demonstration, we'll use a fast pre-trained model to fine-tune.
    model_checkpoint = "Helsinki-NLP/opus-mt-en-fr" # Placeholder, ideally we use a proper base model
    print(f"Loading tokenizer and model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    max_input_length = 128
    max_target_length = 128
    
    def preprocess_function(examples):
        inputs = examples["merina"]
        targets = examples["betsileo"]
        model_inputs = tokenizer(
            text=inputs, 
            text_target=targets, 
            max_length=max_input_length, 
            truncation=True
        )
        return model_inputs
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    batch_size = 16
    args = Seq2SeqTrainingArguments(
        "merina-betsileo-model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("merina-betsileo-model-final")
    print("Training complete!")

if __name__ == "__main__":
    main()
