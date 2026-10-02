import os
import sentencepiece as spm

# Configuration
input_file = "./data/all_dataFinal.txt"
model_prefix = "boloky_tokenizer"
vocab_size = 32000

print("=== ENTRAÎNEMENT D'UN NOUVEAU TOKENIZER ===")

if not os.path.exists(input_file):
    print(f"❌ Erreur: Le fichier d'entrée '{input_file}' est introuvable. Assurez-vous que le dossier './data' contient bien le fichier.")
    exit(1)

# Entraînement de SentencePiece
print(f"Entraînement du modèle SentencePiece sur {input_file} (vocab_size={vocab_size})...")
print("Cela peut prendre quelques minutes selon la taille des données.")

try:
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1, 
        model_type="unigram",   
        pad_id=0,
        unk_id=1,
        bos_id=-1,              
        eos_id=2,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>"
    )

    print(f"\n✅ Terminé ! Le modèle a été sauvegardé sous les noms :")
    print(f"  - {model_prefix}.model (le modèle binaire)")
    print(f"  - {model_prefix}.vocab (le vocabulaire lisible)")

except Exception as e:
    print(f"\n❌ Une erreur est survenue lors de l'entraînement : {e}")
