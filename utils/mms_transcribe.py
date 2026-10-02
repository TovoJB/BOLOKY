import torch
import librosa
import sys
import os
from transformers import Wav2Vec2ForCTC, AutoProcessor


def transcribe(audio_path, model_id="facebook/mms-1b-all", lang="mlg"):
    """
    Transcripteur audio utilisant le modèle Meta MMS.
    Cherche d'abord le modèle localement dans './mms_model'.
    Découpe l'audio en chunks de 30s pour éviter les erreurs OOM.
    """
    local_model_path = "./mms_model"

    if os.path.exists(local_model_path):
        model_to_load = local_model_path
        print(f"Chargement du modèle depuis le dossier local : {local_model_path}")
    else:
        model_to_load = model_id
        print(f"Chargement du modèle depuis Hugging Face : {model_id}")

    if not os.path.exists(audio_path):
        print(f"Erreur : Le fichier {audio_path} n'existe pas.")
        return None

    print(f"Chargement du modèle {model_id} pour la langue '{lang}'...")

    try:
        # Le modèle MMS 1B nécessite ~4GB en float32.
        # Le GTX 1650 Ti (4GB) n'a pas assez de VRAM libre (OS + affichage prennent ~200MB).
        # On force le CPU qui fonctionne parfaitement (confirmé par les tests).
        device = "cpu"
        torch_dtype = torch.float32
        print(f"Inférence sur : {device} (float32)")

        # Charger le processeur (tokenizer + feature extractor)
        processor = AutoProcessor.from_pretrained(model_to_load, target_lang=lang)

        model = Wav2Vec2ForCTC.from_pretrained(
            model_to_load,
            target_lang=lang,
            ignore_mismatched_sizes=True,
            torch_dtype=torch_dtype
        )

        model.to(device)
        model.eval()
        print(f"Modèle chargé sur : {device} (Dtype: {torch_dtype})")

        print(f"Traitement de l'audio : {audio_path}")

        # Charger et rééchantillonner à 16kHz
        audio, _ = librosa.load(audio_path, sr=16000)

        duration = len(audio) / 16000
        print(f"Chargement de l'audio ({duration:.1f} secondes)...")
        print(f"Amplitude max : {max(abs(audio)):.4f}")

        # Découpage manuel en chunks de 30s
        chunk_duration_s = 30
        chunk_size = chunk_duration_s * 16000
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

        print(f"Transcription par {len(chunks)} morceau(x) de {chunk_duration_s}s...")

        all_transcriptions = []
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] ({len(chunk)/16000:.1f}s)...", end=" ", flush=True)

            inputs = processor(chunk, sampling_rate=16_000, return_tensors="pt")

            # Caster input_values en float16 si GPU
            input_values = inputs.input_values.to(device=device, dtype=torch_dtype)

            # attention_mask reste en entier (pas de cast float)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device)

            with torch.no_grad():
                if attention_mask is not None:
                    logits = model(input_values=input_values, attention_mask=attention_mask).logits
                else:
                    logits = model(input_values=input_values).logits

            ids = torch.argmax(logits, dim=-1)[0]
            chunk_text = processor.decode(ids)
            all_transcriptions.append(chunk_text)

            preview = chunk_text[:60] + "..." if len(chunk_text) > 60 else chunk_text
            print(f"OK -> '{preview}'")

        transcription = " ".join(all_transcriptions).strip()
        print(f"\nTranscription terminée ! ({len(transcription)} caractères)")

        return transcription

    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python mms_transcribe.py <chemin_vers_audio>")
        print("Exemple: python mms_transcribe.py dataBrute/BibleAudioData/NT/01-Matio/ch_01.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    result = transcribe(audio_file)

    if result is not None:
        if len(result.strip()) == 0:
            print("\nAttention : La transcription est vide.")
        else:
            print("\n" + "="*40)
            print("TRANSCRIPTION :")
            print("="*40)
            print(result)
            print("="*40)

        # Enregistrer dans le dossier 'output'
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = os.path.basename(audio_file)
        output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".txt")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"\nTranscription enregistrée dans : {output_file}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement : {e}")
    else:
        print("\nLa transcription a échoué.")
