import gradio as gr
import sentencepiece as spm
import os

# Chemin vers le modèle (en remontant d'un dossier car on est dans ./ui)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "boloky_tokenizer.model")

# Initialisation du tokenizer
sp = spm.SentencePieceProcessor()

def load_model():
    if os.path.exists(MODEL_PATH):
        sp.load(MODEL_PATH)
        return True
    return False

model_loaded = load_model()

def tokenize(text):
    if not model_loaded:
        return [("Erreur", "Modèle introuvable")], "0", "Veuillez vérifier que le modèle est bien dans le dossier /tokenizer/"
    
    if not text.strip():
        return [], "0", ""

    # Encodage en pièces (texte) et en IDs (nombres)
    pieces = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    
    # Formatage pour l'affichage coloré
    highlighted_output = []
    details = []
    
    for i, (piece, token_id) in enumerate(zip(pieces, ids)):
        # On alterne les labels A et B juste pour avoir des couleurs différentes visuellement
        color_label = "A" if i % 2 == 0 else "B"
        highlighted_output.append((piece, color_label))
        
        # Détails en texte brut
        details.append(f"Token {i+1}: '{piece}' (ID: {token_id})")

    return highlighted_output, str(len(pieces)), "\n".join(details)

# Interface Graphique avec Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("""
    # 🦜 Boloky Tokenizer Tester
    Cette interface vous permet de visualiser instantanément comment votre modèle SentencePiece découpe le texte malgache.
    """)
    
    if not model_loaded:
        gr.Warning(f"⚠️ Modèle introuvable au chemin : {MODEL_PATH}")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texte d'entrée", 
                placeholder="Tapez une phrase en malgache ici...", 
                lines=5
            )
            
            with gr.Row():
                clear_btn = gr.Button("Effacer")
                submit_btn = gr.Button("Tokeniser", variant="primary")
                
            token_count = gr.Textbox(label="Nombre total de tokens")

        with gr.Column(scale=1):
            gr.Markdown("### Visualisation des Tokens")
            output_visual = gr.HighlightedText(
                label="Découpage",
                combine_adjacent=False,
                show_legend=False,
                color_map={"A": "#a7f3d0", "B": "#bae6fd"} # Couleurs douces alternées
            )
            
            with gr.Accordion("Voir les détails (IDs)", open=False):
                output_details = gr.Textbox(label="Liste des IDs", lines=10)

    # Actions
    input_text.change(fn=tokenize, inputs=input_text, outputs=[output_visual, token_count, output_details])
    submit_btn.click(fn=tokenize, inputs=input_text, outputs=[output_visual, token_count, output_details])
    clear_btn.click(lambda: ("", [], "0", ""), inputs=[], outputs=[input_text, output_visual, token_count, output_details])

if __name__ == "__main__":
    print(f"Lancement de l'interface graphique...")
    print(f"Modèle chargé depuis : {MODEL_PATH}")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
