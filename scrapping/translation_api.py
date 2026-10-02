from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Merina to Betsileo Translator API")

# Allow Next.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str

print("Loading model...")
# Path to the fine-tuned model saved in the current directory
model_path = "./merina-betsileo-model-final"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model from {model_path}. Please make sure you have finished training the model.")
    print("Falling back to a placeholder model for demonstration purposes...")
    # Fallback to an empty model/dummy translation if the model is not trained yet.
    tokenizer = None
    model = None

@app.get("/")
async def root():
    return {"status": "online", "model": "merina-betsileo"}

@app.post("/translate")
async def translate(req: TranslationRequest):
    if tokenizer is None or model is None:
        # Dummy translation
        return {"translation": f"[Betsileo dialect translation for: {req.text}]"}

    inputs = tokenizer(req.text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated_text}

if __name__ == "__main__":
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
