import os
import pandas as pd
from preprocessing import apply_preprocessing
from openai import OpenAI
from dotenv import load_dotenv

# Lade API-Key aus .env-Datei
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def load_prompt(path="prompts/default_prompt.txt") -> str:
    """Lädt die Prompt-Vorlage aus einer Textdatei."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_summary(text: str, prompt_template: str, model="gpt-4") -> str:
    """Erstellt eine Zusammenfassung mittels OpenAI GPT-Modell."""
    full_prompt = prompt_template + "\n" + text
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("API-Fehler:", e)
        return ""
