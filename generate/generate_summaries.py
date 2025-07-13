import os
import pandas as pd
from preprocessing import apply_preprocessing
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # .env-Datei laden
api_key = os.getenv("OPENAI_API_KEY")  # API-Key lesen
client = OpenAI(api_key=api_key)  # OpenAI-Client initialisieren

def load_prompt(path="prompts/default_prompt.txt") -> str:  # Prompt-Vorlage laden
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_summary(text: str, prompt_template: str, model="gpt-4") -> str:  # API-Request für Zusammenfassung
    full_prompt = prompt_template + "\n" + text  # Prompt zusammenbauen
    try:
        response = client.chat.completions.create(  # API-Aufruf an GPT-4
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()  # Antwort zurückgeben
    except Exception as e:
        print("API-Fehler:", e)  # Fehlerbehandlung bei API-Problemen
        return ""  # Leerer String als Fallback
