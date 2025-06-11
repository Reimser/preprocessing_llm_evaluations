"""
Generiert Zusammenfassungen mit GPT für verschiedene Preprocessing-Varianten.
"""

import os
from pathlib import Path
from typing import Dict, List
import openai
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_prompt_template() -> str:
    """Lädt die Prompt-Vorlage."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "default_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def generate_summary(text: str, model: str = "gpt-3.5-turbo") -> str:
    """Generiert eine Zusammenfassung mit GPT."""
    prompt_template = load_prompt_template()
    prompt = prompt_template.format(text=text)
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent, der Texte zusammenfasst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fehler bei der Zusammenfassungsgenerierung: {e}")
        return ""

def process_text_variants(text_variants: Dict[str, str]) -> Dict[str, str]:
    """Verarbeitet verschiedene Textvarianten und generiert Zusammenfassungen."""
    summaries = {}
    for variant_name, text in text_variants.items():
        summary = generate_summary(text)
        summaries[variant_name] = summary
    return summaries 