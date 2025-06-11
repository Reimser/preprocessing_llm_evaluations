# LLM Preprocessing Vergleichspipeline

Dieses Projekt implementiert eine Pipeline zur Untersuchung des Einflusses verschiedener Preprocessing-Strategien auf die Qualität von LLM-generierten Zusammenfassungen. Die Evaluation erfolgt durch Vergleich mit Goldstandard-Zusammenfassungen (CNN/DailyMail) mittels Embeddings.

## Projektstruktur

```
llm_preprocessing_project/
│
├── data/                       # Download & Speicherung des CNN/DailyMail-Datensatzes
│   └── raw/                    # Unveränderte Originaldaten
│   └── processed/              # Preprocessed-Varianten
│
├── preprocessing/             # Alle Preprocessing-Methoden als Module
│   ├── clean.py                # z. B. Sonderzeichen entfernen, normalize text
│   ├── stopwords.py            # Entfernt Stoppwörter
│   ├── lemmatize.py            # Lemmatization mit spaCy
│   ├── truncate.py             # Kürzt Text auf N Tokens
│   └── sentence_filter.py      # Keyword-, NER-, Hauptsatzfilter
│
├── prompts/                   # GPT-Prompt-Vorlagen für Generierung
│   └── default_prompt.txt
│
├── generate/                  # GPT-Zusammenfassungen erzeugen
│   └── generate_summaries.py
│
├── evaluate/                  # Embedding-Vergleich & Metriken
│   └── compare_embeddings.py
│
├── models/                    # SentenceTransformer & GPT-Konfigurationen
│   └── embedder.py
│
├── utils/                     # Hilfsfunktionen
│   └── loader.py              # Datensatz laden
│   └── writer.py              # Ergebnisse speichern
│
├── results/                   # Ergebnisse + Visualisierungen
│   └── similarities.csv
│
├── notebooks/                 # Explorative Analysen
│   └── overview.ipynb
│
├── main.py                    # Steuerung der Pipeline
├── requirements.txt
└── README.md
```

## Installation

1. Klone das Repository:
```bash
git clone [repository-url]
cd llm_preprocessing_project
```

2. Erstelle eine virtuelle Umgebung und aktiviere sie:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installiere die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

4. Lade die spaCy-Modelle:
```bash
python -m spacy download en_core_web_sm
```

5. Erstelle eine `.env` Datei im Hauptverzeichnis mit deinem OpenAI API-Key:
```
OPENAI_API_KEY=dein-api-key
```

## Verwendung

1. Lade den CNN/DailyMail-Datensatz in das `data/raw` Verzeichnis.

2. Führe die Pipeline aus:
```bash
python main.py --data_dir data --output_dir results
```

Die Pipeline wird:
- Die Texte mit verschiedenen Preprocessing-Strategien verarbeiten
- GPT-Zusammenfassungen für jede Variante generieren
- Die Zusammenfassungen mit dem Goldstandard vergleichen
- Die Ergebnisse im `results` Verzeichnis speichern

## Preprocessing-Strategien

- **Clean**: Entfernt Sonderzeichen und normalisiert den Text
- **Stopwords**: Entfernt Stoppwörter
- **Lemmatize**: Führt Lemmatisierung mit spaCy durch
- **Truncate**: Kürzt den Text auf eine maximale Token-Anzahl
- **Sentence Filter**: Filtert Sätze basierend auf Keywords, NER oder syntaktischer Komplexität

## Evaluation

Die Evaluation erfolgt durch:
1. Vergleich der Embeddings zwischen verschiedenen Varianten
2. Vergleich mit Goldstandard-Zusammenfassungen
3. Berechnung von Ähnlichkeitsmetriken
4. Visualisierung der Ergebnisse

## Lizenz

[Lizenzinformationen]