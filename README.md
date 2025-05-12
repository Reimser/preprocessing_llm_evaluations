# Preprocessing-LLM-Evaluation
## Projektbeschreibung
Dieses Projekt untersucht den Einfluss verschiedener Preprocessing-Formate auf die Qualität und Strukturvalidität von Textausgaben, die durch Large Language Models (LLMs) generiert werden.
Im Fokus steht die Entwicklung eines embedding-basierten Gütescores zur objektiven Bewertung der generierten Texte.

Das Projekt ist Vorarbeit für eine spätere End-to-End-Pipeline zur automatisierten Erstellung qualifizierter Arbeitszeugnisse.

## Zielsetzung
Vergleich von drei Eingabeformaten:

Freitext

Stichpunkte

JSON-strukturierte Eingaben

Generierung strukturierter Texte mit GPT-3.5 oder GPT-4 unter Einsatz von Instructor.

Entwicklung eines Embedding-basierten Similarity Scores auf Basis von Sentence-Transformers.

Empirische Bewertung der Preprocessing-Strategien anhand objektiver Metriken.

## Technologiestack
Python 3.11+

OpenAI API (openai)

Instructor (instructor)

Pydantic (pydantic)

Sentence-Transformers (sentence-transformers)

scikit-learn (sklearn)

Pandas (pandas)

Matplotlib oder Seaborn (optional für Visualisierung)

## Projektstruktur
```bash
Kopieren
Bearbeiten
preprocessing-llm-evaluation/
├── data/                 # Synthetische HR-Profile
├── preprocessing/        # Preprocessing-Funktionen (Textvarianten)
├── generator/            # GPT-Instructor-Calls
├── scoring/              # Entwicklung des Embedding-Scores
├── tests/                # Unit-Tests für Scoring und Preprocessing
├── README.md
├── requirements.txt
└── .gitignore
```