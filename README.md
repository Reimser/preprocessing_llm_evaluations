# Projekt: Analyse des Einflusses von Preprocessing auf die semantische Qualität von GPT-4-Zusammenfassungen
## Projektbeschreibung
Dieses Projekt untersucht den Einfluss verschiedener Preprocessing-Strategien auf die semantische Qualität von durch GPT-4 generierten Zusammenfassungen. Bewertet wird die Qualität anhand der Kosinus-Ähnlichkeit von Sentence-BERT-Embeddings im Vergleich zu Referenzzusammenfassungen aus dem CNN/DailyMail-Datensatz.

Die Arbeit dient als empirische Analyse und stellt alle Ergebnisse reproduzierbar zur Verfügung.
Die Dozentin kann den gesamten Workflow mit einem Beispiel durchlaufen lassen, da API-Aufrufe kostenpflichtig sind. Die vollständigen Ergebnisse mit 100 Beispielen sind bereits vollständig vorhanden und dokumentiert.

## Projektstruktur

```
C:.
│   .env                              # Umgebungsvariablen, z. B. API-Key
│   .gitignore                        # Git-Konfigurationsdatei (nicht versionierte Dateien)
│   evaluate_embeddings.py            # Script zur Berechnung der Similarity Scores
│   main.py                           # Hauptskript zum Generieren der GPT-4 Zusammenfassungen
│   README.md                         # Projektbeschreibung und Anleitung
│   requirements.txt                  # Alle Python-Abhängigkeiten
│
├───data                              # Alle Datensätze
│   ├───eval
│   │       eval_subset1.csv           # Test-Subset mit 100 Artikeln + Referenzzusammenfassungen
│   │       eval_subset.csv            # Test set mit 3 Beispielen fuer den Testdurchlauf!!!!
│   ├───processed
│   │       eval_results_with_all_strategies1.csv  # Ergebnisse aller Strategien auf 100 Artikeln
│   │
│   └───raw
│           cnn.csv                   # Originalartikel (optional / Raw-Daten)
│
├───generate
│       generate_summaries.py         # GPT-4 API Anbindung und Prompt-basierte Generierung
│
├───models
│       embedder.py                   # Wrapper für Sentence-BERT zum Berechnen von Embeddings
│
├───notebooks
│       results_visu.ipynb            # Notebook zur Visualisierung der Ergebnisse
│
├───preprocessing                     # Module für Textvorverarbeitung
│       clean.py                      # Entfernen von Sonderzeichen, Normalisierung
│       lemmatize.py                  # Lemmatisierung mit spaCy
│       sentence_filter.py            # Filterung von Hauptsätzen
│       stopwords.py                  # Entfernen von Stoppwörtern
│       tfidf.py                      # TF-IDF Analysemodul (optional, nicht zentral)
│       truncate.py                   # Token-basierte Kürzung auf 512 Tokens
│       __init__.py                   # Init-Datei für Python-Package-Struktur
│
├───prompts
│       default_prompt.txt            # Standard-Prompt-Template für GPT-4 API
│
├───results
│      
│       evaluation_results1.csv        # Endgültige Ähnlichkeits-Ergebnisse aller Strategien
│
└───utils
        create_eval_subset.py         # Script zur Erstellung des Subsets mit 100 Artikeln
```

## Abhängigkeiten
Alle benötigten Python-Pakete sind in requirements.txt aufgeführt. Installation mit:

pip install -r requirements.txt

Zusätzlich muss das spaCy-Sprachmodell en_core_web_sm installiert werden:

python -m spacy download en_core_web_sm

## Ausführungsreihenfolge
### Vorbereitung

Eine .env-Datei mit gültigem OPENAI_API_KEY muss vorhanden sein.

Hinweis: Falls die bereitgestellten API-Credentials abgelaufen sind oder nicht mehr funktionieren, bitte direkt bei mir melden – ich stelle neue gültige Credentials bereit.

### Generierung der Zusammenfassungen

Das Projekt ist so vorbereitet, dass main.py standardmäßig nur mit einem einzelnen Beispiel ausgeführt wird.
Dies dient dem Nachweis der Funktionsweise im Testlauf und vermeidet unnötige Kosten durch GPT-4 API-Anfragen.

### Start des Testlaufs:

python main.py

Die vollständigen Ergebnisse für alle 100 Artikel sind bereits im Projekt enthalten (eval_results_with_all_strategies.csv) und dokumentieren den kompletten Vergleich aller Preprocessing-Strategien.

### Evaluation der semantischen Qualität

python evaluate_embeddings.py

Die Ergebnisse werden automatisch in results/evaluation_results.csv gespeichert. Diese Datei enthält alle Bewertungen für alle Preprocessing-Strategien und die 100 Beispiele.

### Visualisierung der Ergebnisse
Optional kann das Notebook notebooks/results_visu.ipynb geöffnet werden, um die Ergebnisse grafisch zu analysieren und darzustellen.

## Hinweise für die Prüferin
Das Projekt kann vollständig nachvollzogen werden, indem main.py und evaluate_embeddings.py mit einem einzelnen Beispiel ausgeführt werden.

Die vollständigen Ergebnisse (mit 100 Beispielen) liegen bereits in den bereitgestellten Dateien vor (eval_results_with_all_strategies.csv und evaluation_results.csv).

Bitte beachten: API-Zugriffe sind kostenpflichtig. Daher wurde die Pipeline so vorbereitet, dass sie im Testlauf nur mit einem Beispiel arbeitet.

Falls die API-Credentials nicht mehr gültig sind: Bitte bei mir melden. Ich stelle kurzfristig neue Zugangsdaten zur Verfügung.