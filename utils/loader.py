"""
Funktionen zum Laden und Vorbereiten des CNN/DailyMail-Datensatzes.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import json
import requests
import gzip
import shutil
from tqdm import tqdm

def download_file(url: str, target_path: Path) -> bool:
    """Lädt eine Datei herunter und zeigt einen Fortschrittsbalken."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(target_path, 'wb') as f, tqdm(
            desc=target_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Fehler beim Herunterladen von {url}: {e}")
        return False

def download_cnn_dailymail(data_dir: str) -> bool:
    """Lädt den CNN/DailyMail-Datensatz herunter."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs für die verschiedenen Splits
    urls = {
        "train": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/data/train.csv.gz",
        "validation": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/data/validation.csv.gz",
        "test": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/data/test.csv.gz"
    }
    
    success = True
    for split, url in urls.items():
        gz_path = raw_dir / f"{split}.csv.gz"
        csv_path = raw_dir / f"{split}.csv"
        
        # Überspringe, wenn die Datei bereits existiert
        if csv_path.exists():
            print(f"{split}.csv existiert bereits, überspringe Download.")
            continue
        
        # Lade herunter
        print(f"Lade {split} Split herunter...")
        if download_file(url, gz_path):
            # Entpacke
            print(f"Entpacke {split}.csv.gz...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Lösche .gz Datei
            gz_path.unlink()
        else:
            success = False
    
    return success

def load_cnn_dailymail(
    data_dir: str,
    split: str = "train"
) -> List[Dict[str, str]]:
    """Lädt den CNN/DailyMail-Datensatz."""
    data_path = Path(data_dir) / "raw" / "cnn.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Datensatz nicht gefunden unter {data_path}")
    
    print(f"Lade Datensatz aus {data_path}...")
    df = pd.read_csv(data_path)
    
    # Konvertiere in das erwartete Format
    articles = []
    for _, row in df.iterrows():
        article = {
            "text": row["article"],  # Originaltext
            "summary": row["highlights"]  # Goldstandard-Zusammenfassung
        }
        articles.append(article)
    
    print(f"Geladen: {len(articles)} Artikel")
    return articles

def load_processed_variants(
    data_dir: str,
    variant_names: List[str]
) -> Dict[str, List[str]]:
    """Lädt die verarbeiteten Textvarianten."""
    variants = {}
    for name in variant_names:
        file_path = Path(data_dir) / "processed" / f"{name}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                variants[name] = json.load(f)
    return variants

def load_goldstandard_summaries(
    data_dir: str
) -> List[str]:
    """Lädt die Goldstandard-Zusammenfassungen."""
    data_path = Path(data_dir) / "raw" / "cnn.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df["highlights"].tolist()
    return []

def prepare_dataset(data_dir: str) -> bool:
    """Bereitet den Datensatz vor (Download und Struktur)."""
    try:
        # Erstelle Verzeichnisstruktur
        data_dir = Path(data_dir)
        for subdir in ["raw", "processed"]:
            (data_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Lade Datensatz
        return download_cnn_dailymail(data_dir)
    except Exception as e:
        print(f"Fehler bei der Datensatzvorbereitung: {e}")
        return False 