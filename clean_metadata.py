# clean_metadata.py
import os
from pathlib import Path
import json

def clean_corrupted_metadata():
    """Supprimer les fichiers de métadonnées corrompus"""
    metadata_dir = Path("uploads/metadata")
    
    if not metadata_dir.exists():
        print("📁 No metadata directory")
        return
    
    deleted = 0
    for file in metadata_dir.glob("*.json"):
        if file.name == "stats.json":
            continue
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                json.load(f)
        except:
            print(f"🗑️ Deleting corrupted: {file.name}")
            file.unlink()
            deleted += 1
    
    print(f"✅ Deleted {deleted} corrupted file(s)")

if __name__ == "__main__":
    clean_corrupted_metadata()