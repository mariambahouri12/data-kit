# clean_all.py
import os
import json
from pathlib import Path

def clean_all():
    """Nettoyer tous les fichiers corrompus"""
    
    # Supprimer les métadonnées corrompues
    metadata_dir = Path("uploads/metadata")
    if metadata_dir.exists():
        for file in metadata_dir.glob("*.json"):
            if file.name == "stats.json":
                continue
            try:
                with open(file, 'r') as f:
                    json.load(f)
            except:
                print(f"🗑️ Deleting: {file.name}")
                file.unlink()
    
    # Supprimer le cache Streamlit
    cache_dir = Path(".streamlit/cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("🗑️ Deleted Streamlit cache")
    
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    clean_all()