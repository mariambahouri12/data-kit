# fix_metadata.py
import json
import os
from pathlib import Path

def fix_metadata():
    """Réparer ou supprimer les métadonnées corrompues"""
    metadata_dir = Path("uploads/metadata")
    
    if not metadata_dir.exists():
        print("📁 No metadata directory found")
        return
    
    corrupted_files = []
    
    for metadata_file in metadata_dir.glob("*.json"):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Corrupted: {metadata_file.name} - {e}")
            corrupted_files.append(metadata_file)
        except Exception as e:
            print(f"❌ Error reading {metadata_file.name}: {e}")
    
    if corrupted_files:
        print(f"\n🔧 Found {len(corrupted_files)} corrupted file(s)")
        response = input("Delete corrupted files? (y/n): ")
        
        if response.lower() == 'y':
            for file in corrupted_files:
                file.unlink()
                print(f"🗑️ Deleted: {file.name}")
            print("✅ Corrupted files deleted")
        else:
            print("ℹ️ Files kept")
    else:
        print("✅ All metadata files are valid")

if __name__ == "__main__":
    fix_metadata()