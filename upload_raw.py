# upload_raw.py
"""
Script pour uploader des fichiers CSV rapidement.

Usage:
    python upload_raw.py --file path/to/file.csv
    python upload_raw.py --dir path/to/directory
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uploads import UploadManager


def main():
    parser = argparse.ArgumentParser(description='Upload CSV files')
    parser.add_argument('--file', type=str, help='Path to CSV file')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--user', type=str, default='default', help='User ID')
    parser.add_argument('--tags', type=str, help='Tags (comma separated)')
    
    args = parser.parse_args()
    
    # Initialiser le gestionnaire
    manager = UploadManager()
    
    # Traiter les tags
    tags = {}
    if args.tags:
        for tag in args.tags.split(','):
            if ':' in tag:
                key, value = tag.split(':', 1)
                tags[key] = value
            else:
                tags[tag] = True
    
    # Upload d'un fichier
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        metadata = manager.upload(content, file_path.name, user_id=args.user, tags=tags)
        print(f"✅ Uploaded: {metadata['filename']}")
        print(f"   ID: {metadata['file_id']}")
        print(f"   Size: {metadata['file_size_mb']:.2f} MB")
        print(f"   Rows: {metadata['analysis']['shape']['rows']}")
        print(f"   Columns: {metadata['analysis']['shape']['columns']}")
    
    # Upload d'un dossier
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"❌ Directory not found: {dir_path}")
            return
        
        csv_files = list(dir_path.glob('*.csv')) + list(dir_path.glob('*.xlsx'))
        print(f"📁 Found {len(csv_files)} files in {dir_path}")
        
        for file_path in csv_files:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                metadata = manager.upload(content, file_path.name, user_id=args.user, tags=tags)
                print(f"✅ Uploaded: {metadata['filename']}")
                print(f"   ID: {metadata['file_id']}")
                
            except Exception as e:
                print(f"❌ Error uploading {file_path.name}: {e}")
    
    else:
        print("❌ Please specify --file or --dir")
        parser.print_help()


if __name__ == '__main__':
    main()