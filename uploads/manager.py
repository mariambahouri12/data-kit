# uploads/manager.py
import os
import json
import hashlib
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import uuid
import warnings


class UploadManager:
    """
    Gestionnaire central des uploads.
    Gère le stockage, les métadonnées et la gestion des fichiers.
    """
    
    def __init__(self, 
                 base_dir: str = "uploads",
                 max_file_size: int = 100 * 1024 * 1024,  # 100 MB
                 allowed_extensions: List[str] = None):
        """
        Args:
            base_dir: Dossier racine des uploads
            max_file_size: Taille maximale du fichier (octets)
            allowed_extensions: Extensions autorisées
        """
        self.base_dir = Path(base_dir)
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or ['.csv', '.xlsx', '.xls', '.parquet']
        
        # Créer les sous-dossiers
        self._create_directories()
        
        # Initialiser les statistiques
        self.stats = {
            'total_uploads': 0,
            'total_size': 0,
            'by_type': {}
        }
        self._load_stats()
    
    def _create_directories(self):
        """Créer la structure de dossiers"""
        directories = [
            self.base_dir,
            self.base_dir / 'raw',
            self.base_dir / 'processed',
            self.base_dir / 'metadata'
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Créer .gitkeep si le dossier est vide
            gitkeep = dir_path / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
    
    def _load_stats(self):
        """Charger les statistiques"""
        stats_file = self.base_dir / 'stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
    
    def _save_stats(self):
        """Sauvegarder les statistiques"""
        stats_file = self.base_dir / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def upload(self, 
               file_content: bytes, 
               filename: str,
               user_id: Optional[str] = None,
               tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Uploader un fichier.
        
        Args:
            file_content: Contenu du fichier
            filename: Nom du fichier
            user_id: ID de l'utilisateur (optionnel)
            tags: Tags supplémentaires
        
        Returns:
            Métadonnées du fichier uploadé
        """
        # Valider le fichier
        self._validate_file(file_content, filename)
        
        # Générer les identifiants
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        
        # Créer le nom de fichier sécurisé
        safe_filename = f"{timestamp}_{base_name}_{file_id}{extension}"
        file_path = self.base_dir / 'raw' / safe_filename
        
        # Sauvegarder le fichier
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Charger et analyser le fichier
        file_info = self._analyze_file(file_path, extension)
        
        # Générer les métadonnées
        metadata = {
            'file_id': file_id,
            'filename': filename,
            'safe_filename': safe_filename,
            'file_path': str(file_path),
            'file_size': len(file_content),
            'file_size_mb': len(file_content) / (1024 * 1024),
            'extension': extension,
            'uploaded_at': datetime.now().isoformat(),
            'user_id': user_id,
            'tags': tags or {},
            'hash': hashlib.md5(file_content).hexdigest(),
            'status': 'uploaded',
            'analysis': file_info
        }
        
        # Sauvegarder les métadonnées
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Mettre à jour les statistiques
        self.stats['total_uploads'] += 1
        self.stats['total_size'] += len(file_content)
        ext = extension.lower()
        self.stats['by_type'][ext] = self.stats['by_type'].get(ext, 0) + 1
        self._save_stats()
        
        return metadata
    
    def _validate_file(self, file_content: bytes, filename: str):
        """Valider le fichier"""
        # Vérifier la taille
        if len(file_content) > self.max_file_size:
            raise ValueError(
                f"File too large: {len(file_content)} > {self.max_file_size} bytes. "
                f"Max size: {self.max_file_size / (1024*1024):.1f} MB"
            )
        
        # Vérifier l'extension
        extension = Path(filename).suffix.lower()
        if extension not in self.allowed_extensions:
            raise ValueError(
                f"Extension '{extension}' not allowed. "
                f"Allowed: {self.allowed_extensions}"
            )
        
        # Vérifier le contenu (pour CSV)
        if extension == '.csv':
            try:
                # Lire les premières lignes pour valider
                import io
                pd.read_csv(io.BytesIO(file_content), nrows=5)
            except Exception as e:
                raise ValueError(f"Invalid CSV file: {str(e)}")
    
    def _analyze_file(self, file_path: Path, extension: str) -> Dict[str, Any]:
        """Analyser le fichier"""
        try:
            if extension == '.csv':
                df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                return {'error': 'Unsupported file type'}
            
            # Analyse de base
            analysis = {
                'shape': {
                    'rows': len(df),
                    'columns': len(df.columns)
                },
                'columns': {
                    'names': df.columns.tolist(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'n_unique': df.nunique().to_dict()
                },
                'missing': {
                    'total': df.isnull().sum().sum(),
                    'percentage': (df.isnull().sum().sum() / df.size) * 100,
                    'by_column': df.isnull().sum().to_dict()
                },
                'memory_usage': {
                    'total': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                    'by_column': (df.memory_usage(deep=True) / (1024 * 1024)).to_dict()
                },
                'dtypes_summary': df.dtypes.value_counts().to_dict(),
                'potential_targets': self._detect_targets(df)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def _detect_targets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Détecter les colonnes cibles potentielles"""
        targets = []
        
        for col in df.columns:
            score = 0
            # Noms évocateurs
            target_keywords = ['target', 'y', 'label', 'class', 'price', 'score', 'rating']
            if any(kw in col.lower() for kw in target_keywords):
                score += 2
            
            # Dernière colonne
            if col == df.columns[-1]:
                score += 1
            
            # Cardinalité appropriée pour classification
            if df[col].nunique() <= 20 and df[col].nunique() > 1:
                score += 1
            
            # Type approprié
            if df[col].dtype in ['object', 'category', 'int64', 'float64']:
                score += 0.5
            
            if score >= 2:
                targets.append({
                    'column': col,
                    'dtype': str(df[col].dtype),
                    'n_unique': df[col].nunique(),
                    'score': score,
                    'suggested_task': 'classification' if df[col].nunique() <= 20 else 'regression'
                })
        
        # Trier par score
        targets.sort(key=lambda x: x['score'], reverse=True)
        return targets[:5]  # Top 5
    
    def load_data(self, file_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Charger les données d'un fichier uploadé.
        
        Args:
            file_id: ID du fichier
        
        Returns:
            (DataFrame, Métadonnées)
        """
        # Charger les métadonnées
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for file '{file_id}' not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Charger les données
        file_path = Path(metadata['file_path'])
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' not found")
        
        extension = metadata.get('extension', '.csv')
        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return df, metadata
    
    def save_processed(self, 
                       df: pd.DataFrame, 
                       file_id: str,
                       format: str = 'csv') -> Dict[str, Any]:
        """
        Sauvegarder les données prétraitées.
        
        Args:
            df: DataFrame à sauvegarder
            file_id: ID du fichier original
            format: Format de sortie ('csv', 'parquet')
        
        Returns:
            Métadonnées du fichier traité
        """
        # Charger les métadonnées originales
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Créer le nom du fichier traité
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"{timestamp}_{file_id}_processed.{format}"
        processed_path = self.base_dir / 'processed' / processed_filename
        
        # Sauvegarder
        if format == 'csv':
            df.to_csv(processed_path, index=False)
        elif format == 'parquet':
            df.to_parquet(processed_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Mettre à jour les métadonnées
        metadata['processed'] = {
            'file_path': str(processed_path),
            'format': format,
            'shape': df.shape,
            'processed_at': datetime.now().isoformat(),
            'columns': df.columns.tolist()
        }
        
        # Sauvegarder les métadonnées mises à jour
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata['processed']
    
    def list_uploads(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Lister tous les fichiers uploadés.
        
        Args:
            filters: Filtres à appliquer (ex: {'status': 'uploaded'})
        
        Returns:
            Liste des métadonnées
        """
        metadata_dir = self.base_dir / 'metadata'
        uploads = []
        
        for metadata_file in metadata_dir.glob('*.json'):
            if metadata_file.name == 'stats.json':
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Appliquer les filtres
                if filters:
                    match = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Résumé
                uploads.append({
                    'file_id': metadata.get('file_id'),
                    'filename': metadata.get('filename'),
                    'uploaded_at': metadata.get('uploaded_at'),
                    'file_size_mb': metadata.get('file_size_mb'),
                    'status': metadata.get('status', 'unknown'),
                    'rows': metadata.get('analysis', {}).get('shape', {}).get('rows'),
                    'columns': metadata.get('analysis', {}).get('shape', {}).get('columns')
                })
                
            except Exception as e:
                warnings.warn(f"Error reading {metadata_file}: {e}")
        
        # Trier par date
        uploads.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
        return uploads
    
    def get_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Récupérer les métadonnées d'un fichier.
        
        Args:
            file_id: ID du fichier
        
        Returns:
            Métadonnées
        """
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for file '{file_id}' not found")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def update_metadata(self, file_id: str, updates: Dict[str, Any]):
        """
        Mettre à jour les métadonnées d'un fichier.
        
        Args:
            file_id: ID du fichier
            updates: Mises à jour à appliquer
        """
        metadata = self.get_metadata(file_id)
        metadata.update(updates)
        metadata['updated_at'] = datetime.now().isoformat()
        
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def delete_upload(self, file_id: str, delete_files: bool = True):
        """
        Supprimer un fichier uploadé.
        
        Args:
            file_id: ID du fichier
            delete_files: Supprimer aussi les fichiers physiques
        """
        metadata = self.get_metadata(file_id)
        
        if delete_files:
            # Supprimer le fichier raw
            raw_path = Path(metadata.get('file_path', ''))
            if raw_path.exists():
                raw_path.unlink()
            
            # Supprimer le fichier processed
            if 'processed' in metadata:
                processed_path = Path(metadata['processed'].get('file_path', ''))
                if processed_path.exists():
                    processed_path.unlink()
        
        # Supprimer les métadonnées
        metadata_path = self.base_dir / 'metadata' / f"{file_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Mettre à jour les statistiques
        if 'file_size' in metadata:
            self.stats['total_size'] -= metadata['file_size']
        self.stats['total_uploads'] -= 1
        self._save_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques"""
        return {
            'total_uploads': self.stats['total_uploads'],
            'total_size_mb': self.stats['total_size'] / (1024 * 1024),
            'by_type': self.stats['by_type'],
            'disk_usage': self._get_disk_usage()
        }
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Obtenir l'utilisation du disque"""
        usage = {}
        
        for subdir in ['raw', 'processed', 'metadata']:
            dir_path = self.base_dir / subdir
            if dir_path.exists():
                size = sum(f.stat().st_size for f in dir_path.glob('**/*') if f.is_file())
                usage[subdir] = {
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'file_count': len(list(dir_path.glob('**/*'))),
                    'path': str(dir_path)
                }
        
        return usage
    
    def export_metadata(self, format: str = 'csv') -> pd.DataFrame:
        """
        Exporter toutes les métadonnées.
        
        Args:
            format: 'csv' ou 'json'
        
        Returns:
            DataFrame des métadonnées
        """
        uploads = self.list_uploads()
        
        if not uploads:
            return pd.DataFrame()
        
        df = pd.DataFrame(uploads)
        
        if format == 'csv':
            return df
        else:
            return df
    
    def search(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Rechercher dans les métadonnées.
        
        Args:
            query: Chaîne de recherche
            case_sensitive: Sensible à la casse
        
        Returns:
            Fichiers correspondants
        """
        results = []
        query = query if case_sensitive else query.lower()
        
        for metadata_file in (self.base_dir / 'metadata').glob('*.json'):
            if metadata_file.name == 'stats.json':
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Rechercher dans les champs clés
                search_text = json.dumps(metadata)
                if not case_sensitive:
                    search_text = search_text.lower()
                
                if query in search_text:
                    results.append({
                        'file_id': metadata.get('file_id'),
                        'filename': metadata.get('filename'),
                        'uploaded_at': metadata.get('uploaded_at')
                    })
                    
            except Exception:
                continue
        
        return results
    
    def clean_old_files(self, days: int = 30):
        """
        Nettoyer les fichiers plus vieux que X jours.
        
        Args:
            days: Nombre de jours
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        metadata_dir = self.base_dir / 'metadata'
        for metadata_file in metadata_dir.glob('*.json'):
            if metadata_file.name == 'stats.json':
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                uploaded_at = datetime.fromisoformat(metadata.get('uploaded_at', ''))
                if uploaded_at < cutoff:
                    self.delete_upload(metadata['file_id'])
                    
            except Exception:
                continue