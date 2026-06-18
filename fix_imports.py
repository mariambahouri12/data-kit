# fix_imports.py
"""
Script pour corriger les imports manquants dans les modules.
Crée les fichiers __init__.py manquants et les imports nécessaires.
"""

import os
from pathlib import Path


def create_missing_upload():
    """Créer le module upload manquant"""
    upload_dir = Path("preprocessing/tabular/upload")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    init_file = upload_dir / "__init__.py"
    with open(init_file, "w") as f:
        f.write('''
"""
Module d'upload pour le preprocessing tabulaire.
"""

from .csv_uploader import CSVUploader

__all__ = ['CSVUploader']
''')
    
    csv_uploader = upload_dir / "csv_uploader.py"
    with open(csv_uploader, "w") as f:
        f.write('''
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from ..base import BasePreprocessor


class CSVUploader(BasePreprocessor):
    """
    Uploader de fichiers CSV.
    Permet de charger et valider des fichiers CSV.
    """
    
    def __init__(self, 
                 sep: str = ',',
                 encoding: str = 'utf-8',
                 **kwargs):
        """
        Args:
            sep: Séparateur CSV
            encoding: Encodage du fichier
        """
        super().__init__(**kwargs)
        self.sep = sep
        self.encoding = encoding
        self.df = None
        self.metadata = {}
    
    def _fit(self, X, y=None):
        pass
    
    def _transform(self, X):
        return X
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Charger un fichier CSV.
        
        Args:
            file_path: Chemin du fichier
        
        Returns:
            DataFrame chargé
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.df = pd.read_csv(file_path, sep=self.sep, encoding=self.encoding)
        
        self.metadata = {
            'filename': file_path.name,
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        }
        
        return self.df
    
    def load_from_bytes(self, content: bytes, filename: str = 'upload.csv') -> pd.DataFrame:
        """
        Charger un fichier CSV depuis des bytes.
        
        Args:
            content: Contenu du fichier en bytes
            filename: Nom du fichier
        
        Returns:
            DataFrame chargé
        """
        import io
        
        self.df = pd.read_csv(io.BytesIO(content), sep=self.sep, encoding=self.encoding)
        
        self.metadata = {
            'filename': filename,
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        }
        
        return self.df
    
    def get_metadata(self) -> dict:
        """Obtenir les métadonnées du fichier chargé"""
        return self.metadata
    
    def validate(self) -> dict:
        """
        Valider le DataFrame chargé.
        
        Returns:
            Dictionnaire de validation
        """
        if self.df is None:
            return {'valid': False, 'error': 'No data loaded'}
        
        issues = []
        
        # Vérifier les colonnes vides
        empty_cols = [c for c in self.df.columns if c == '' or c is None]
        if empty_cols:
            issues.append(f"Empty column names: {empty_cols}")
        
        # Vérifier les colonnes dupliquées
        dup_cols = self.df.columns[self.df.columns.duplicated()].tolist()
        if dup_cols:
            issues.append(f"Duplicate column names: {dup_cols}")
        
        # Vérifier les valeurs manquantes
        missing_pct = self.df.isnull().sum().sum() / self.df.size * 100
        if missing_pct > 50:
            issues.append(f"High missing percentage: {missing_pct:.1f}%")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_rows': len(self.df),
            'n_cols': len(self.df.columns),
            'missing_pct': missing_pct
        }
''')
    
    print(f"✅ Created upload module: {upload_dir}")


def fix_utils_base():
    """Corriger l'import base dans utils"""
    utils_base = Path("utils/base.py")
    
    if not utils_base.exists():
        # Créer le fichier base.py manquant
        utils_base.parent.mkdir(parents=True, exist_ok=True)
        with open(utils_base, "w") as f:
            f.write('''
"""
Base classes for utils modules.
"""

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Classe de base pour les composants du registre"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtenir le nom du composant"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Obtenir la description du composant"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Obtenir la version du composant"""
        pass
''')
        print(f"✅ Created: {utils_base}")
    
    # Corriger l'import dans utils/registry.py si nécessaire
    registry_file = Path("utils/registry.py")
    if registry_file.exists():
        content = registry_file.read_text()
        if 'from .base import BaseComponent' not in content:
            # Ajouter l'import
            new_content = 'from .base import BaseComponent\n' + content
            registry_file.write_text(new_content)
            print(f"✅ Fixed import in: {registry_file}")


def create_missing_validators():
    """Créer les fichiers validators et visualizers manquants"""
    
    # validators.py
    validators_file = Path("preprocessing/utils/validators.py")
    if not validators_file.exists():
        validators_file.parent.mkdir(parents=True, exist_ok=True)
        with open(validators_file, "w") as f:
            f.write('''
"""
Validators for data quality checks.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DataValidator:
    """Validateur de données"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validation_results = {}
    
    def validate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Valider le DataFrame"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {},
            'checks': {}
        }
        
        # Vérifier les valeurs manquantes
        missing_pct = X.isnull().sum().sum() / X.size * 100
        results['info']['missing_pct'] = missing_pct
        if missing_pct > 50:
            results['warnings'].append(f"High missing percentage: {missing_pct:.1f}%")
        
        # Vérifier les doublons
        n_duplicates = X.duplicated().sum()
        results['info']['duplicates'] = n_duplicates
        if n_duplicates > 0:
            results['warnings'].append(f"Found {n_duplicates} duplicate rows")
        
        # Vérifier les colonnes constantes
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            results['warnings'].append(f"Constant columns: {constant_cols}")
        
        results['info']['n_rows'] = len(X)
        results['info']['n_cols'] = len(X.columns)
        
        return results
''')
        print(f"✅ Created: {validators_file}")
    
    # visualizers.py
    visualizers_file = Path("preprocessing/utils/visualizers.py")
    if not visualizers_file.exists():
        with open(visualizers_file, "w") as f:
            f.write('''
"""
Visualizers for data exploration.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    """Visualiseur de données"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_distribution(self, X: pd.DataFrame, n_cols: int = 3):
        """Visualiser les distributions"""
        columns = X.select_dtypes(include=[np.number]).columns
        n_cols = min(n_cols, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                X[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(col)
        
        for i in range(len(columns), len(axes)):
            axes[i].set_axis_off()
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, X: pd.DataFrame):
        """Visualiser la matrice de corrélation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Not enough numeric columns", ha='center', va='center')
            return fig
        
        corr = X[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        return fig
    
    def plot_outliers(self, X: pd.DataFrame):
        """Visualiser les outliers"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        n_cols = min(3, len(numeric_cols))
        
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*4, 4))
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            axes[i].boxplot(X[col].dropna())
            axes[i].set_title(col)
        
        plt.tight_layout()
        return fig
    
    def plot_pca(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Visualiser la PCA"""
        from sklearn.decomposition import PCA
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Not enough numeric columns for PCA", ha='center', va='center')
            return fig
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X[numeric_cols])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        if y is not None:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
            plt.colorbar(scatter)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        return fig
''')
        print(f"✅ Created: {visualizers_file}")


def fix_all():
    """Corriger tous les imports manquants"""
    print("🔧 Fixing missing imports...")
    
    create_missing_upload()
    fix_utils_base()
    create_missing_validators()
    
    print("\n✅ All fixes applied!")
    print("\n📁 Structure créée:")
    print("  • preprocessing/tabular/upload/")
    print("  • preprocessing/tabular/upload/__init__.py")
    print("  • preprocessing/tabular/upload/csv_uploader.py")
    print("  • utils/base.py")
    print("  • preprocessing/utils/validators.py")
    print("  • preprocessing/utils/visualizers.py")


if __name__ == "__main__":
    fix_all()