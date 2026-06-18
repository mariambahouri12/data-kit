import pandas as pd
from pathlib import Path
from typing import Optional, Any
from ...base import BasePreprocessor


class CSVUploader(BasePreprocessor):
    """
    CSV file uploader.
    Allows loading and validating CSV files.
    """
    
    def __init__(self, 
                 sep: str = ',',
                 encoding: str = 'utf-8',
                 **kwargs):
        """
        Args:
            sep: CSV separator
            encoding: File encoding
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
        Load a CSV file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Loaded DataFrame
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
        Load a CSV file from bytes.
        
        Args:
            content: File content in bytes
            filename: File name
        
        Returns:
            Loaded DataFrame
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
        """Get file metadata"""
        return self.metadata
    
    def validate(self) -> dict:
        """
        Validate the loaded DataFrame.
        
        Returns:
            Validation dictionary
        """
        if self.df is None:
            return {'valid': False, 'error': 'No data loaded'}
        
        issues = []
        
        # Check empty columns
        empty_cols = [c for c in self.df.columns if c == '' or c is None]
        if empty_cols:
            issues.append(f"Empty column names: {empty_cols}")
        
        # Check duplicate columns
        dup_cols = self.df.columns[self.df.columns.duplicated()].tolist()
        if dup_cols:
            issues.append(f"Duplicate column names: {dup_cols}")
        
        # Check missing values
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