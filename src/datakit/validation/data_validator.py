
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings


class DataValidator:
    """
    Validateur de données pour vérifier la qualité et la cohérence.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Afficher les messages de validation
        """
        self.verbose = verbose
        self.validation_results = {}
    
    def validate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Valider complètement un DataFrame.
        
        Args:
            X: DataFrame à valider
            y: Target (optionnel)
        
        Returns:
            Dictionnaire des résultats de validation
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {},
            'checks': {}
        }
        
        # Liste des vérifications
        checks = [
            self.check_missing_values,
            self.check_duplicates,
            self.check_column_names,
            self.check_data_types,
            self.check_outliers,
            self.check_constant_columns,
            self.check_high_cardinality,
            self.check_correlations
        ]
        
        for check in checks:
            check_name = check.__name__.replace('check_', '')
            try:
                result = check(X, y)
                results['checks'][check_name] = result
                
                if not result.get('passed', True):
                    results['is_valid'] = False
                    if result.get('severity') == 'error':
                        results['errors'].append(result.get('message', ''))
                    else:
                        results['warnings'].append(result.get('message', ''))
                
                if 'info' in result:
                    results['info'][check_name] = result['info']
                    
            except Exception as e:
                results['is_valid'] = False
                results['errors'].append(f"Error in {check_name}: {str(e)}")
        
        if self.verbose:
            self._print_report(results)
        
        self.validation_results = results
        return results
    
    def check_missing_values(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les valeurs manquantes"""
        missing_pct = X.isnull().sum() / len(X) * 100
        total_missing = X.isnull().sum().sum()
        total_cells = X.size
        
        columns_with_missing = missing_pct[missing_pct > 0].index.tolist()
        columns_high_missing = missing_pct[missing_pct > 20].index.tolist()
        
        result = {
            'passed': True,
            'severity': 'warning',
            'info': {
                'total_missing': total_missing,
                'missing_percentage': (total_missing / total_cells) * 100,
                'columns_with_missing': len(columns_with_missing),
                'columns_high_missing': len(columns_high_missing)
            }
        }
        
        if total_missing > 0:
            if columns_high_missing:
                result['passed'] = False
                result['severity'] = 'error'
                result['message'] = f"Columns with >20% missing values: {columns_high_missing}"
            elif len(columns_with_missing) > 0:
                result['message'] = f"Columns with missing values: {columns_with_missing[:5]}{'...' if len(columns_with_missing) > 5 else ''}"
        
        return result
    
    def check_duplicates(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les doublons"""
        n_duplicates = X.duplicated().sum()
        
        result = {
            'passed': True,
            'severity': 'warning',
            'info': {
                'n_duplicates': n_duplicates,
                'duplicate_percentage': (n_duplicates / len(X)) * 100
            }
        }
        
        if n_duplicates > 0:
            result['message'] = f"Found {n_duplicates} duplicate rows ({n_duplicates/len(X)*100:.1f}%)"
            if n_duplicates / len(X) > 0.1:
                result['passed'] = False
                result['severity'] = 'error'
        
        return result
    
    def check_column_names(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les noms de colonnes"""
        issues = []
        
        for col in X.columns:
            # Vérifier les espaces
            if ' ' in col:
                issues.append(f"Column '{col}' contains spaces")
            # Vérifier les caractères spéciaux
            if not col.isidentifier():
                issues.append(f"Column '{col}' contains special characters")
            # Vérifier la longueur
            if len(col) > 50:
                issues.append(f"Column '{col}' is too long ({len(col)} chars)")
        
        result = {
            'passed': len(issues) == 0,
            'severity': 'warning' if issues else 'info',
            'info': {
                'n_issues': len(issues)
            }
        }
        
        if issues:
            result['message'] = f"Column naming issues: {issues[:3]}{'...' if len(issues) > 3 else ''}"
        
        return result
    
    def check_data_types(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les types de données"""
        dtypes = X.dtypes
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Vérifier les colonnes object qui devraient être catégorielles
        potential_categorical = []
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].nunique() < len(X) * 0.1:  # Moins de 10% de valeurs uniques
                potential_categorical.append(col)
        
        result = {
            'passed': True,
            'severity': 'info',
            'info': {
                'n_numeric': len(numeric_cols),
                'n_categorical': len(categorical_cols),
                'n_datetime': len(datetime_cols),
                'potential_categorical': potential_categorical
            }
        }
        
        if potential_categorical:
            result['message'] = f"Columns that could be categorical: {potential_categorical[:5]}{'...' if len(potential_categorical) > 5 else ''}"
        
        return result
    
    def check_outliers(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les outliers"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outliers_count = {}
        
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()
            if n_outliers > 0:
                outliers_count[col] = n_outliers
        
        result = {
            'passed': True,
            'severity': 'warning',
            'info': {
                'columns_with_outliers': len(outliers_count),
                'total_outliers': sum(outliers_count.values())
            }
        }
        
        if outliers_count:
            top_cols = sorted(outliers_count.items(), key=lambda x: x[1], reverse=True)[:3]
            result['message'] = f"Outliers found in columns: {', '.join([f'{col}({count})' for col, count in top_cols])}{'...' if len(outliers_count) > 3 else ''}"
        
        return result
    
    def check_constant_columns(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les colonnes constantes"""
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)
        
        result = {
            'passed': len(constant_cols) == 0,
            'severity': 'error' if constant_cols else 'info',
            'info': {
                'constant_columns': constant_cols,
                'n_constant': len(constant_cols)
            }
        }
        
        if constant_cols:
            result['message'] = f"Constant columns found: {constant_cols}"
        
        return result
    
    def check_high_cardinality(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les colonnes à haute cardinalité"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        high_cardinality = {}
        
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique > 50:
                high_cardinality[col] = n_unique
        
        result = {
            'passed': True,
            'severity': 'warning',
            'info': {
                'high_cardinality_columns': high_cardinality,
                'n_high_cardinality': len(high_cardinality)
            }
        }
        
        if high_cardinality:
            top_cols = sorted(high_cardinality.items(), key=lambda x: x[1], reverse=True)[:3]
            result['message'] = f"High cardinality columns: {', '.join([f'{col}({n})' for col, n in top_cols])}{'...' if len(high_cardinality) > 3 else ''}"
        
        return result
    
    def check_correlations(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Vérifier les corrélations fortes"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        high_corr = []
        
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))
        
        result = {
            'passed': True,
            'severity': 'warning',
            'info': {
                'high_correlations': len(high_corr),
                'correlation_pairs': high_corr
            }
        }
        
        if high_corr:
            result['message'] = f"High correlations found: {len(high_corr)} pairs > 0.8"
        
        return result
    
    def _print_report(self, results: Dict[str, Any]):
        """Afficher un rapport de validation"""
        print("\n" + "="*60)
        print("📊 DATA VALIDATION REPORT")
        print("="*60)
        
        status = "✅ PASSED" if results['is_valid'] else "❌ FAILED"
        print(f"Status: {status}")
        print("-"*60)
        
        if results['errors']:
            print("🔴 ERRORS:")
            for error in results['errors']:
                print(f"  • {error}")
            print("-"*60)
        
        if results['warnings']:
            print("🟡 WARNINGS:")
            for warning in results['warnings']:
                print(f"  • {warning}")
            print("-"*60)
        
        if results['info']:
            print("ℹ️ INFO:")
            for key, value in results['info'].items():
                print(f"  • {key}: {value}")
            print("-"*60)
        
        print("="*60 + "\n")