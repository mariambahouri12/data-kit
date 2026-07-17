# preprocessing/utils/visualizers.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings


class DataVisualizer:
    """
    Visualiseur de données pour l'exploration et l'analyse.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn'):
        """
        Args:
            figsize: Taille par défaut des figures
            style: Style de matplotlib
        """
        self.figsize = figsize
        self.style = style
        
        # Définir le style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_missing_values(self, X: pd.DataFrame, title: str = "Missing Values") -> plt.Figure:
        """Visualiser les valeurs manquantes"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        missing_pct = X.isnull().sum() / len(X) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) == 0:
            ax.text(0.5, 0.5, "✅ No missing values", 
                   ha='center', va='center', fontsize=16)
            ax.set_axis_off()
        else:
            bars = ax.barh(range(len(missing_pct)), missing_pct.values, color='#FF6B6B')
            ax.set_yticks(range(len(missing_pct)))
            ax.set_yticklabels(missing_pct.index, fontsize=10)
            ax.set_xlabel('Missing Percentage (%)', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Ajouter les valeurs
            for i, (bar, val) in enumerate(zip(bars, missing_pct.values)):
                ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, X: pd.DataFrame, 
                                method: str = 'pearson',
                                title: str = "Correlation Matrix") -> plt.Figure:
        """Visualiser la matrice de corrélation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "⚠️ Not enough numeric columns for correlation",
                   ha='center', va='center', fontsize=16)
            ax.set_axis_off()
            return fig
        
        corr = X[numeric_cols].corr(method=method)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                    cmap='RdBu_r', center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, X: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         n_cols: int = 3,
                         title: str = "Distribution") -> plt.Figure:
        """Visualiser les distributions des colonnes"""
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(columns) > 9:
                columns = columns[:9]
        
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, col in enumerate(columns):
            if i >= len(axes):
                break
            
            ax = axes[i]
            X[col].hist(bins=30, ax=ax, color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax.set_title(col, fontsize=11)
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            
            # Ajouter la statistique
            stats = f"μ={X[col].mean():.2f}\nσ={X[col].std():.2f}"
            ax.text(0.95, 0.95, stats, transform=ax.transAxes,
                   va='top', ha='right', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Supprimer les axes vides
        for i in range(len(columns), len(axes)):
            axes[i].set_axis_off()
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_boxplot(self, X: pd.DataFrame,
                    columns: Optional[List[str]] = None,
                    title: str = "Boxplot") -> plt.Figure:
        """Visualiser les boîtes à moustaches"""
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) > 15:
            columns = columns[:15]
        
        fig, ax = plt.subplots(figsize=(max(8, len(columns)*0.8), 6))
        
        # Préparer les données
        data_to_plot = [X[col].dropna() for col in columns]
        bp = ax.boxplot(data_to_plot, labels=columns, patch_artist=True)
        
        # Colorier les boîtes
        for patch in bp['boxes']:
            patch.set_facecolor('#4ECDC4')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Rotation des labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_outliers(self, X: pd.DataFrame,
                     columns: Optional[List[str]] = None,
                     title: str = "Outlier Detection") -> plt.Figure:
        """Visualiser les outliers"""
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) > 3:
            columns = columns[:3]
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*4, 4))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # Calculer IQR
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Tracer les données
            ax.scatter(X.index, X[col], alpha=0.5, s=30, color='#4ECDC4')
            
            # Tracer les limites
            ax.axhline(y=lower, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=upper, color='red', linestyle='--', alpha=0.7)
            
            # Colorier les outliers
            outliers = (X[col] < lower) | (X[col] > upper)
            if outliers.sum() > 0:
                ax.scatter(X.index[outliers], X[col][outliers], 
                          color='red', s=50, alpha=0.7, label='Outliers')
            
            ax.set_title(f'{col}\n({outliers.sum()} outliers)', fontsize=11)
            ax.set_xlabel('Index', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_pca(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                 n_components: int = 2,
                 title: str = "PCA Visualization") -> plt.Figure:
        """Visualiser la réduction PCA"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "⚠️ Not enough numeric columns for PCA",
                   ha='center', va='center', fontsize=16)
            ax.set_axis_off()
            return fig
        
        # Appliquer PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X[numeric_cols])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if n_components == 2:
            # 2D plot
            if y is not None:
                # Colorier par classe
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color='#4ECDC4')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            
        else:
            # 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                                    c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                          alpha=0.7, color='#4ECDC4')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
        
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_class_balance(self, y: pd.Series, title: str = "Class Balance") -> plt.Figure:
        """Visualiser l'équilibre des classes"""
        counts = y.value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        axes[0].bar(counts.index.astype(str), counts.values, color='#4ECDC4')
        axes[0].set_title('Class Distribution', fontsize=12)
        axes[0].set_xlabel('Class', fontsize=10)
        axes[0].set_ylabel('Count', fontsize=10)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        # Ajouter les valeurs
        for i, (label, count) in enumerate(counts.items()):
            axes[0].text(i, count + max(counts)*0.02, str(count), 
                        ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#A8E6CF', '#FFB74D']
        axes[1].pie(counts.values, labels=counts.index.astype(str), 
                   autopct='%1.1f%%', colors=colors[:len(counts)])
        axes[1].set_title('Class Proportions', fontsize=12)
        
        # Ajouter l'info d'imbalance
        if len(counts) >= 2:
            imbalance_ratio = counts.max() / counts.min()
            fig.suptitle(f'{title} (Imbalance Ratio: {imbalance_ratio:.2f})', 
                        fontsize=14, y=1.02)
        else:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importances: Dict[str, float],
                                top_n: int = 20,
                                title: str = "Feature Importance") -> plt.Figure:
        """Visualiser l'importance des features"""
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        if top_n:
            sorted_importances = sorted_importances[:top_n]
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_importances)*0.3)))
        
        features = [item[0] for item in sorted_importances]
        scores = [item[1] for item in sorted_importances]
        
        bars = ax.barh(features, scores, color='#4ECDC4')
        
        # Colorier le meilleur
        bars[0].set_color('#FF6B6B')
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        
        # Ajouter les valeurs
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + max(scores)*0.01, i, f'{score:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_data_summary(self, X: pd.DataFrame) -> plt.Figure:
        """Visualiser un résumé complet des données"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Types de données
        ax1 = plt.subplot(2, 3, 1)
        dtypes = X.dtypes.value_counts()
        ax1.pie(dtypes.values, labels=[str(t) for t in dtypes.index], autopct='%1.1f%%')
        ax1.set_title('Data Types', fontsize=12)
        
        # 2. Valeurs manquantes
        ax2 = plt.subplot(2, 3, 2)
        missing = X.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            ax2.bar(missing.index, missing.values, color='#FF6B6B')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_title(f'Missing Values (n={len(missing)})', fontsize=12)
            ax2.set_ylabel('Count', fontsize=10)
        else:
            ax2.text(0.5, 0.5, '✅ No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_axis_off()
        
        # 3. Shape
        ax3 = plt.subplot(2, 3, 3)
        ax3.text(0.1, 0.7, f"Rows: {X.shape[0]}", fontsize=14, va='center')
        ax3.text(0.1, 0.5, f"Columns: {X.shape[1]}", fontsize=14, va='center')
        ax3.text(0.1, 0.3, f"Memory: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB", 
                fontsize=14, va='center')
        ax3.set_axis_off()
        ax3.set_title('Dataset Info', fontsize=12)
        
        # 4. Statistiques numériques
        ax4 = plt.subplot(2, 3, 4)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = X[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
            ax4.table(cellText=stats.round(2).values[:5],
                     rowLabels=stats.index[:5],
                     colLabels=stats.columns,
                     cellLoc='center', loc='center')
            ax4.set_axis_off()
            ax4.set_title('Numeric Stats (first 5)', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No numeric columns', ha='center', va='center', fontsize=14)
            ax4.set_axis_off()
        
        # 5. Catégories
        ax5 = plt.subplot(2, 3, 5)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            n_categories = {col: X[col].nunique() for col in categorical_cols}
            top_cats = sorted(n_categories.items(), key=lambda x: x[1], reverse=True)[:5]
            ax5.bar([c[0] for c in top_cats], [c[1] for c in top_cats], color='#FFE66D')
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
            ax5.set_title(f'Categorical Columns (n={len(categorical_cols)})', fontsize=12)
            ax5.set_ylabel('Categories', fontsize=10)
        else:
            ax5.text(0.5, 0.5, 'No categorical columns', ha='center', va='center', fontsize=14)
            ax5.set_axis_off()
        
        # 6. Doublons
        ax6 = plt.subplot(2, 3, 6)
        n_duplicates = X.duplicated().sum()
        if n_duplicates > 0:
            ax6.text(0.1, 0.7, f"Duplicate rows: {n_duplicates}", fontsize=14, va='center')
            ax6.text(0.1, 0.5, f"Percentage: {n_duplicates/len(X)*100:.2f}%", fontsize=14, va='center')
            ax6.set_axis_off()
        else:
            ax6.text(0.5, 0.5, '✅ No Duplicates', ha='center', va='center', fontsize=14)
            ax6.set_axis_off()
        ax6.set_title('Duplicates', fontsize=12)
        
        plt.suptitle('Data Summary Report', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig