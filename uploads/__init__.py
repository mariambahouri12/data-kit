# uploads/__init__.py
"""
Upload Management Package - Gestion des fichiers uploadés

Ce package gère le téléchargement, le stockage et la gestion
des fichiers de données (CSV, Excel, etc.).
"""

from .manager import UploadManager

__all__ = [
    'UploadManager'
]