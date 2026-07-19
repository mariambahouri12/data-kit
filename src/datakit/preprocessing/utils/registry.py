# utils/registry.py
"""
Registre générique pour tout type de composants.
"""

from typing import Dict, Any, Type, List, Optional, Callable
import inspect
import importlib
import pkgutil
import warnings
from ..base import BaseComponent


class Registry:
    """
    Registre générique pour découvrir et gérer des composants.
    Supporte l'auto-découverte et l'enregistrement manuel.
    """
    
    _instance = None
    _components = {}
    _metadata = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._components = {}
        self._metadata = {}
    
    def register(self, 
                 name: str, 
                 component_class: Type,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Enregistrer un composant manuellement.
        
        Args:
            name: Nom du composant
            component_class: Classe du composant
            metadata: Métadonnées optionnelles
        """
        self._components[name.lower()] = component_class
        self._metadata[name.lower()] = metadata or {}
        
        # Ajouter le nom de la classe
        self._metadata[name.lower()]['class_name'] = component_class.__name__
    
    def discover(self, 
                 package_name: str,
                 base_class: Type,
                 exclude_modules: Optional[List[str]] = None):
        """
        Découvrir automatiquement les composants dans un package.
        
        Args:
            package_name: Nom du package (ex: 'models', 'preprocessing')
            base_class: Classe de base à rechercher
            exclude_modules: Modules à exclure
        """
        exclude_modules = exclude_modules or ['base', 'registry', 'factory']
        
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            warnings.warn(f"Package '{package_name}' not found")
            return
        
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name in exclude_modules:
                continue
            
            try:
                module = importlib.import_module(f"{package_name}.{module_info.name}")
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, base_class) and 
                        obj != base_class and 
                        not inspect.isabstract(obj) and
                        obj.__module__ == module.__name__):
                        
                        # Enregistrer le composant
                        component_name = name.replace('Model', '').replace('Preprocessor', '').lower()
                        self.register(component_name, obj)
                        
            except Exception as e:
                warnings.warn(f"Error loading module {module_info.name}: {e}")
    
    def get(self, name: str) -> Optional[Type]:
        """
        Obtenir un composant par son nom.
        
        Args:
            name: Nom du composant
        
        Returns:
            Classe du composant ou None
        """
        return self._components.get(name.lower())
    
    def get_all(self) -> Dict[str, Type]:
        """
        Obtenir tous les composants.
        
        Returns:
            Dictionnaire {nom: classe}
        """
        return self._components.copy()
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Obtenir les métadonnées d'un composant.
        
        Args:
            name: Nom du composant
        
        Returns:
            Métadonnées
        """
        return self._metadata.get(name.lower(), {})
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtenir les métadonnées de tous les composants.
        
        Returns:
            Dictionnaire {nom: métadonnées}
        """
        return self._metadata.copy()
    
    def list(self) -> List[str]:
        """
        Lister tous les composants.
        
        Returns:
            Liste des noms
        """
        return list(self._components.keys())
    
    def exists(self, name: str) -> bool:
        """
        Vérifier si un composant existe.
        
        Args:
            name: Nom du composant
        
        Returns:
            True si le composant existe
        """
        return name.lower() in self._components
    
    def create(self, name: str, **kwargs) -> Optional[BaseComponent]:
        """
        Créer une instance d'un composant.
        
        Args:
            name: Nom du composant
            **kwargs: Paramètres pour l'instanciation
        
        Returns:
            Instance du composant ou None
        """
        component_class = self.get(name)
        if component_class is None:
            return None
        
        return component_class(**kwargs)
    
    def clear(self):
        """Vider le registre"""
        self._components = {}
        self._metadata = {}
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Obtenir des informations détaillées sur un composant.
        
        Args:
            name: Nom du composant
        
        Returns:
            Informations sur le composant
        """
        component_class = self.get(name)
        if component_class is None:
            return {}
        
        info = {
            'name': name,
            'class': component_class.__name__,
            'module': component_class.__module__,
            'metadata': self.get_metadata(name),
            'is_abstract': inspect.isabstract(component_class),
            'methods': [m for m in dir(component_class) if not m.startswith('_')]
        }
        
        # Ajouter les paramètres du constructeur
        sig = inspect.signature(component_class.__init__)
        info['parameters'] = {
            name: {
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty and name not in ['self', 'args', 'kwargs']
            }
            for name, param in sig.parameters.items()
            if name not in ['self', 'args', 'kwargs']
        }
        
        return info
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtenir un résumé du registre.
        
        Returns:
            Résumé
        """
        return {
            'n_components': len(self._components),
            'components': self.list(),
            'metadata': self._metadata
        }
    
    def save_to_file(self, filepath: str):
        """
        Sauvegarder le registre dans un fichier.
        
        Args:
            filepath: Chemin du fichier
        """
        import json
        data = {
            'components': list(self._components.keys()),
            'metadata': self._metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Charger le registre depuis un fichier.
        
        Args:
            filepath: Chemin du fichier
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Note: On ne peut charger que les métadonnées
        # Les classes doivent être importées séparément
        self._metadata = data.get('metadata', {})


# Registry singleton
def get_registry() -> Registry:
    """Obtenir l'instance du registre"""
    return Registry()