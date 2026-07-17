
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
