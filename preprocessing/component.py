from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Interface pour les composants exposés dans un registre."""

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def get_description(self) -> str:
        ...

    @abstractmethod
    def get_version(self) -> str:
        ...