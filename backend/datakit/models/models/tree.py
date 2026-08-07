"""
Modèles à base d'arbres de décision.

`DecisionTreeModel` et `RandomForestModel` sont deux classes distinctes,
chacune définissant explicitement son schéma de paramètres et sa méthode
`_create_model` — sur le même modèle que linear.py et ensemble.py — plutôt
qu'une seule classe qui devinait son comportement en inspectant
`self.__class__.__name__` (fragile : cassé par tout renommage futur).
"""
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseModel
from .registry import register_model


class TreeModel(BaseModel):
    """Classe de base partageant le schéma commun aux modèles à base d'arbres.
    Non enregistrée directement (pas de @register_model) : c'est une base,
    pas un modèle utilisable telle quelle."""

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'max_depth': {
                'type': 'int', 'default': None, 'min': 1, 'max': 100,
                'description': 'Maximum tree depth (None = unlimited)', 'category': 'tree'
            },
            'min_samples_split': {
                'type': 'int', 'default': 2, 'min': 2, 'max': 100,
                'description': 'Minimum samples to split a node', 'category': 'tree'
            },
            'min_samples_leaf': {
                'type': 'int', 'default': 1, 'min': 1, 'max': 100,
                'description': 'Minimum samples per leaf', 'category': 'tree'
            },
            'max_features': {
                'type': 'str', 'default': 'sqrt', 'choices': ['auto', 'sqrt', 'log2', None],
                'description': 'Number of features to consider', 'category': 'tree'
            },
            'random_state': {
                'type': 'int', 'default': 42,
                'description': 'Random seed', 'category': 'reproducibility'
            }
        }

    def _create_model(self, **params):
        raise NotImplementedError("Subclasses must implement _create_model")


@register_model
class DecisionTreeModel(TreeModel):
    """Arbre de décision simple."""

    def _create_model(self, **params):
        if self.task == "classification":
            return DecisionTreeClassifier(**params)
        return DecisionTreeRegressor(**params)


@register_model
class RandomForestModel(TreeModel):
    """Forêt aléatoire (ensemble d'arbres de décision)."""

    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema.update({
            'n_estimators': {
                'type': 'int', 'default': 100, 'min': 10, 'max': 1000,
                'description': 'Number of trees', 'category': 'ensemble'
            },
            'bootstrap': {
                'type': 'bool', 'default': True,
                'description': 'Use bootstrap samples', 'category': 'ensemble'
            },
            'n_jobs': {
                'type': 'int', 'default': -1, 'min': -1, 'max': 16,
                'description': 'Number of parallel jobs (-1 = all cores)', 'category': 'performance'
            }
        })
        return schema

    def _create_model(self, **params):
        if self.task == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)
