"""
Assistant orchestrator for the DataKit AI assistant.

Ce module contient le moteur : embedding, classification,
cache sémantique Redis, RAG (Qdrant + dataset context), LLM.

Il ne gère PAS :
    - l'initialisation paresseuse / les erreurs applicatives
    - le format de réponse exposé à l'API
    - le statut "is_available"

Ces responsabilités appartiennent à services.assistant_service.AssistantService,
qui délègue son travail réel à cette classe.

Flow:

    question
        ↓
    embedding
        ↓
    ML classifier (private / shared)
        ↓
    scope resolution (fallback private -> shared si aucun
    dataset n'est chargé, pour rester cohérent entre
    search() et store())
        ↓
    Redis semantic cache
        ├── hit  -> réponse directe
        └── miss -> RAG
                        │
                        ├── shared  -> Qdrant seul (connaissance générale)
                        └── private -> Qdrant + dataset_context
                        ↓
                       LLM
                        ↓
                 Redis cache (store)
"""

import logging
from typing import Optional

import pandas as pd

from .classifier.classifier_ml import QueryClassifier
from .context.context_manager import ContextManager
from .embeddings.embeddings import EmbeddingModel
from .llm.ollama_client import OllamaClient
from .llm.prompt_builder import PromptBuilder
from .cache.semantic_cache import SemanticCache
from .models.response import AssistantResponse
from .rag.retriever import QdrantRetriever


logger = logging.getLogger(__name__)


class AssistantOrchestrator:
    """Orchestrates the cache-first RAG flow (le moteur, pas la façade)."""

    NO_DATASET_CONTEXT = (
        "Aucun contexte dataset applicable pour cette question."
    )

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier,
        semantic_cache: SemanticCache,
        retriever: QdrantRetriever,
        context_manager: ContextManager,
        prompt_builder: PromptBuilder,
        llm: OllamaClient,
    ) -> None:
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.semantic_cache = semantic_cache
        self.retriever = retriever
        self.context_manager = context_manager
        self.prompt_builder = prompt_builder
        self.llm = llm

    # =============================================================
    # Scope resolution
    # =============================================================

    def _resolve_scope(
        self,
        predicted_scope: str,
        dataset_fingerprint: Optional[str],
    ) -> str:
        """
        Réconcilie la prédiction du classifier avec la disponibilité
        réelle d'un dataset (voir search() vs store() dans
        SemanticCache : le premier tolère l'absence de fingerprint,
        le second lève une ValueError).
        """

        if predicted_scope == "private" and not dataset_fingerprint:
            logger.info(
                "Scope 'private' prédit sans dataset chargé — "
                "fallback vers 'shared'."
            )
            return "shared"

        return predicted_scope

    # =============================================================
    # Dataset context (uniquement pour le scope "private")
    # =============================================================

    def _build_dataset_context(self, scope: str) -> str:
        """
        Injecte le contexte dataset uniquement si la question est
        "private". Une question "shared" n'a pas besoin de l'état
        du dataset courant.
        """

        if scope != "private":
            return self.NO_DATASET_CONTEXT

        return self.context_manager.to_markdown_context()

    # =============================================================
    # Ask
    # =============================================================

    def ask(self, question: str) -> AssistantResponse:
        """Process a user question through the cache-first RAG flow."""

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        embedding = self.embedding_model.encode_query(question)

        predicted_scope = self.classifier.predict(embedding)

        dataset_fingerprint = (
            self.context_manager.get_dataset_fingerprint()
        )

        scope = self._resolve_scope(
            predicted_scope=predicted_scope,
            dataset_fingerprint=dataset_fingerprint,
        )

        cached = self.semantic_cache.search(
            embedding=embedding,
            scope=scope,
            dataset_fingerprint=dataset_fingerprint,
        )

        if cached:
            logger.info("Semantic cache hit: %s", scope)

            return AssistantResponse(
                answer=cached.answer,
                source=f"cache_{scope}",
                similarity=cached.similarity,
            )

        logger.info("Semantic cache miss: %s", scope)

        # RAG : Qdrant toujours consulté, dataset_context conditionnel
        documents = self.retriever.retrieve(question)

        dataset_context = self._build_dataset_context(scope)

        prompt = self.prompt_builder.build(
            question=question,
            documents=documents,
            dataset_context=dataset_context,
        )

        answer = self.llm.generate(prompt)

        self.semantic_cache.store(
            question=question,
            answer=answer,
            embedding=embedding,
            scope=scope,
            dataset_fingerprint=dataset_fingerprint,
        )

        return AssistantResponse(answer=answer, source="rag")

    # =============================================================
    # Dataset / preprocessing context
    # =============================================================

    def update_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        """Update current dataset context."""

        self.context_manager.update_dataset(dataframe, dataset_name)

    def update_preprocessing(
        self,
        operation_name: str,
        columns: list[str],
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a preprocessing operation."""

        self.context_manager.update_preprocessing(
            operation_name,
            columns,
            parameters,
        )