"""
Assistant orchestrator for the DataKit AI assistant.

This module contains the engine: embedding, classification,
Redis semantic cache, RAG (hybrid Qdrant + BM25 search + dataset
context), LLM.

It does NOT handle:
    - lazy initialization / application-level errors
    - the response format exposed by the API
    - the "is_available" status

These responsibilities belong to services.assistant_service.AssistantService,
which delegates the actual work to this class.

Flow:

    question
        |
    embedding
        |
    ML classifier (private / shared)
        |
    scope resolution (fallback private -> shared if no dataset is
    loaded, to stay consistent between search() and store())
        |
    Redis semantic cache
        |-- hit  -> direct answer
        |-- miss -> RAG
                        |
                        |-- shared  -> hybrid search only (general knowledge)
                        |-- private -> hybrid search + dataset_context
                        |
                       LLM
                        |
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
from .rag.hybrid_retriever import HybridRetriever


logger = logging.getLogger(__name__)


class AssistantOrchestrator:
    """Orchestrates the cache-first RAG flow (the engine, not the facade)."""

    NO_DATASET_CONTEXT = "No applicable dataset context for this question."

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier,
        semantic_cache: SemanticCache,
        retriever: HybridRetriever,
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
        Reconciles the classifier's prediction with the actual
        availability of a dataset (see search() vs store() in
        SemanticCache: the former tolerates a missing fingerprint,
        the latter raises a ValueError).
        """

        if predicted_scope == "private" and not dataset_fingerprint:
            logger.info(
                "Scope 'private' predicted without a loaded dataset - "
                "falling back to 'shared'."
            )
            return "shared"

        return predicted_scope

    # =============================================================
    # Dataset context (only for the "private" scope)
    # =============================================================

    def _build_dataset_context(self, scope: str) -> str:
        """
        Injects the dataset context only when the question is
        "private". A "shared" question does not need the state of
        the current dataset.
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

        dataset_fingerprint = self.context_manager.get_dataset_fingerprint()

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

        # RAG: hybrid retrieval is always used, dataset_context is conditional.
        # The embedding computed above is reused here to avoid encoding
        # the same query a second time.
        documents = self.retriever.retrieve(question, embedding=embedding)

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