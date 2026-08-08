"""
AI Assistant module for DataKit.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import PreprocessingContext

from .llm.ollama_client import OllamaClient
from .llm.prompt_manager import PromptManager
from .llm.response_formatter import ResponseFormatter

from .rag.embeddings import EmbeddingModel
from .rag.retriever import Retriever
from .rag.vector_store import VectorStore

from .rag.file_catalog import FileCatalog
from .rag.router import DocumentRouter
from .rag.document_loader import DocumentLoader
from .rag.metadata_store import MetadataStore

from .agents.langgraph_agent import LangGraphAgent
from .agents.graph import create_graph


logger = logging.getLogger(__name__)


def _build_rag_core():
    """
    Build the embedding, vector store and retriever components.
    """

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_model,
        vector_store,
    )

    return (
        embedding_model,
        vector_store,
        retriever,
    )


def _build_document_router(
    llm_client,
    knowledge_base_path,
):
    """
    Build the document routing components.
    """

    if not knowledge_base_path:

        logger.warning(
            "No knowledge base path provided. "
            "Running without document routing."
        )

        return (
            None,
            None,
            None,
            None,
        )

    try:

        file_catalog = FileCatalog(
            knowledge_base_path
        )

        file_catalog.build()

        document_loader = DocumentLoader(
            knowledge_base_path
        )

        metadata_store = MetadataStore()

        router = DocumentRouter(
            llm_client,
            file_catalog,
        )

        logger.info(
            "Document Router initialized."
        )

        return (
            router,
            file_catalog,
            document_loader,
            metadata_store,
        )

    except Exception:

        logger.exception(
            "Router initialization failed."
        )

        return (
            None,
            None,
            None,
            None,
        )


def _build_or_load_index(
    retriever,
    vector_store,
    file_catalog,
    document_loader,
    metadata_store,
):
    """
    Build the FAISS index if it does not exist,
    otherwise load the existing index.
    """

    if file_catalog is None:
        return

    # =========================================================
    # Build index
    # =========================================================

    if not vector_store.exists:

        logger.info(
            "Building FAISS index from Markdown documents..."
        )

        documents = file_catalog.get_documents()

        if not documents:
            logger.warning(
                "No Markdown documents found."
            )
            return

        all_files = [
            document.get("file", "")
            for document in documents
            if document.get("file")
        ]

        if not all_files:
            return

        all_documents = document_loader.load(
            all_files
        )

        if not all_documents:
            logger.warning(
                "No documents could be loaded."
            )
            return

        all_documents = metadata_store.enrich(
            all_documents
        )

        retriever.initialize(
            all_documents
        )

        logger.info(
            "FAISS built with %s documents.",
            len(all_documents),
        )

    # =========================================================
    # Load existing index
    # =========================================================

    else:

        logger.info(
            "FAISS index already exists. Loading..."
        )

        if vector_store.load():

            retriever.mark_ready()

            logger.info(
                "FAISS loaded with %s documents.",
                len(vector_store.metadata),
            )


def _load_index_fallback(
    retriever,
    vector_store,
):
    """
    Fallback attempt to load an existing vector store.
    """

    if retriever.is_ready:
        return

    try:

        if vector_store.load():

            retriever.mark_ready()

            logger.info(
                "Vector store loaded from disk."
            )

    except Exception:

        logger.exception(
            "Could not load vector store."
        )


def create_assistant(
    model_name: str = "mistral",
    knowledge_base_path: Optional[str] = None,
    context_manager: Optional[ContextManager] = None,
) -> dict:
    """
    Create a fully configured Agentic RAG AI Assistant
    using LangGraph orchestration.
    """

    logger.info(
        "Creating DataKit AI Assistant with LangGraph..."
    )

    # =========================================================
    # RAG core
    # =========================================================

    (
        embedding_model,
        vector_store,
        retriever,
    ) = _build_rag_core()

    # =========================================================
    # LLM / prompt / formatting
    # =========================================================

    prompt_manager = PromptManager()

    response_formatter = ResponseFormatter()

    llm_client = OllamaClient(
        model_name=model_name
    )

    # =========================================================
    # Document routing
    # =========================================================

    (
        router,
        file_catalog,
        document_loader,
        metadata_store,
    ) = _build_document_router(
        llm_client,
        knowledge_base_path,
    )

    # =========================================================
    # FAISS index
    # =========================================================

    _build_or_load_index(
        retriever,
        vector_store,
        file_catalog,
        document_loader,
        metadata_store,
    )

    # =========================================================
    # Shared context
    # =========================================================

    if context_manager is None:

        context_manager = ContextManager(
            DatasetContextBuilder(),
            PreprocessingContext(),
        )

    # =========================================================
    # LLM availability
    # =========================================================

    status = llm_client.check_connection()

    if status.get("status"):

        logger.info(
            "Ollama ready with model: %s",
            model_name,
        )

    else:

        logger.warning(
            "%s",
            status.get("message"),
        )

    # =========================================================
    # Fallback FAISS loading
    # =========================================================

    _load_index_fallback(
        retriever,
        vector_store,
    )

    # =========================================================
    # LangGraph
    # =========================================================

    graph = create_graph(
        retriever=retriever,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        router=router,
        context_manager=context_manager,
        response_formatter=response_formatter,
    )

    agent = LangGraphAgent(
        graph
    )

    logger.info(
        "LangGraph Agent created successfully."
    )

    return {
        "agent": agent,
        "context_manager": context_manager,
        "llm_client": llm_client,
        "retriever": retriever,
        "vector_store": vector_store,
        "router": router,
        "file_catalog": file_catalog,
        "document_loader": document_loader,
        "metadata_store": metadata_store,
        "prompt_manager": prompt_manager,
        "response_formatter": response_formatter,
    }


__all__ = [
    "ContextManager",
    "DatasetContextBuilder",
    "PreprocessingContext",
    "OllamaClient",
    "PromptManager",
    "ResponseFormatter",
    "EmbeddingModel",
    "Retriever",
    "VectorStore",
    "FileCatalog",
    "DocumentRouter",
    "DocumentLoader",
    "MetadataStore",
    "LangGraphAgent",
    "create_graph",
]