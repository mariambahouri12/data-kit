"""
AI Assistant module for DataKit.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    "logger"
]


def _build_rag_core():
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    retriever = Retriever(embedding_model, vector_store)
    return embedding_model, vector_store, retriever


def _build_document_router(llm_client, knowledge_base_path):
    if not knowledge_base_path:
        logger.warning(
            "⚠️ No knowledge base path provided. "
            "Running without document routing."
        )
        return None, None, None, None

    try:
        file_catalog = FileCatalog(knowledge_base_path)
        file_catalog.build()

        document_loader = DocumentLoader(knowledge_base_path)
        metadata_store = MetadataStore()
        router = DocumentRouter(llm_client, file_catalog)

        logger.info("✅ Document Router initialized")
        return router, file_catalog, document_loader, metadata_store

    except Exception:
        logger.exception("⚠️ Router initialization failed")
        return None, None, None, None


def _build_or_load_index(retriever, vector_store, file_catalog, document_loader, metadata_store):
    if file_catalog is None:
        return

    if not vector_store.exists:
        logger.info("🔄 Building FAISS index from documents...")

        documents = file_catalog.get_documents()
        if not documents:
            return

        if isinstance(documents[0], dict):
            all_files = [d.get("file", "") for d in documents if d.get("file")]
        else:
            all_files = documents

        if not all_files:
            return

        all_documents = document_loader.load(all_files)
        if not all_documents:
            return

        all_documents = metadata_store.enrich(all_documents)
        retriever.initialize(all_documents)
        logger.info(f"✅ FAISS built with {len(all_documents)} documents")

    else:
        logger.info("📂 FAISS index already exists, loading...")
        if vector_store.load():
            retriever.mark_ready()
            logger.info(f"✅ FAISS loaded with {len(vector_store.metadata)} documents")


def _load_index_fallback(retriever, vector_store):
    if retriever.is_ready:
        return

    try:
        if vector_store.load():
            retriever.mark_ready()
            logger.info("✅ Vector store loaded from disk")
    except Exception:
        logger.exception("⚠️ Could not load vector store")


def create_assistant(
    model_name: str = "mistral",
    knowledge_base_path: Optional[str] = None,
    context_manager: Optional[ContextManager] = None
) -> dict:
    """
    Factory function to create a fully configured
    Agentic RAG AI Assistant with LangGraph orchestration.

    Args:
        model_name: Nom du modèle Ollama.
        knowledge_base_path: Chemin vers la base de connaissances RAG.
        context_manager: Instance de ContextManager à utiliser. Si None,
            une nouvelle instance isolée est créée (comportement legacy,
            utile pour les tests unitaires). En production, le service
            appelant (AssistantService) doit passer le singleton partagé
            de services/ai_context_state.py, pour que les données de
            dataset/preprocessing enregistrées par UploadService et
            PreprocessingService soient visibles par l'agent.
    """
    logger.info("Creating DataKit AI Assistant with LangGraph...")

    embedding_model, vector_store, retriever = _build_rag_core()
    prompt_manager = PromptManager()
    response_formatter = ResponseFormatter()
    llm_client = OllamaClient(model_name=model_name)

    router, file_catalog, document_loader, metadata_store = _build_document_router(
        llm_client, knowledge_base_path
    )
    _build_or_load_index(retriever, vector_store, file_catalog, document_loader, metadata_store)

    if context_manager is None:
        # Fallback isolé (tests, usage standalone) — non partagé avec
        # les autres services de l'app.
        context_manager = ContextManager(
            DatasetContextBuilder(),
            PreprocessingContext()
        )

    status = llm_client.check_connection()
    if status.get("status"):
        logger.info(f"✅ Ollama ready with model: {model_name}")
    else:
        logger.warning(f"⚠️ {status.get('message')}")

    _load_index_fallback(retriever, vector_store)

    graph = create_graph(
        retriever=retriever,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        router=router,
        context_manager=context_manager,
        response_formatter=response_formatter
    )
    agent = LangGraphAgent(graph)

    logger.info("✅ LangGraph Agent created successfully")

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
        "response_formatter": response_formatter
    }