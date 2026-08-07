"""
AI Assistant module for DataKit.
"""

import logging
import os
from typing import Optional

# Configuration du logging - éviter les doublons
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

# Context
from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import PreprocessingContext

# LLM
from .llm.ollama_client import OllamaClient
from .llm.prompt_manager import PromptManager
from .llm.response_formatter import ResponseFormatter

# RAG
from .rag.embeddings import EmbeddingModel
from .rag.retriever import Retriever
from .rag.vector_store import VectorStore

# RAG components
from .rag.file_catalog import FileCatalog
from .rag.router import DocumentRouter
from .rag.document_loader import DocumentLoader
from .rag.metadata_store import MetadataStore

# LangGraph Agent
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


def create_assistant(
    model_name: str = "mistral",
    knowledge_base_path: Optional[str] = None
) -> dict:
    """
    Factory function to create a fully configured
    Agentic RAG AI Assistant with LangGraph orchestration.
    """

    logger.info(
        "Creating DataKit AI Assistant with LangGraph..."
    )

    # ============================
    # Core RAG components
    # ============================

    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    retriever = Retriever(
        embedding_model,
        vector_store
    )
    prompt_manager = PromptManager()
    response_formatter = ResponseFormatter()
    llm_client = OllamaClient(
        model_name=model_name
    )

    # ============================
    # Document Router & Loader
    # ============================

    router = None
    file_catalog = None
    document_loader = None
    metadata_store = None

    if knowledge_base_path:
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
                file_catalog
            )

            logger.info(
                "✅ Document Router initialized"
            )

            # ============================
            # Initialize FAISS with documents (only if index doesn't exist)
            # ============================
            
            # Check if FAISS index already exists using property
            if not vector_store.exists:
                logger.info(
                    "🔄 Building FAISS index from documents..."
                )
                
                # Get documents safely
                documents = file_catalog.get_documents()
                
                if documents:
                    # Handle both dict and string formats
                    if isinstance(documents[0], dict):
                        all_files = [
                            d.get("file", "")
                            for d in documents
                            if d.get("file")
                        ]
                    else:
                        all_files = documents
                    
                    if all_files:
                        all_documents = document_loader.load(all_files)
                        
                        if all_documents:
                            # Enrich with metadata (preserves all fields)
                            all_documents = metadata_store.enrich(all_documents)
                            
                            # Initialize retriever
                            retriever.initialize(all_documents)
                            logger.info(
                                f"✅ FAISS built with {len(all_documents)} documents"
                            )
            else:
                logger.info(
                    "📂 FAISS index already exists, loading..."
                )
                if vector_store.load():
                    retriever._is_initialized = True
                    logger.info(
                        f"✅ FAISS loaded with {len(vector_store.metadata)} documents"
                    )

        except Exception as e:
            logger.warning(
                f"⚠️ Router initialization failed: {e}"
            )

    else:
        logger.warning(
            "⚠️ No knowledge base path provided. "
            "Running without document routing."
        )

    # ============================
    # Context manager
    # ============================

    dataset_builder = DatasetContextBuilder()
    preprocessing_context = PreprocessingContext()
    context_manager = ContextManager(
        dataset_builder,
        preprocessing_context
    )

    # ============================
    # Check Ollama
    # ============================

    status = llm_client.check_connection()

    if status.get("status"):
        logger.info(
            f"✅ Ollama ready with model: {model_name}"
        )
    else:
        logger.warning(
            f"⚠️ {status.get('message')}"
        )

    # ============================
    # Load existing FAISS index (fallback)
    # ============================

    if not retriever._is_initialized:
        try:
            if vector_store.load():
                retriever._is_initialized = True
                logger.info(
                    "✅ Vector store loaded from disk"
                )

        except Exception as e:
            logger.warning(
                f"⚠️ Could not load vector store: {e}"
            )

    # ============================
    # Create LangGraph Agent
    # ============================

    graph = create_graph(
        retriever=retriever,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        router=router,
        context_manager=context_manager,
        response_formatter=response_formatter
    )

    agent = LangGraphAgent(graph)

    logger.info(
        "✅ LangGraph Agent created successfully"
    )

    # ============================
    # Return everything
    # ============================

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