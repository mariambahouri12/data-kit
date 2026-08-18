"""
AI Assistant factory.
"""

from dataclasses import dataclass
from typing import Optional

from .cache.redis_client import RedisClient
from .cache.redis_index import RedisVectorIndex
from .cache.semantic_cache import SemanticCache
from .classifier.classifier_ml import QueryClassifier
from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import PreprocessingContext
from .embeddings.embeddings import EmbeddingModel
from .llm.ollama_client import OllamaClient
from .llm.prompt_builder import PromptBuilder
from .orchestrator import AssistantOrchestrator
from .rag.chunker import TextChunker
from .rag.document_loader import DocumentLoader
from .rag.document_processor import DocumentProcessor
from .rag.hybrid_retriever import HybridRetriever
from .rag.lexical.bm25_client import BM25Client
from .rag.lexical.bm25_indexer import BM25Indexer
from .rag.qdrant_client import QdrantVectorStore
from .rag.qdrant_indexer import QdrantIndexer
from .rag.retriever import QdrantRetriever


@dataclass
class AssistantRuntime:
    """Orchestrator + indexer + context_manager, ready to use."""

    orchestrator: AssistantOrchestrator
    indexer: QdrantIndexer
    context_manager: ContextManager


def _default_context_manager() -> ContextManager:
    """Fallback when no context_manager is injected (tests, scripts)."""

    return ContextManager(
        dataset_builder=DatasetContextBuilder(),
        preprocessing_context=PreprocessingContext(),
    )


def _build_orchestrator(
    embedding_model: EmbeddingModel,
    classifier_model_path: str,
    redis_url: str,
    vector_store: QdrantVectorStore,
    bm25_client: BM25Client,
    context_manager: ContextManager,
    ollama_model: str,
    semantic_top_k: int,
    lexical_top_k: int,
    final_top_k: int,
    rrf_k: int,
) -> AssistantOrchestrator:

    classifier = QueryClassifier(model_path=classifier_model_path)

    redis = RedisClient(url=redis_url)
    RedisVectorIndex(
        client=redis.client,
        dimension=embedding_model.dimension,
    ).create_indexes()

    semantic_cache = SemanticCache(
        client=redis.client,
        dimension=embedding_model.dimension,
    )

    semantic_retriever = QdrantRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=semantic_top_k,
    )

    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_client=bm25_client,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
    )

    return AssistantOrchestrator(
        embedding_model=embedding_model,
        classifier=classifier,
        semantic_cache=semantic_cache,
        retriever=hybrid_retriever,
        context_manager=context_manager,
        prompt_builder=PromptBuilder(),
        llm=OllamaClient(model=ollama_model),
    )


def _build_indexer(
    embedding_model: EmbeddingModel,
    vector_store: QdrantVectorStore,
    bm25_client: BM25Client,
    knowledge_base_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> QdrantIndexer:

    return QdrantIndexer(
        loader=DocumentLoader(knowledge_base_path),
        processor=DocumentProcessor(),
        chunker=TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_indexer=BM25Indexer(client=bm25_client),
    )


def create_assistant(
    context_manager: Optional[ContextManager] = None,
    classifier_model_path: str = (
        "datakit/ai_assistant/classifier/model/query_classifier.joblib"
    ),
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    redis_url: str = "redis://localhost:6379",
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "datakit_knowledge",
    elasticsearch_url: str = "http://localhost:9200",
    elasticsearch_index: str = "datakit_knowledge_bm25",
    knowledge_base_path: str = "knowledge_base",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    ollama_model: str = "mistral",
    semantic_top_k: int = 10,
    lexical_top_k: int = 10,
    final_top_k: int = 5,
    rrf_k: int = 60,
) -> AssistantRuntime:
    """
    Create a fully configured DataKit assistant runtime, wired for
    hybrid retrieval (Qdrant semantic search + Elasticsearch BM25
    lexical search, fused with Reciprocal Rank Fusion).
    """

    embedding_model = EmbeddingModel(model_name=embedding_model_name)

    vector_store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=qdrant_collection,
    )

    bm25_client = BM25Client(
        url=elasticsearch_url,
        index_name=elasticsearch_index,
    )

    context_manager = context_manager or _default_context_manager()

    orchestrator = _build_orchestrator(
        embedding_model=embedding_model,
        classifier_model_path=classifier_model_path,
        redis_url=redis_url,
        vector_store=vector_store,
        bm25_client=bm25_client,
        context_manager=context_manager,
        ollama_model=ollama_model,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
    )

    indexer = _build_indexer(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_client=bm25_client,
        knowledge_base_path=knowledge_base_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return AssistantRuntime(
        orchestrator=orchestrator,
        indexer=indexer,
        context_manager=context_manager,
    )