"""
Hybrid retrieval using semantic search + BM25 lexical search
with Reciprocal Rank Fusion (RRF).
"""

from typing import Optional

import numpy as np

from .lexical.bm25_client import BM25Client
from .retriever import QdrantRetriever


class HybridRetriever:
    """
    Hybrid retriever.

    Combines:

        Semantic search
            +
        BM25 lexical search
            v
        Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        semantic_retriever: QdrantRetriever,
        bm25_client: BM25Client,
        semantic_top_k: int = 10,
        lexical_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:

        self.semantic_retriever = semantic_retriever
        self.bm25_client = bm25_client

        self.semantic_top_k = semantic_top_k
        self.lexical_top_k = lexical_top_k
        self.final_top_k = final_top_k

        # Standard RRF constant.
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Perform hybrid retrieval.

        1. Semantic search
        2. BM25 search
        3. RRF fusion
        4. Return top results

        `embedding` can be passed in when the caller already computed
        the query embedding (e.g. for classification or cache lookup),
        to avoid encoding the same query twice.
        """

        semantic_results = self.semantic_retriever.retrieve(
            query=query,
            embedding=embedding,
            top_k=self.semantic_top_k,
        )

        lexical_results = self._bm25_search(query)

        fused_results = self._rrf_fusion(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
        )

        return fused_results[: self.final_top_k]

    def _bm25_search(self, query: str) -> list[dict]:
        """
        Search Elasticsearch using BM25.

        Elasticsearch's `text` field uses its inverted index and
        BM25 scoring.
        """

        if not self.bm25_client.client.indices.exists(
            index=self.bm25_client.index_name
        ):
            return []

        response = self.bm25_client.client.search(
            index=self.bm25_client.index_name,
            query={"match": {"content": {"query": query}}},
            size=self.lexical_top_k,
        )

        results = []

        hits = response.get("hits", {}).get("hits", [])

        for rank, hit in enumerate(hits, start=1):

            source = hit.get("_source", {})

            results.append(
                {
                    "chunk_id": source.get("chunk_id", ""),
                    "document_id": source.get("document_id", ""),
                    "content": source.get("content", ""),
                    "document_name": source.get("document_name", ""),
                    "source": source.get("source", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "rank": rank,
                    "retrieval_method": "lexical",
                }
            )

        return results

    def _rrf_fusion(
        self,
        semantic_results: list[dict],
        lexical_results: list[dict],
    ) -> list[dict]:
        """
        Combine semantic and lexical rankings using RRF.

        RRF formula:

            RRF(d) = 1 / (k + rank_semantic) + 1 / (k + rank_lexical)

        A document appearing in both rankings receives contributions
        from both lists. A document appearing in only one list still
        receives a score, from that list alone.
        """

        fused: dict[str, dict] = {}

        def _register(result: dict, method: str) -> None:

            chunk_id = result["chunk_id"]

            if not chunk_id:
                return

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result["document_id"],
                    "content": result["content"],
                    "document_name": result["document_name"],
                    "source": result["source"],
                    "rrf_score": 0.0,
                    "semantic_score": None,
                    "lexical_score": None,
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (
                self.rrf_k + result["rank"]
            )

            if method == "semantic":
                fused[chunk_id]["semantic_score"] = result["score"]
            else:
                fused[chunk_id]["lexical_score"] = result["score"]

        for result in semantic_results:
            _register(result, "semantic")

        for result in lexical_results:
            _register(result, "lexical")

        ranked_results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        for rank, result in enumerate(ranked_results, start=1):
            result["rank"] = rank
            result["retrieval_method"] = "hybrid"

        return ranked_results