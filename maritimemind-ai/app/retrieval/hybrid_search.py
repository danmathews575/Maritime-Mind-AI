from typing import Any, Dict, List, Tuple
from app.configs.config import settings
from app.models.schemas import RetrievalResult, RetrievalScores, TextChunk
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25IndexService
from app.services.embedding import TextEmbeddingService

class HybridSearchEngine:
    """
    Combines dense vector search (ChromaDB) and sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF) to produce a unified ranked list.
    """

    def __init__(self, vector_store: VectorStoreService, bm25_index: BM25IndexService, embedder: TextEmbeddingService):
        self.vs = vector_store
        self.bm25 = bm25_index
        self.embedder = embedder

    def search(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Executes hybrid search by querying both vector and BM25 systems,
        fusing the results with RRF, and returning fully populated RetrievalResult objects.
        """
        query_embedding = self.embedder.embed_query(query)

        # 1. Vector Search
        vector_results = self._vector_search(query_embedding, top_k, filters)
        
        # 2. BM25 Search (No metadata filtering built into rank-bm25 by default here, 
        # but we'll fuse first and we can post-filter if strict filters are needed, 
        # or we just rely on vector results for strict filters).
        # We will pass filters to vector search.
        # Since bm25 does not support filters out of the box in this implementation, 
        # we will fetch BM25 and later discard chunks that don't match the filter.
        bm25_results = self._bm25_search(query, top_k)

        # 3. RRF Fusion
        fused_scores = self._rrf_fusion(bm25_results, vector_results, k=settings.RRF_K)

        # Sort combined results descending
        sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_fused_ids = [chunk_id for chunk_id, _ in sorted_fused[:top_k]]

        # 4. Fetch full chunk data and build results
        return self._build_retrieval_results(top_fused_ids, fused_scores, bm25_results, vector_results, filters)

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25.is_built:
            return []
        return self.bm25.search(query, top_k=top_k)

    def _vector_search(self, query_embedding: List[float], top_k: int, filters: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        hits = self.vs.query_text(query_embedding, top_k=top_k, filters=filters)
        # ChromaDB distances are often cosine distances (smaller is better). 
        # We convert to a similarity score for the raw score logic, though RRF only uses rank.
        # RRF is rank-based, so just preserve the ordered list. The distance is helpful for raw scores.
        # hits are already sorted by distance by Chroma.
        return [(hit["id"], hit["distance"]) for hit in hits]

    def _rrf_fusion(self, bm25_results: List[Tuple[str, float]], vector_results: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
        """
        Calculates RRF score: 1 / (k + rank)
        """
        fused = {}

        # rank is 1-indexed
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (1.0 / (k + rank))

        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (1.0 / (k + rank))

        return fused

    def _build_retrieval_results(
        self, 
        fused_ids: List[str], 
        fused_scores: Dict[str, float], 
        bm25_results: List[Tuple[str, float]], 
        vector_results: List[Tuple[str, float]],
        filters: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        if not fused_ids:
            return []

        chunks = self.vs.get_text_chunks_by_ids(fused_ids)
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

        # Apply post-filtering for BM25 results that might not match the filters
        if filters:
            filtered_chunks = {}
            for cid, chunk in chunk_map.items():
                match = True
                for k, v in filters.items():
                    if getattr(chunk, k, None) != v:
                        match = False
                        break
                if match:
                    filtered_chunks[cid] = chunk
            chunk_map = filtered_chunks

        # Convert bm25 and vector results to dicts for easy lookup
        bm25_dict = {cid: score for cid, score in bm25_results}
        vector_dict = {cid: score for cid, score in vector_results}

        results = []
        for cid in fused_ids:
            if cid not in chunk_map:
                continue
            
            chunk = chunk_map[cid]
            # Convert distance to similarity if needed, though for now we store raw returned values
            # Vector score is distance (smaller = closer), BM25 score is raw BM25 (larger = better)
            # RRF handles ranking correctly regardless.
            scores = RetrievalScores(
                bm25_score=bm25_dict.get(cid, 0.0),
                vector_score=vector_dict.get(cid, 0.0),
                final_score=fused_scores.get(cid, 0.0)
            )
            
            results.append(RetrievalResult(chunk=chunk, scores=scores))

        return results
