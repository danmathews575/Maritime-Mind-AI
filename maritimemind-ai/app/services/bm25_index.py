"""
BM25 Index Service — Phase 3.4

Builds and queries a BM25 sparse keyword index from maritime text chunks.
Uses rank-bm25 with a shared tokenizer for consistency between build and query.
Serializable to/from disk via pickle for persistence.

This index complements the dense vector search in ChromaDB:
- BM25 excels at exact terminology matching (ISO codes, model numbers).
- Dense vectors excel at semantic understanding.
- Phase 4 fuses both via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.configs.config import settings
from app.models.schemas import TextChunk
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.bm25_index")

# ---------------------------------------------------------------------------
# Shared tokenizer (MUST be identical at build time and query time)
# ---------------------------------------------------------------------------

# Common English stopwords — kept minimal for maritime domain
# (some "stopwords" like "fire" or "not" are critical in maritime context)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "its", "as", "if", "then", "so", "than", "very", "just",
})

# Regex: split on non-alphanumeric (but keep hyphens within words, e.g., "MAN-B&W")
_TOKENIZE_PATTERN = re.compile(r"[^\w\-]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text for BM25: lowercase, split on whitespace/punctuation,
    remove stopwords. Keeps hyphens within tokens for maritime codes.

    This function MUST be used for both index building and query-time search.
    Using different tokenizers will produce zero or garbage results.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase tokens with stopwords removed.
    """
    tokens = _TOKENIZE_PATTERN.split(text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class BM25IndexService:
    """
    BM25 sparse keyword index for maritime text chunks.

    Lifecycle
    ---------
    1. ``build_index(chunks)`` — tokenize all chunks, build BM25Okapi index
    2. ``save(path)``          — pickle-serialize the index + chunk IDs to disk
    3. ``load(path)``          — deserialize from disk
    4. ``search(query, top_k)`` — return ranked (chunk_id, score) pairs

    Design notes
    ------------
    - The tokenizer is shared at module level to guarantee consistency.
    - Chunk IDs are stored alongside the index so search returns chunk_id directly.
    - The index includes the raw tokenized corpus for inspection/debugging.
    """

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._default_path = settings.BM25_INDEX_PATH

    @property
    def is_built(self) -> bool:
        """Returns True if an index has been built or loaded."""
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        """Returns the number of documents in the index."""
        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[TextChunk]) -> None:
        """
        Tokenize all chunks and build the BM25Okapi index.

        Args:
            chunks: List of TextChunk objects from the ingestion pipeline.
        """
        if not chunks:
            logger.warning("No chunks provided — BM25 index not built")
            return

        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [tokenize(c.content) for c in chunks]

        # Filter out empty tokenizations (shouldn't happen, but safety)
        valid = [(cid, tokens) for cid, tokens in zip(self._chunk_ids, self._tokenized_corpus) if tokens]
        if not valid:
            logger.warning("All chunks produced empty tokenizations — index not built")
            return

        self._chunk_ids = [v[0] for v in valid]
        self._tokenized_corpus = [v[1] for v in valid]

        self._index = BM25Okapi(self._tokenized_corpus)

        avg_len = sum(len(t) for t in self._tokenized_corpus) / len(self._tokenized_corpus)
        logger.info(
            f"BM25 index built: {len(self._chunk_ids)} documents, "
            f"avg {avg_len:.1f} tokens/doc"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search the BM25 index with a text query.

        Args:
            query: Raw text query string.
            top_k: Number of top results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by descending score.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if not self.is_built:
            raise RuntimeError("BM25 index not built. Call build_index() or load() first.")

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning(f"Query produced no tokens after tokenization: '{query}'")
            return []

        scores = self._index.get_scores(query_tokens)

        # Pair chunk_ids with scores and sort descending
        scored = list(zip(self._chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k, filtering zero-score results
        results = [(cid, score) for cid, score in scored[:top_k] if score > 0.0]

        logger.debug(f"BM25 search '{query}': {len(results)} results (top score: {results[0][1]:.4f})" if results else f"BM25 search '{query}': 0 results")
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Serialize the BM25 index to disk via pickle.

        Args:
            path: Output file path. Defaults to settings.BM25_INDEX_PATH.

        Returns:
            The path where the index was saved.
        """
        if not self.is_built:
            raise RuntimeError("No index to save. Call build_index() first.")

        save_path = path or self._default_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        data = {
            "index": self._index,
            "chunk_ids": self._chunk_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved: {save_path} ({len(self._chunk_ids)} docs)")
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a previously saved BM25 index from disk.

        Args:
            path: Input file path. Defaults to settings.BM25_INDEX_PATH.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        load_path = path or self._default_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"BM25 index not found: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self._index = data["index"]
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["tokenized_corpus"]

        logger.info(f"BM25 index loaded: {load_path} ({len(self._chunk_ids)} docs)")
