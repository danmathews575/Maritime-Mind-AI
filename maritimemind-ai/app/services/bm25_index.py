"""
BM25 Index Service — Hardened Retrieval Architecture

Builds and queries a BM25 sparse keyword index from maritime text chunks.
Hardened with:
- Porter Stemming (improves recall for inflected terms: "inspecting" -> "inspect")
- Maritime Code Preservation (prevents stemming of "MAN-B&W" or "ISO-8217")
- Maritime Synonym Expansion (e.g., "LO" -> "lube oil")
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer

from app.configs.config import settings
from app.models.schemas import TextChunk
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.bm25_index")

# Initialize stemmer globally
_stemmer = PorterStemmer()

# Common English stopwords — kept minimal for maritime domain
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "its", "as", "if", "then", "so", "than", "very", "just",
})

# Maritime domain synonyms for expansion
_MARITIME_SYNONYMS = {
    "lo": ["lube", "oil", "lubricating", "oil"],
    "fo": ["fuel", "oil"],
    "hfo": ["heavy", "fuel", "oil"],
    "mdo": ["marine", "diesel", "oil"],
    "cw": ["cooling", "water"],
    "fw": ["fresh", "water"],
    "sw": ["sea", "water"],
    "me": ["main", "engine"],
    "ae": ["auxiliary", "engine"],
    "dg": ["diesel", "generator"],
    "rpm": ["revolutions", "per", "minute"],
}

# Regex: split on non-alphanumeric (but keep hyphens within words, e.g., "MAN-B&W")
_TOKENIZE_PATTERN = re.compile(r"[^\w\-]+")

# Pattern to detect maritime codes that should NOT be stemmed
_CODE_PATTERN = re.compile(r"^[A-Z0-9\-]+$")


def tokenize(text: str, expand: bool = False) -> List[str]:
    """
    Tokenizes text for BM25: lowercase, split on whitespace/punctuation,
    remove stopwords, and apply Porter Stemming.
    
    Hardened features:
    - Preserves maritime codes (all-caps or hyphenated alphanumeric) from stemming
    - Optionally expands common maritime acronyms
    """
    raw_tokens = _TOKENIZE_PATTERN.split(text)
    processed = []
    
    for t in raw_tokens:
        if not t or len(t) <= 1:
            continue
            
        t_lower = t.lower()
        if t_lower in _STOPWORDS:
            continue
            
        # Check if it's a code (before lowercasing the original token)
        is_code = bool(_CODE_PATTERN.match(t))
        
        # Expand synonyms if requested (usually only query time)
        if expand and t_lower in _MARITIME_SYNONYMS:
            processed.append(t_lower) # Keep the acronym
            processed.extend(_MARITIME_SYNONYMS[t_lower]) # Add expansion
            continue
            
        if is_code:
            # Don't stem codes, just lowercase
            processed.append(t_lower)
        else:
            # Apply stemming
            processed.append(_stemmer.stem(t_lower))
            
    return processed


class BM25IndexService:
    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._default_path = settings.BM25_INDEX_PATH

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    def build_index(self, chunks: List[TextChunk]) -> None:
        if not chunks:
            logger.warning("No chunks provided — BM25 index not built")
            return

        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        self._chunk_ids = [c.chunk_id for c in chunks]
        # Do not expand synonyms during corpus build to save space
        self._tokenized_corpus = [tokenize(c.content, expand=False) for c in chunks]

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
            f"avg {avg_len:.1f} stemmed tokens/doc"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if not self.is_built:
            raise RuntimeError("BM25 index not built. Call build_index() or load() first.")

        # Expand synonyms on the query
        query_tokens = tokenize(query, expand=True)
        if not query_tokens:
            logger.warning(f"Query produced no tokens after tokenization: '{query}'")
            return []

        scores = self._index.get_scores(query_tokens)

        scored = list(zip(self._chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [(cid, score) for cid, score in scored[:top_k] if score > 0.0]

        return results

    def save(self, path: Optional[str] = None) -> str:
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

        logger.info(f"BM25 index saved: {save_path}")
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        load_path = path or self._default_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"BM25 index not found: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self._index = data["index"]
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["tokenized_corpus"]

        logger.info(f"BM25 index loaded: {load_path} ({len(self._chunk_ids)} docs)")
