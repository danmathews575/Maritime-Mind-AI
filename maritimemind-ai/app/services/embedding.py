"""
Text Embedding Service — Phase 3.1

Wraps SentenceTransformers for maritime text embedding.
Lazy-loads the model on first use (singleton pattern) to avoid slow startup.

Produces 384-dim vectors by default (all-MiniLM-L6-v2).
"""
from __future__ import annotations

from typing import List, Optional
import functools

from tqdm import tqdm

from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.embedding")

# ---------------------------------------------------------------------------
# Module-level singleton for model (lazy loaded)
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model once."""
    global _model
    if _model is None:
        logger.info(f"Loading text embedding model: {settings.TEXT_EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            settings.TEXT_EMBEDDING_MODEL,
            device=settings.DEVICE,
        )
        dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Text embedding model loaded — dimension: {dim}")
    return _model


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class TextEmbeddingService:
    """
    Generates dense text embeddings using SentenceTransformers.

    Design notes
    ------------
    - Model is lazy-loaded via module-level singleton (first call ~5-10s on CPU).
    - All subsequent calls reuse the loaded model with zero overhead.
    - Batch embedding uses tqdm for progress tracking on large corpora.
    - Device routing (cpu/cuda) comes from settings.DEVICE.

    Usage::

        svc = TextEmbeddingService()
        vec = svc.embed_text("cooling pump maintenance procedure")
        # vec is List[float] of length 384
    """

    def __init__(self) -> None:
        self._model_name = settings.TEXT_EMBEDDING_MODEL
        self._batch_size = settings.EMBEDDING_BATCH_SIZE

    @property
    def model(self):
        """Returns the lazy-loaded SentenceTransformer model."""
        return _get_model()

    @property
    def dimension(self) -> int:
        """Returns the embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=1024)
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Input text string.

        Returns:
            List of floats (embedding vector).
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed a batch of text strings with optional progress bar.

        Args:
            texts:         List of input text strings.
            show_progress: Whether to display tqdm progress bar.

        Returns:
            List of embedding vectors (each is List[float]).
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts (batch_size={self._batch_size})")

        embeddings = self.model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        result = [emb.tolist() for emb in embeddings]
        logger.info(f"Text embedding complete: {len(result)} vectors of dim {len(result[0])}")
        return result

    @functools.lru_cache(maxsize=1024)
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query. Alias for embed_text, semantically distinct
        for clarity at query time vs. indexing time.

        The multilingual model (paraphrase-multilingual-MiniLM-L12-v2)
        does not require query prefixes — symmetric embedding for
        both queries and documents.

        Args:
            query: User search query string.

        Returns:
            Embedding vector as List[float].
        """
        return self.embed_text(query)
