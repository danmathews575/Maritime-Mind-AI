"""
Image Embedding Service (OpenCLIP) — Phase 3.2

Embeds images AND text into a shared 512-dim CLIP vector space, enabling
cross-modal retrieval: text query → relevant image.

Uses OpenCLIP with the ViT-B-32 / laion2b_s34b_b79k checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.clip_embedding")

# ---------------------------------------------------------------------------
# Module-level singletons (lazy loaded)
# ---------------------------------------------------------------------------

_clip_model = None
_preprocess = None
_tokenizer = None


def _get_clip():
    """Lazy-load the OpenCLIP model, preprocess function, and tokenizer once."""
    global _clip_model, _preprocess, _tokenizer
    if _clip_model is None:
        logger.info(
            f"Loading CLIP model: {settings.CLIP_MODEL_NAME} "
            f"(pretrained: {settings.CLIP_PRETRAINED})"
        )
        import open_clip

        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name=settings.CLIP_MODEL_NAME,
            pretrained=settings.CLIP_PRETRAINED,
            device=settings.DEVICE,
        )
        _tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)
        _clip_model.eval()

        logger.info(f"CLIP model loaded on {settings.DEVICE}")
    return _clip_model, _preprocess, _tokenizer


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class ImageEmbeddingService:
    """
    Generates dense embeddings for images and text in the shared CLIP space.

    Two embedding modes
    -------------------
    1. **Image embedding** — for indexing extracted engineering diagrams.
       Uses the CLIP vision encoder via ``embed_image()``.

    2. **Text-for-image-search embedding** — for query-time cross-modal search.
       Uses the CLIP text encoder via ``embed_text_for_image_search()``.
       This produces a 512-dim vector in the **same space** as image embeddings,
       enabling direct cosine similarity comparison.

    Design notes
    ------------
    - Model is lazy-loaded via module-level singleton (~15-30s on CPU first call).
    - All tensors are processed in ``torch.no_grad()`` for inference.
    - Batch processing with tqdm for large image sets.
    """

    def __init__(self) -> None:
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._device = settings.DEVICE

    @property
    def model(self):
        model, _, _ = _get_clip()
        return model

    @property
    def preprocess(self):
        _, preprocess, _ = _get_clip()
        return preprocess

    @property
    def tokenizer(self):
        _, _, tokenizer = _get_clip()
        return tokenizer

    @property
    def dimension(self) -> int:
        """CLIP ViT-B-32 produces 512-dim vectors."""
        return 512

    # ------------------------------------------------------------------
    # Public API: Image embedding
    # ------------------------------------------------------------------

    def embed_image(self, image_path: str) -> List[float]:
        """
        Embed a single image file into the CLIP visual space.

        Args:
            image_path: Absolute path to a PNG/JPEG image file.

        Returns:
            Normalized 512-dim embedding vector as List[float].
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return [0.0] * self.dimension

        preprocessed = self.preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self.model.encode_image(preprocessed)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()

    def embed_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed a batch of image files with optional progress bar.

        Failed images produce zero-vectors (logged as warnings).

        Args:
            image_paths:   List of absolute paths to image files.
            show_progress: Whether to display tqdm progress bar.

        Returns:
            List of 512-dim embedding vectors (each is List[float]).
        """
        if not image_paths:
            return []

        logger.info(f"Embedding {len(image_paths)} images (batch_size={self._batch_size})")
        results: List[List[float]] = []
        batch: List[torch.Tensor] = []
        batch_indices: List[int] = []

        # Pre-allocate zero vectors for all indices
        results = [[0.0] * self.dimension] * len(image_paths)

        iterator = tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc="Image embeddings",
            disable=not show_progress,
        )

        for idx, path in iterator:
            try:
                img = Image.open(path).convert("RGB")
                preprocessed = self.preprocess(img)
                batch.append(preprocessed)
                batch_indices.append(idx)
            except Exception as e:
                logger.warning(f"Skipping image {path}: {e}")
                continue

            # Process batch when full
            if len(batch) >= self._batch_size:
                embeddings = self._encode_image_batch(batch)
                for i, emb in zip(batch_indices, embeddings):
                    results[i] = emb
                batch.clear()
                batch_indices.clear()

        # Final partial batch
        if batch:
            embeddings = self._encode_image_batch(batch)
            for i, emb in zip(batch_indices, embeddings):
                results[i] = emb

        valid = sum(1 for r in results if any(v != 0.0 for v in r))
        logger.info(f"Image embedding complete: {valid}/{len(image_paths)} successful")
        return results

    # ------------------------------------------------------------------
    # Public API: Cross-modal text embedding
    # ------------------------------------------------------------------

    def embed_text_for_image_search(self, query: str) -> List[float]:
        """
        Encode a text query into the CLIP visual space for cross-modal retrieval.

        This uses CLIP's **text encoder** (not SentenceTransformers) to produce a
        512-dim vector in the **same space** as image embeddings. This enables:
            text query → cosine similarity → relevant image

        Args:
            query: Text search query (e.g., "cooling pump wiring diagram").

        Returns:
            Normalized 512-dim embedding vector as List[float].
        """
        tokens = self.tokenizer([query]).to(self._device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_image_batch(self, tensors: List[torch.Tensor]) -> List[List[float]]:
        """Encodes a batch of preprocessed image tensors into normalized embeddings."""
        stacked = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            features = self.model.encode_image(stacked)
            features = features / features.norm(dim=-1, keepdim=True)

        return [features[i].cpu().tolist() for i in range(features.shape[0])]
