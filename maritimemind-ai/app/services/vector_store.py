"""
ChromaDB Vector Store Service — Final Retrieval-Aware Architecture
Manages maritime text and image collections.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.configs.config import settings
from app.models.schemas import ImageMetadata, TextChunk, QueryIntent
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.vector_store")

_client: Optional[chromadb.ClientAPI] = None

def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = settings.CHROMADB_PERSIST_DIR
        logger.info(f"Initializing ChromaDB persistent client at: {persist_dir}")
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client

class VectorStoreService:
    def __init__(self) -> None:
        self._client = _get_client()
        self._text_collection_name = settings.TEXT_COLLECTION_NAME
        self._image_collection_name = settings.IMAGE_COLLECTION_NAME

    def _get_or_create_text_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._text_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _get_or_create_image_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._image_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_text_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings: return 0
        if len(chunks) != len(embeddings): raise ValueError("Mismatch length")

        collection = self._get_or_create_text_collection()
        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(self._build_text_metadata(chunk))

        batch_size = 500
        upserted = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end], embeddings=embeddings[i:end],
                documents=documents[i:end], metadatas=metadatas[i:end],
            )
            upserted += end - i

        logger.info(f"Upserted {upserted} text chunks")
        return upserted

    def query_text(
        self, embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        collection = self._get_or_create_text_collection()
        query_params = {
            "query_embeddings": [embedding], "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters: query_params["where"] = filters
        return self._flatten_results(collection.query(**query_params))

    def get_all_text_chunks(self) -> List[TextChunk]:
        collection = self._get_or_create_text_collection()
        results = collection.get(include=["documents", "metadatas"])
        if not results or not results.get("ids"): return []
        chunks = []
        ids = results["ids"]
        docs = results["documents"]
        metas = results["metadatas"]
        
        for i in range(len(ids)):
            chunks.append(self._reconstruct_text_chunk(ids[i], docs[i], metas[i]))
        return chunks

    def get_text_chunks_by_ids(self, ids: List[str]) -> List[TextChunk]:
        if not ids:
            return []
        collection = self._get_or_create_text_collection()
        results = collection.get(ids=ids, include=["documents", "metadatas"])
        if not results or not results.get("ids"):
            return []
        chunks = []
        r_ids = results["ids"]
        docs = results["documents"]
        metas = results["metadatas"]
        
        for i in range(len(r_ids)):
            chunks.append(self._reconstruct_text_chunk(r_ids[i], docs[i], metas[i]))
        return chunks

    def add_image_embeddings(self, images: List[ImageMetadata], embeddings: List[List[float]]) -> int:
        if not images or not embeddings: return 0
        if len(images) != len(embeddings): raise ValueError("Mismatch length")

        collection = self._get_or_create_image_collection()
        ids, documents, metadatas = [], [], []

        for img in images:
            ids.append(img.image_id)
            doc_text = f"{img.caption or ''} {img.ocr_text or ''}".strip() or "(image)"
            documents.append(doc_text)
            metadatas.append(self._build_image_metadata(img))

        batch_size = 500
        upserted = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end], embeddings=embeddings[i:end],
                documents=documents[i:end], metadatas=metadatas[i:end],
            )
            upserted += end - i

        logger.info(f"Upserted {upserted} images")
        return upserted

    def get_images_by_ids(self, ids: List[str]) -> List[ImageMetadata]:
        if not ids:
            return []
        collection = self._get_or_create_image_collection()
        results = collection.get(ids=ids, include=["metadatas"])
        if not results or not results.get("ids"):
            return []
            
        images = []
        r_ids = results["ids"]
        metas = results["metadatas"]
        
        for i in range(len(r_ids)):
            meta = metas[i]
            
            # Helper to safely parse bbox
            bbox = None
            if "bbox_x0" in meta and "bbox_y0" in meta and "bbox_x1" in meta and "bbox_y1" in meta:
                from app.models.schemas import BoundingBox
                bbox = BoundingBox(
                    x0=float(meta["bbox_x0"]),
                    y0=float(meta["bbox_y0"]),
                    x1=float(meta["bbox_x1"]),
                    y1=float(meta["bbox_y1"])
                )

            images.append(ImageMetadata(
                image_id=r_ids[i],
                manual_name=meta.get("manual_name", ""),
                ship_id=meta.get("ship_id") or None,
                language=meta.get("language") or None,
                page_number=meta.get("page_number", 0),
                image_path=meta.get("image_path", ""),
                section_title=meta.get("section_title", ""),
                caption=meta.get("caption", ""),
                tags=json.loads(meta.get("tags", "[]")),
                bbox=bbox,
                related_chunk_ids=json.loads(meta.get("related_chunk_ids", "[]")),
                ocr_text=meta.get("ocr_text", ""),
                ocr_quality=float(meta.get("ocr_quality", 1.0)),
                diagram_confidence=float(meta.get("diagram_confidence", 1.0)),
                embedding_model=meta.get("embedding_model", ""),
            ))
        return images

    def query_images(
        self, embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        collection = self._get_or_create_image_collection()
        query_params = {
            "query_embeddings": [embedding], "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters: query_params["where"] = filters
        return self._flatten_results(collection.query(**query_params))

    def get_collection_stats(self) -> Dict[str, Any]:
        text_col = self._get_or_create_text_collection()
        image_col = self._get_or_create_image_collection()
        return {
            "text_collection": {"name": self._text_collection_name, "count": text_col.count()},
            "image_collection": {"name": self._image_collection_name, "count": image_col.count()},
        }

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{name}': {e}")

    def reset_all(self) -> None:
        self.delete_collection(self._text_collection_name)
        self.delete_collection(self._image_collection_name)

    @staticmethod
    def _build_text_metadata(chunk: TextChunk) -> Dict[str, Any]:
        return {
            "manual_name": chunk.manual_name,
            "ship_id": chunk.ship_id or "",
            "language": chunk.language or "",
            "department": chunk.department or "",
            "subsystem": chunk.subsystem or "",
            "document_type": chunk.document_type or "",
            "page_number": chunk.page_number,
            "section_title": chunk.section_title or "",
            "contains_procedure": chunk.contains_procedure,
            "contains_warning": chunk.contains_warning,
            "contains_emergency_workflow": chunk.contains_emergency_workflow,
            "contains_diagram_reference": chunk.contains_diagram_reference,
            "importance": chunk.importance or "medium",
            "applicable_intents": json.dumps([q.value for q in chunk.applicable_intents]),
            "hierarchy_path": json.dumps(chunk.hierarchy_path),
            "related_image_ids": json.dumps(chunk.related_image_ids),
            "diagram_references": json.dumps(chunk.diagram_references),
            "keywords": json.dumps(chunk.keywords),
            "previous_chunk_id": chunk.previous_chunk_id or "",
            "next_chunk_id": chunk.next_chunk_id or "",
            "parent_chunk_id": chunk.parent_chunk_id or "",
            "embedding_model": chunk.embedding_model or "",
            "created_at": str(chunk.created_at),
        }

    @staticmethod
    def _reconstruct_text_chunk(chunk_id: str, document: str, meta: dict) -> TextChunk:
        """Reconstructs a TextChunk from ChromaDB stored metadata."""
        return TextChunk(
            chunk_id=chunk_id,
            manual_name=meta.get("manual_name", ""),
            content=document,
            page_number=meta.get("page_number", 0),
            ship_id=meta.get("ship_id") or None,
            language=meta.get("language") or None,
            department=meta.get("department") or "",
            subsystem=meta.get("subsystem") or "general",
            document_type=meta.get("document_type") or "manual",
            section_title=meta.get("section_title") or "",
            contains_procedure=meta.get("contains_procedure", False),
            contains_warning=meta.get("contains_warning", False),
            contains_emergency_workflow=meta.get("contains_emergency_workflow", False),
            contains_diagram_reference=meta.get("contains_diagram_reference", False),
            importance=meta.get("importance") or "medium",
            applicable_intents=[QueryIntent(q) for q in json.loads(meta.get("applicable_intents", "[]"))],
            hierarchy_path=json.loads(meta.get("hierarchy_path", "[]")),
            related_image_ids=json.loads(meta.get("related_image_ids", "[]")),
            diagram_references=json.loads(meta.get("diagram_references", "[]")),
            keywords=json.loads(meta.get("keywords", "[]")),
            previous_chunk_id=meta.get("previous_chunk_id") or None,
            next_chunk_id=meta.get("next_chunk_id") or None,
            parent_chunk_id=meta.get("parent_chunk_id") or None,
            embedding_model=meta.get("embedding_model") or "",
        )

    @staticmethod
    def _build_image_metadata(img: ImageMetadata) -> Dict[str, Any]:
        return {
            "manual_name": img.manual_name,
            "ship_id": img.ship_id or "",
            "language": img.language or "",
            "page_number": img.page_number,
            "image_path": img.image_path,
            "section_title": img.section_title or "",
            "caption": img.caption or "",
            "tags": json.dumps(img.tags),
            "related_chunk_ids": json.dumps(img.related_chunk_ids),
            "ocr_text": img.ocr_text or "",
            "ocr_quality": float(img.ocr_quality),
            "diagram_confidence": float(img.diagram_confidence),
            "embedding_model": img.embedding_model or "",
            "created_at": str(img.created_at),
        }

    @staticmethod
    def _flatten_results(results: Dict) -> List[Dict[str, Any]]:
        if not results or not results.get("ids") or not results["ids"][0]: return []
        flat = []
        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            flat.append({
                "id": ids[i],
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 0.0,
            })
        return flat
