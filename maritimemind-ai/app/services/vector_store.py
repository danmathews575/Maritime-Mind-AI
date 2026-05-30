"""
Qdrant Vector Store Service — V2 Architecture
Replaces ChromaDB. Manages maritime text and image collections.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.configs.config import settings
from app.models.schemas import ImageMetadata, TextChunk, QueryIntent
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.vector_store")

_client: Optional[QdrantClient] = None

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        qdrant_host = getattr(settings, "QDRANT_HOST", "localhost")
        qdrant_port = getattr(settings, "QDRANT_PORT", 6333)
        if qdrant_host.lower() == "local":
            qdrant_path = getattr(settings, "QDRANT_PATH", "./vector_store/qdrant_local")
            import os
            os.makedirs(qdrant_path, exist_ok=True)
            logger.info(f"Initializing Qdrant client in local mode at {qdrant_path}")
            _client = QdrantClient(path=qdrant_path)
        else:
            logger.info(f"Initializing Qdrant client at {qdrant_host}:{qdrant_port}")
            _client = QdrantClient(host=qdrant_host, port=qdrant_port)
    return _client

def _generate_uuid(id_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, id_str))

class VectorStoreService:
    def __init__(self) -> None:
        self._client = _get_client()
        self._text_collection_name = settings.TEXT_COLLECTION_NAME
        self._image_collection_name = settings.IMAGE_COLLECTION_NAME
        # We assume dim=384 for text (all-MiniLM-L6-v2) and dim=512 for CLIP, but it's safer to configure it.
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        try:
            text_dim = getattr(settings, "TEXT_EMBEDDING_DIM", 384)
            image_dim = getattr(settings, "IMAGE_EMBEDDING_DIM", 512)
            
            if not self._client.collection_exists(self._text_collection_name):
                self._client.create_collection(
                    collection_name=self._text_collection_name,
                    vectors_config=models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                )
                # Create payload indexes for metadata filtering
                for field in ["department", "manual_name", "importance"]:
                    self._client.create_payload_index(
                        collection_name=self._text_collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                self._client.create_payload_index(
                    collection_name=self._text_collection_name,
                    field_name="contains_procedure",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                
            if not self._client.collection_exists(self._image_collection_name):
                self._client.create_collection(
                    collection_name=self._image_collection_name,
                    vectors_config=models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
                )
                self._client.create_payload_index(
                    collection_name=self._image_collection_name,
                    field_name="manual_name",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self._client.create_payload_index(
                    collection_name=self._image_collection_name,
                    field_name="ocr_text",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=30,
                        lowercase=True,
                    ),
                )
                self._client.create_payload_index(
                    collection_name=self._image_collection_name,
                    field_name="caption",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=30,
                        lowercase=True,
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to verify/create collections: {e}")

    def add_text_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings: return 0
        if len(chunks) != len(embeddings): raise ValueError("Mismatch length")

        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = self._build_text_metadata(chunk)
            payload["_document"] = chunk.content
            payload["_original_id"] = chunk.chunk_id
            
            points.append(
                models.PointStruct(
                    id=_generate_uuid(chunk.chunk_id),
                    vector=emb,
                    payload=payload
                )
            )

        batch_size = 500
        upserted = 0
        for i in range(0, len(points), batch_size):
            end = min(i + batch_size, len(points))
            self._client.upsert(
                collection_name=self._text_collection_name,
                points=points[i:end]
            )
            upserted += end - i

        logger.info(f"Upserted {upserted} text chunks")
        return upserted

    def query_text(
        self, embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_qdrant_filter(filters)
        
        hits = self._client.search(
            collection_name=self._text_collection_name,
            query_vector=embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )
        return self._flatten_hits(hits)

    def get_all_text_chunks(self) -> List[TextChunk]:
        # Note: Qdrant scroll API is used for retrieving all
        chunks = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._text_collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for record in records:
                meta = dict(record.payload)
                doc = meta.pop("_document", "")
                orig_id = meta.pop("_original_id", str(record.id))
                chunks.append(self._reconstruct_text_chunk(orig_id, doc, meta))
            if offset is None:
                break
        return chunks

    def get_text_chunks_by_ids(self, ids: List[str]) -> List[TextChunk]:
        if not ids: return []
        uuid_to_orig = {_generate_uuid(i): i for i in ids}
        records = self._client.retrieve(
            collection_name=self._text_collection_name,
            ids=list(uuid_to_orig.keys()),
            with_payload=True
        )
        chunks = []
        for r in records:
            meta = dict(r.payload)
            doc = meta.pop("_document", "")
            orig_id = meta.pop("_original_id", uuid_to_orig.get(str(r.id), str(r.id)))
            chunks.append(self._reconstruct_text_chunk(orig_id, doc, meta))
        return chunks

    def add_image_embeddings(self, images: List[ImageMetadata], embeddings: List[List[float]]) -> int:
        if not images or not embeddings: return 0
        if len(images) != len(embeddings): raise ValueError("Mismatch length")

        points = []
        for img, emb in zip(images, embeddings):
            payload = self._build_image_metadata(img)
            doc_text = f"{img.caption or ''} {img.ocr_text or ''}".strip() or "(image)"
            payload["_document"] = doc_text
            payload["_original_id"] = img.image_id
            
            points.append(
                models.PointStruct(
                    id=_generate_uuid(img.image_id),
                    vector=emb,
                    payload=payload
                )
            )

        batch_size = 500
        upserted = 0
        for i in range(0, len(points), batch_size):
            end = min(i + batch_size, len(points))
            self._client.upsert(
                collection_name=self._image_collection_name,
                points=points[i:end]
            )
            upserted += end - i

        logger.info(f"Upserted {upserted} images")
        return upserted

    def get_images_by_ids(self, ids: List[str]) -> List[ImageMetadata]:
        if not ids: return []
        uuid_to_orig = {_generate_uuid(i): i for i in ids}
        records = self._client.retrieve(
            collection_name=self._image_collection_name,
            ids=list(uuid_to_orig.keys()),
            with_payload=True
        )
        images = []
        for r in records:
            meta = dict(r.payload)
            orig_id = meta.pop("_original_id", uuid_to_orig.get(str(r.id), str(r.id)))
            
            bbox = None
            if "bbox_x0" in meta and "bbox_y0" in meta and "bbox_x1" in meta and "bbox_y1" in meta:
                from app.models.schemas import BoundingBox
                bbox = BoundingBox(
                    x0=float(meta["bbox_x0"]), y0=float(meta["bbox_y0"]),
                    x1=float(meta["bbox_x1"]), y1=float(meta["bbox_y1"])
                )

            images.append(ImageMetadata(
                image_id=orig_id,
                manual_name=meta.get("manual_name", ""),
                ship_id=meta.get("ship_id") or None,
                language=meta.get("language") or None,
                page_number=meta.get("page_number", 0),
                image_path=meta.get("image_path", ""),
                section_title=meta.get("section_title", ""),
                caption=meta.get("caption", ""),
                tags=json.loads(meta.get("tags", "[]")) if isinstance(meta.get("tags"), str) else meta.get("tags", []),
                bbox=bbox,
                related_chunk_ids=json.loads(meta.get("related_chunk_ids", "[]")) if isinstance(meta.get("related_chunk_ids"), str) else meta.get("related_chunk_ids", []),
                ocr_text=meta.get("ocr_text", ""),
                ocr_quality=float(meta.get("ocr_quality", 1.0)),
                diagram_confidence=float(meta.get("diagram_confidence", 1.0)),
                diagram_type=meta.get("diagram_type", "UNKNOWN"),
                embedding_model=meta.get("embedding_model", ""),
            ))
        return images

    def query_images(
        self, embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_qdrant_filter(filters)
        hits = self._client.search(
            collection_name=self._image_collection_name,
            query_vector=embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )
        return self._flatten_hits(hits)

    def search_images_by_keyword(self, keyword: str, limit: int = 10) -> List[ImageMetadata]:
        """Path 3: Keyword payload search on ocr_text and caption."""
        try:
            # We search in ocr_text or caption. 
            # In Qdrant we can use a Should condition.
            filter_obj = models.Filter(
                should=[
                    models.FieldCondition(
                        key="ocr_text",
                        match=models.MatchText(text=keyword)
                    ),
                    models.FieldCondition(
                        key="caption",
                        match=models.MatchText(text=keyword)
                    )
                ]
            )
            records, _ = self._client.scroll(
                collection_name=self._image_collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            images = []
            for r in records:
                meta = dict(r.payload)
                orig_id = meta.pop("_original_id", str(r.id))
                bbox = None
                if "bbox_x0" in meta and "bbox_y0" in meta and "bbox_x1" in meta and "bbox_y1" in meta:
                    from app.models.schemas import BoundingBox
                    bbox = BoundingBox(
                        x0=float(meta["bbox_x0"]), y0=float(meta["bbox_y0"]),
                        x1=float(meta["bbox_x1"]), y1=float(meta["bbox_y1"])
                    )

                images.append(ImageMetadata(
                    image_id=orig_id,
                    manual_name=meta.get("manual_name", ""),
                    ship_id=meta.get("ship_id") or None,
                    language=meta.get("language") or None,
                    page_number=meta.get("page_number", 0),
                    image_path=meta.get("image_path", ""),
                    section_title=meta.get("section_title", ""),
                    caption=meta.get("caption", ""),
                    tags=json.loads(meta.get("tags", "[]")) if isinstance(meta.get("tags"), str) else meta.get("tags", []),
                    bbox=bbox,
                    related_chunk_ids=json.loads(meta.get("related_chunk_ids", "[]")) if isinstance(meta.get("related_chunk_ids"), str) else meta.get("related_chunk_ids", []),
                    ocr_text=meta.get("ocr_text", ""),
                    ocr_quality=float(meta.get("ocr_quality", 1.0)),
                    diagram_confidence=float(meta.get("diagram_confidence", 1.0)),
                    diagram_type=meta.get("diagram_type", "UNKNOWN"),
                    embedding_model=meta.get("embedding_model", ""),
                ))
            return images
        except Exception as e:
            logger.warning(f"Keyword image search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            txt = self._client.get_collection(self._text_collection_name)
            img = self._client.get_collection(self._image_collection_name)
            return {
                "text_collection": {"name": self._text_collection_name, "count": txt.points_count},
                "image_collection": {"name": self._image_collection_name, "count": img.points_count},
            }
        except Exception:
            return {"text_collection": {"count": 0}, "image_collection": {"count": 0}}

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{name}': {e}")

    def reset_all(self) -> None:
        self.delete_collection(self._text_collection_name)
        self.delete_collection(self._image_collection_name)

    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filters: return None
        conditions = []
        for k, v in filters.items():
            conditions.append(models.FieldCondition(
                key=k, match=models.MatchValue(value=v)
            ))
        return models.Filter(must=conditions)

    @staticmethod
    def _flatten_hits(hits: List[models.ScoredPoint]) -> List[Dict[str, Any]]:
        flat = []
        for hit in hits:
            meta = dict(hit.payload or {})
            doc = meta.pop("_document", "")
            orig_id = meta.pop("_original_id", str(hit.id))
            # Qdrant returns cosine similarity (higher is better). 
            # Chroma returns distance (smaller is better).
            # Convert similarity to distance for backwards compatibility.
            distance = 1.0 - hit.score
            flat.append({
                "id": orig_id,
                "document": doc,
                "metadata": meta,
                "distance": distance,
            })
        return flat

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
            "applicable_intents": [q.value for q in chunk.applicable_intents],
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
        """Reconstructs a TextChunk from Qdrant stored metadata."""
        def safe_json_load(val):
            if isinstance(val, str):
                try: return json.loads(val)
                except: return []
            return val or []
            
        app_intents = safe_json_load(meta.get("applicable_intents", []))
        if isinstance(app_intents, str):
            app_intents = [app_intents]
            
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
            applicable_intents=[QueryIntent(q) for q in app_intents if q],
            hierarchy_path=safe_json_load(meta.get("hierarchy_path", [])),
            related_image_ids=safe_json_load(meta.get("related_image_ids", [])),
            diagram_references=safe_json_load(meta.get("diagram_references", [])),
            keywords=safe_json_load(meta.get("keywords", [])),
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
            "tags": img.tags,
            "related_chunk_ids": img.related_chunk_ids,
            "ocr_text": img.ocr_text or "",
            "ocr_quality": float(img.ocr_quality),
            "diagram_confidence": float(img.diagram_confidence),
            "diagram_type": img.diagram_type,
            "embedding_model": img.embedding_model or "",
            "created_at": str(img.created_at),
        }
