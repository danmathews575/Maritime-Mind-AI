"""
Staged Ingestion Validation — Cross-Domain Maritime PDF Corpus
Processes 5 representative PDFs and runs retrieval validation queries after each.

Validates:
  - engineering troubleshooting retrieval
  - emergency workflow retrieval  
  - navigation retrieval
  - large-document stability
  - multimodal diagram grounding
  - metadata routing accuracy
  - contextual chunk expansion
  - hybrid BM25 + vector search
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath("."))

from app.ingestion.pipeline import IngestionPipeline
from app.services.vector_store import VectorStoreService
from app.services.embedding import TextEmbeddingService
from app.services.clip_embedding import ImageEmbeddingService
from app.services.bm25_index import BM25IndexService
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.staged_validation")

# ── Staged ingestion manifest ───────────────────────────────────────────────
STAGED_PDFS = [
    {
        "path": "data/raw_pdfs/engineering/ship fuel system.pdf",
        "label": "Ship Fuel System (Engineering)",
        "domain": "engineering",
        "queries": [
            "How do I troubleshoot a fuel oil pressure drop?",
            "What are the fuel purifier maintenance steps?",
            "fuel system valve operation procedure",
        ],
        "expected_intents": ["TROUBLESHOOTING", "PROCEDURE"],
    },
    {
        "path": "data/raw_pdfs/safety/EngineRoomFires_TSC.pdf",
        "label": "Engine Room Fires (Safety / Emergency)",
        "domain": "safety",
        "queries": [
            "What is the emergency procedure for engine room fire?",
            "How to activate fixed fire suppression system?",
            "fire detection alarm response steps",
        ],
        "expected_intents": ["EMERGENCY", "PROCEDURE"],
    },
    {
        "path": "data/raw_pdfs/deck/Loss-Prevention-Article-Ballast-Operation-and-Maintenance-Practice-07-2023.pdf",
        "label": "Ballast Operation Manual (Deck)",
        "domain": "deck",
        "queries": [
            "What is the correct ballast water exchange procedure?",
            "ballast pump startup sequence",
            "How to prevent ballast tank corrosion?",
        ],
        "expected_intents": ["PROCEDURE", "TROUBLESHOOTING"],
    },
    {
        "path": "data/raw_pdfs/navigation/radar manual.pdf",
        "label": "Radar Manual (Navigation — Large Document)",
        "domain": "navigation",
        "queries": [
            "How to calibrate radar heading alignment?",
            "ARPA target acquisition procedure",
            "radar interference rejection settings",
        ],
        "expected_intents": ["PROCEDURE", "DIAGRAM_REQUEST"],
    },
    {
        "path": "data/raw_pdfs/engineering/736871061-Wartsila-26-Maintenance-Manual.pdf",
        "label": "Wärtsilä Maintenance Manual (Engineering)",
        "domain": "engineering",
        "queries": [
            "What is the cylinder head overhaul procedure?",
            "How to adjust fuel injection timing on Wartsila engine?",
            "Wartsila 26 cooling water pump maintenance",
        ],
        "expected_intents": ["PROCEDURE", "TROUBLESHOOTING"],
    },
]

# ── Retrieval Validation Queries ─────────────────────────────────────────────

def run_retrieval_validation(
    queries: List[str],
    domain: str,
    embedder: TextEmbeddingService,
    clip_embedder: ImageEmbeddingService,
    vs: VectorStoreService,
    bm25: BM25IndexService,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Runs retrieval queries and returns a validation report."""
    results = []

    for query in queries:
        query_embedding = embedder.embed_batch([query])[0]

        # Dense vector search
        vector_hits = vs.query_text(
            embedding=query_embedding,
            top_k=top_k,
            filters={"department": domain} if domain != "general" else None,
        )

        # BM25 sparse search
        bm25_hits = bm25.search(query, top_k=top_k) if bm25.is_built else []

        # Image retrieval (cross-modal) — MUST use CLIP 512-dim text encoder
        clip_query_embedding = clip_embedder.embed_text_for_image_search(query)
        image_hits = vs.query_images(embedding=clip_query_embedding, top_k=3) if clip_query_embedding else []

        result = {
            "query": query,
            "vector_hits": len(vector_hits),
            "bm25_hits": len(bm25_hits),
            "image_hits": len(image_hits),
            "top_chunk": None,
            "top_chunk_metadata": {},
            "linked_images": 0,
        }

        if vector_hits:
            top = vector_hits[0]
            meta = top.get("metadata", {})
            result["top_chunk"] = top.get("document", "")[:200]
            result["top_chunk_metadata"] = {
                "chunk_id": top.get("id", ""),
                "department": meta.get("department", ""),
                "subsystem": meta.get("subsystem", ""),
                "section_title": meta.get("section_title", ""),
                "contains_procedure": meta.get("contains_procedure", False),
                "contains_warning": meta.get("contains_warning", False),
                "contains_emergency": meta.get("contains_emergency_workflow", False),
                "importance": meta.get("importance", ""),
                "applicable_intents": meta.get("applicable_intents", "[]"),
                "has_prev_chunk": bool(meta.get("previous_chunk_id")),
                "has_next_chunk": bool(meta.get("next_chunk_id")),
                "related_images": meta.get("related_image_ids", "[]"),
                "distance": round(top.get("distance", 0), 4),
            }
            related_imgs = json.loads(meta.get("related_image_ids", "[]"))
            result["linked_images"] = len(related_imgs)

        results.append(result)

    return {"domain": domain, "query_results": results}


def print_validation_report(label: str, report: Dict[str, Any]):
    """Prints a formatted retrieval validation report to console."""
    print(f"\n{'='*70}")
    print(f"  RETRIEVAL VALIDATION: {label}")
    print(f"{'='*70}")
    for r in report["query_results"]:
        print(f"\n  Query: '{r['query']}'")
        print(f"    Vector hits : {r['vector_hits']} | BM25 hits: {r['bm25_hits']} | Image hits: {r['image_hits']}")
        print(f"    Linked imgs : {r['linked_images']}")
        if r["top_chunk"]:
            m = r["top_chunk_metadata"]
            print(f"    Top chunk   : [{m['chunk_id']}] (dist={m['distance']})")
            print(f"    Section     : {m['section_title']}")
            print(f"    Subsystem   : {m['subsystem']}")
            print(f"    Importance  : {m['importance']}")
            print(f"    Intents     : {m['applicable_intents']}")
            proc = "[Y]" if m["contains_procedure"] else "[N]"
            warn = "[Y]" if m["contains_warning"] else "[N]"
            emrg = "[Y]" if m["contains_emergency"] else "[N]"
            prev = "[Y]" if m["has_prev_chunk"] else "[N]"
            nxt  = "[Y]" if m["has_next_chunk"] else "[N]"
            print(f"    Procedure:{proc} Warning:{warn} Emergency:{emrg} | Prev:{prev} Next:{nxt}")
            print(f"    Preview: {r['top_chunk'][:150]}...")
        else:
            print(f"    [!] No results returned for this query")
    print()


def main():
    print("\n" + "="*70)
    print("  MARITIMEMIND AI — STAGED INGESTION & RETRIEVAL VALIDATION")
    print("="*70)

    vs = VectorStoreService()
    embedder = TextEmbeddingService()
    clip_embedder = ImageEmbeddingService()
    bm25 = BM25IndexService()
    pipeline = IngestionPipeline()

    all_reports = []
    summary = []

    for stage_num, pdf_spec in enumerate(STAGED_PDFS, 1):
        print(f"\n{'-'*70}")
        print(f"  STAGE {stage_num}/5: {pdf_spec['label']}")
        print(f"{'-'*70}")

        # ── Ingest ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        result = pipeline.run(pdf_spec["path"], force=True)
        elapsed = time.perf_counter() - t0

        print(f"  → {result}")

        if not result.success:
            print(f"  [FAILED] Skipping retrieval validation for this stage.")
            summary.append({
                "stage": stage_num,
                "label": pdf_spec["label"],
                "status": "FAILED",
                "chunks": 0,
                "images": 0,
                "elapsed": round(elapsed, 1),
            })
            continue

        print(f"  → {result.chunk_count} chunks | {result.image_count} images | {elapsed:.1f}s")

        # ── Rebuild BM25 after each ingest ───────────────────────────────────
        all_chunks = vs.get_all_text_chunks()
        if all_chunks:
            bm25.build_index(all_chunks)
            bm25.save()
            print(f"  → BM25 index updated: {len(all_chunks)} total chunks")

        # ── Retrieval Validation ─────────────────────────────────────────────
        report = run_retrieval_validation(
            queries=pdf_spec["queries"],
            domain=pdf_spec["domain"],
            embedder=embedder,
            clip_embedder=clip_embedder,
            vs=vs,
            bm25=bm25,
        )
        print_validation_report(pdf_spec["label"], report)
        all_reports.append({"stage": stage_num, "label": pdf_spec["label"], "report": report})

        summary.append({
            "stage": stage_num,
            "label": pdf_spec["label"],
            "status": "OK",
            "chunks": result.chunk_count,
            "images": result.image_count,
            "elapsed": round(elapsed, 1),
        })

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STAGED VALIDATION SUMMARY")
    print(f"{'='*70}")
    for s in summary:
        status_sym = "[OK]" if s["status"] == "OK" else "[FAIL]"
        print(f"  {status_sym} Stage {s['stage']}: {s['label']}")
        print(f"       chunks={s['chunks']} | images={s['images']} | time={s['elapsed']}s")

    # ── Save Full Report ─────────────────────────────────────────────────────
    report_path = Path("data/metadata/staged_validation_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_reports": all_reports}, f, indent=2, default=str)
    print(f"\n  Full report saved to: {report_path}")

    # Collection stats
    stats = vs.get_collection_stats()
    print(f"\n  ChromaDB state:")
    print(f"    Text chunks : {stats['text_collection']['count']}")
    print(f"    Images      : {stats['image_collection']['count']}")
    print()


if __name__ == "__main__":
    main()
