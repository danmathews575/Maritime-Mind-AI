import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
from app.services.vector_store import VectorStoreService
from app.retrieval.controller import RetrievalController
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.eval.corpus_report")

def generate_report():
    logger.info("Initializing vector store for statistics gathering...")
    vs = VectorStoreService()
    
    # Gathering chunks
    text_chunks = vs.get_all_text_chunks()
    total_chunks = len(text_chunks)
    
    # Gathering images
    # We don't have a direct `get_all_images` method in VectorStoreService currently.
    # We can get image count from stats
    stats = vs.get_collection_stats()
    total_images = stats.get("image_collection", {}).get("count", 0)
    
    # We might need to query chroma directly for all images to get metadata for linking stats
    collection = vs._get_or_create_image_collection()
    img_results = collection.get(include=["metadatas"])
    img_metas = img_results.get("metadatas", [])
    
    department_counts = Counter()
    subsystem_counts = Counter()
    chunks_with_images = 0
    images_with_chunks = 0
    
    for chunk in text_chunks:
        department_counts[chunk.department] += 1
        subsystem_counts[chunk.subsystem] += 1
        if chunk.related_image_ids:
            chunks_with_images += 1
            
    for meta in img_metas:
        if meta and meta.get("related_chunk_ids") and len(json.loads(meta.get("related_chunk_ids", "[]"))) > 0:
            images_with_chunks += 1

    report = {
        "statistics": {
            "total_chunks": total_chunks,
            "total_images": total_images,
            "department_coverage": dict(department_counts),
            "subsystem_coverage": dict(subsystem_counts),
            "multimodal_linking": {
                "chunks_with_linked_images": chunks_with_images,
                "images_with_linked_chunks": images_with_chunks,
                "percent_chunks_grounded_to_images": round((chunks_with_images / max(1, total_chunks)) * 100, 2),
                "percent_images_grounded_to_chunks": round((images_with_chunks / max(1, total_images)) * 100, 2),
            }
        },
        "query_evaluations": {}
    }

    queries = {
        "Engineering": [
            "Why is fuel pressure low?",
            "Explain cooling water circulation.",
            "Show injector layout."
        ],
        "Safety": [
            "Show engine room fire response.",
            "Display evacuation procedure."
        ],
        "Deck Operations": [
            "Explain ballast operation.",
            "Show mooring workflow."
        ],
        "Navigation": [
            "Explain radar target tracking.",
            "Show ECDIS interface."
        ]
    }

    logger.info("Running evaluation queries...")
    controller = RetrievalController()
    
    for category, q_list in queries.items():
        report["query_evaluations"][category] = {}
        for q in q_list:
            logger.info(f"Evaluating: '{q}'")
            results = controller.retrieve(q, top_k=3)
            
            # Extract top chunks and images
            top_chunks = []
            for r in results:
                top_chunks.append({
                    "chunk_id": r.chunk.chunk_id,
                    "confidence_score": round(r.scores.confidence_score, 3),
                    "subsystem": r.chunk.subsystem,
                    "snippet": r.chunk.content[:100] + "..."
                })
            
            top_images = []
            if results and results[0].images:
                for img in results[0].images:
                    top_images.append({
                        "image_id": img.metadata.image_id,
                        "caption": img.metadata.caption,
                        "final_score": round(img.explainability.final_score, 3),
                        "retrieval_reasons": img.explainability.retrieval_reason
                    })
                    
            report["query_evaluations"][category][q] = {
                "retrieved_chunks_count": len(top_chunks),
                "retrieved_images_count": len(top_images),
                "top_chunks": top_chunks,
                "top_images": top_images
            }

    # Export report
    output_path = Path("evaluation/final_corpus_report.json")
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Report exported to {output_path}")

if __name__ == "__main__":
    generate_report()
