"""
app/api/routes/ingestion.py
============================
Endpoints to manage PDF ingestion into the knowledge base.

POST /api/v1/ingest            — Trigger ingestion for a PDF or directory
GET  /api/v1/ingest/status     — Return manifest (all ingested manuals)
DELETE /api/v1/ingest/{name}   — Remove a manual entry from the manifest
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from app.api.routes.auth import get_current_user

from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    IngestStatusResponse,
    IngestStatusItem,
)
from app.services.manifest import IngestionManifest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


def _run_ingestion(pdf_path: str, force: bool) -> IngestResponse:
    """Synchronous ingestion helper — called directly for single-file ingestion."""
    from app.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    try:
        result = pipeline.run(pdf_path=pdf_path, force=force)
        return IngestResponse(
            status="completed",
            manual_name=result.manual_name,
            chunk_count=result.chunk_count,
            image_count=result.image_count,
            errors=result.error_traceback.splitlines()[:3] if result.error_traceback else [],
        )
    except Exception as e:
        logger.exception(f"Ingestion failed for {pdf_path}: {e}")
        return IngestResponse(
            status="error",
            manual_name=Path(pdf_path).stem,
            errors=[str(e)],
        )


@router.post("", response_model=IngestResponse, summary="Trigger PDF ingestion")
async def ingest_pdf(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
) -> IngestResponse:
    """
    Triggers ingestion for a single PDF file or all PDFs in a directory.

    - If `pdf_path` is provided: ingests synchronously and returns counts.
    - If `pdf_dir` is provided: ingests asynchronously in the background.
    """
    if not request.pdf_path and not request.pdf_dir:
        raise HTTPException(status_code=422, detail="Provide either pdf_path or pdf_dir")

    if request.pdf_path:
        pdf = Path(request.pdf_path)
        if not pdf.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")
        # Run synchronously for single file (fast enough for API response)
        return _run_ingestion(request.pdf_path, request.force_reingest)

    # Directory ingestion — background task
    pdf_dir = Path(request.pdf_dir)
    if not pdf_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.pdf_dir}")

    def _bg_ingest_dir() -> None:
        from app.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        pipeline.run_directory(str(pdf_dir), force=request.force_reingest)

    background_tasks.add_task(_bg_ingest_dir)
    return IngestResponse(
        status="started",
        manual_name="batch",
        errors=[],
    )


@router.get("/status", response_model=IngestStatusResponse,
            summary="Get ingestion manifest status")
async def get_ingest_status(user: dict = Depends(get_current_user)) -> IngestStatusResponse:
    """Returns the full ingestion manifest — which manuals are indexed and their counts."""
    manifest = IngestionManifest()
    entries = manifest.all_entries()

    items = [
        IngestStatusItem(
            manual_name=name,
            status=data.get("status", "UNKNOWN"),
            chunk_count=data.get("chunk_count", 0),
            image_count=data.get("image_count", 0),
            processed_date=str(data.get("processed_date", "")),
        )
        for name, data in entries.items()
    ]
    return IngestStatusResponse(total_manuals=len(items), manuals=items)


@router.delete("/{manual_name}", status_code=204,
               summary="Remove a manual from the ingestion manifest")
async def delete_manual(manual_name: str, user: dict = Depends(get_current_user)) -> None:
    """
    Removes a manual entry from the manifest.
    Does NOT delete the actual ChromaDB vectors or BM25 index (re-ingest to update).
    """
    manifest = IngestionManifest()
    if manual_name not in manifest.all_entries():
        raise HTTPException(status_code=404, detail=f"Manual '{manual_name}' not in manifest")
    manifest._data.pop(manual_name, None)
    manifest.save()
