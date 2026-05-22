"""
Ingestion Manifest — Phase 2.5
Tracks which PDF manuals have been ingested, their status, and counts.
Enables idempotent re-ingestion: already-completed files are skipped.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.configs.config import settings
from app.models.schemas import IngestionManifestEntry
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.manifest")

MANIFEST_FILE = "ingestion_manifest.json"


class IngestionManifest:
    """
    Persistent JSON manifest tracking ingestion state for all PDFs.

    Stored at: {METADATA_DIR}/ingestion_manifest.json
    Structure:
        {
            "manual_name": {
                "status": "COMPLETED" | "FAILED" | "IN_PROGRESS",
                "processed_date": "ISO8601",
                "chunk_count": N,
                "image_count": N,
                "errors": []
            }
        }
    """

    STATUS_COMPLETED = "COMPLETED"
    STATUS_FAILED = "FAILED"
    STATUS_IN_PROGRESS = "IN_PROGRESS"

    def __init__(self) -> None:
        self._metadata_dir = Path(settings.METADATA_DIR)
        self._manifest_path = self._metadata_dir / MANIFEST_FILE
        self._data: Dict[str, dict] = {}
        self.load()

    def load(self) -> dict:
        """Loads manifest from disk. Returns empty dict if file doesn't exist."""
        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        if not self._manifest_path.exists():
            self._data = {}
            return self._data

        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info(f"Loaded manifest: {len(self._data)} entries")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load manifest, starting fresh: {e}")
            self._data = {}

        return self._data

    def save(self) -> None:
        """Persists manifest to disk as JSON."""
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            logger.debug(f"Manifest saved: {self._manifest_path}")
        except OSError as e:
            logger.error(f"Failed to save manifest: {e}")

    def update(
        self,
        manual_name: str,
        status: str,
        chunk_count: int = 0,
        image_count: int = 0,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Updates or creates a manifest entry for a manual."""
        self._data[manual_name] = {
            "status": status,
            "processed_date": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "image_count": image_count,
            "errors": errors or [],
        }
        self.save()

    def is_processed(self, manual_name: str) -> bool:
        """Returns True if the manual has a COMPLETED status in the manifest."""
        entry = self._data.get(manual_name, {})
        return entry.get("status") == self.STATUS_COMPLETED

    def get_entry(self, manual_name: str) -> Optional[IngestionManifestEntry]:
        """Returns typed IngestionManifestEntry for a manual, or None."""
        entry = self._data.get(manual_name)
        if not entry:
            return None
        return IngestionManifestEntry(
            status=entry["status"],
            processed_date=datetime.fromisoformat(entry["processed_date"]),
            chunk_count=entry["chunk_count"],
            image_count=entry["image_count"],
            errors=entry.get("errors", []),
        )

    def all_entries(self) -> Dict[str, dict]:
        """Returns the full raw manifest dictionary."""
        return dict(self._data)

    def mark_in_progress(self, manual_name: str) -> None:
        """Marks a manual as currently being ingested."""
        self.update(manual_name, self.STATUS_IN_PROGRESS)

    def mark_completed(
        self,
        manual_name: str,
        chunk_count: int,
        image_count: int
    ) -> None:
        """Marks a manual ingestion as successfully completed."""
        self.update(
            manual_name,
            self.STATUS_COMPLETED,
            chunk_count=chunk_count,
            image_count=image_count,
        )

    def mark_failed(self, manual_name: str, errors: List[str]) -> None:
        """Marks a manual ingestion as failed with error messages."""
        self.update(manual_name, self.STATUS_FAILED, errors=errors)
