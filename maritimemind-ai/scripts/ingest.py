#!/usr/bin/env python
"""
scripts/ingest.py — MaritimeMind AI Ingestion CLI
Phase 2 — Updated entry point

Drives the IngestionPipeline from the command line.

Examples
--------
# Ingest all PDFs in the default data directory:
    python scripts/ingest.py

# Ingest all PDFs in a custom directory:
    python scripts/ingest.py --dir data/raw_pdfs/

# Ingest a single PDF file:
    python scripts/ingest.py --pdf data/raw_pdfs/ship_manual.pdf

# Force re-ingestion (even if already completed in manifest):
    python scripts/ingest.py --force

# Show ingestion manifest status:
    python scripts/ingest.py --status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pipeline import IngestionPipeline
from app.services.manifest import IngestionManifest
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.cli.ingest")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="MaritimeMind AI — Multimodal PDF Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--pdf",
        metavar="PATH",
        help="Path to a single PDF file to ingest",
    )
    source.add_argument(
        "--dir",
        metavar="DIR",
        default="data/raw_pdfs",
        help="Directory of PDFs to ingest (default: data/raw_pdfs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-ingest files even if already marked COMPLETED in manifest",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Print the ingestion manifest and exit",
    )
    return parser


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def show_status() -> None:
    """Print the current ingestion manifest as formatted JSON."""
    manifest = IngestionManifest()
    data = manifest.load()
    if not data:
        print("Manifest is empty — no files have been ingested yet.")
        return
    print(json.dumps(data, indent=2, default=str))


def run_ingestion(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline. Returns exit code (0 = success)."""
    pipeline = IngestionPipeline()

    if args.pdf:
        # ── Single-file mode ──────────────────────────────────────────
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.is_file():
            logger.error(f"File not found: {pdf_path}")
            print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
            return 1

        result = pipeline.run(str(pdf_path), force=args.force)
        print(str(result))
        return 0 if result.success else 1

    else:
        # ── Directory batch mode ──────────────────────────────────────
        pdf_dir = Path(args.dir).resolve()
        if not pdf_dir.is_dir():
            logger.error(f"Directory not found: {pdf_dir}")
            print(f"ERROR: Directory not found: {pdf_dir}", file=sys.stderr)
            return 1

        results = pipeline.run_directory(str(pdf_dir), force=args.force)

        # Print summary table
        print("\n" + "=" * 70)
        print(f"  INGESTION SUMMARY  ({len(results)} file(s) processed)")
        print("=" * 70)
        for r in results:
            print(f"  {r}")
        print("=" * 70)

        failed = [r for r in results if not r.success and not r.skipped]
        if failed:
            print(f"\n  {len(failed)} file(s) FAILED. See logs for details.")
            return 1
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.status:
        show_status()
        sys.exit(0)

    exit_code = run_ingestion(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
