import os
import subprocess
from pathlib import Path

PDFS_TO_PROCESS = [
    "data/raw_pdfs/navigation/radar manual.pdf",
    "data/raw_pdfs/safety/EngineRoomFires_TSC.pdf",
    "data/raw_pdfs/safety/MARPOL Anx I - OPA - VRP - VGP.pdf",
    "data/raw_pdfs/safety/Ship_Evacuation-Guidelines_Simulation_Validation_a.pdf",
    "data/raw_pdfs/safety/USCG-MSM-VOL-III-CHAP-16.pdf"
]

def main():
    for pdf_rel_path in PDFS_TO_PROCESS:
        pdf_path = Path(pdf_rel_path).resolve()
        print(f"Running ingestion for: {pdf_path.name}")
        subprocess.run(["python", "scripts/ingest.py", "--force", "--pdf", str(pdf_path)], check=True)

if __name__ == "__main__":
    main()
