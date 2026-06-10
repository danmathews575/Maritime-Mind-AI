<div align="center">

# MaritimeMind AI

**A multimodal, multi-agent Retrieval-Augmented Generation (RAG) system for maritime engineering operations.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9-DC143C?style=flat)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat)](./LICENSE)

*Deployed aboard vessels and in maritime operations centers. Ingests technical manuals, regulatory documents, and engineering schematics — answers complex operational queries with grounded, cited responses and retrieved engineering diagrams.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agent Pipeline](#agent-pipeline)
- [Retrieval Engine](#retrieval-engine)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Document Corpus](#document-corpus)
- [API Reference](#api-reference)
- [Scripts](#scripts)
- [Testing](#testing)
- [Demo Queries](#demo-queries)

---

## Overview

MaritimeMind AI is a production-grade AI assistant purpose-built for the maritime domain. Unlike generic chatbots, it operates on a **closed, verified corpus** of ship manuals — every answer is grounded in retrieved source documents and accompanied by source citations, confidence scores, and relevant engineering diagrams.

**Core capabilities:**

- Answer natural-language queries about engine procedures, fault diagnostics, regulatory compliance, and ship systems
- Retrieve and display engineering schematics, wiring diagrams, and cross-sectional drawings using CLIP visual search
- Detect query language automatically and respond in the same language (Arabic, Spanish, French, etc.)
- Stream responses token-by-token in real time via Server-Sent Events
- Route emergency queries through a dedicated fast-path that bypasses retry loops for immediate response

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MaritimeMind AI                             │
│                                                                     │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐   │
│  │   React UI  │    │              FastAPI Backend              │   │
│  │  (Vite/TS)  │◄──►│   /api/v1/chat/stream  (SSE Streaming)   │   │
│  └─────────────┘    └───────────────┬──────────────────────────┘   │
│                                     │                               │
│                         ┌───────────▼───────────┐                  │
│                         │   LangGraph Workflow   │                  │
│                         │   (6-Agent Pipeline)   │                  │
│                         └───────────┬───────────┘                  │
│                                     │                               │
│              ┌──────────────────────┼──────────────────────┐       │
│              │                      │                       │       │
│   ┌──────────▼──────┐  ┌───────────▼──────┐  ┌────────────▼────┐  │
│   │  Qdrant Vector  │  │   BM25 Pickle    │  │  Redis Cache    │  │
│   │  Store (local)  │  │   Index (sparse) │  │  (responses)    │  │
│   └─────────────────┘  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

The system is built on a **stateful LangGraph directed graph** with six specialized agents. Each query traverses a different path through the graph depending on classified intent.

```
                         ┌──────────────────────┐
                         │   Context Router      │
                         │   Agent               │
                         │                       │
                         │  • Query expansion    │
                         │  • Intent classifier  │
                         │  • Strategy selector  │
                         └──────┬───────────────-┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
  ┌───────────────┐    ┌────────────────┐    ┌─────────────────┐
  │ Visual        │    │ Text Retrieval │    │ Diagnosis       │
  │ Specialist    │    │ Node           │    │ Agent           │
  │               │    │                │    │                 │
  │ • CLIP search │    │ • BM25 search  │    │ • Fault trees   │
  │ • OCR match   │    │ • Dense vector │    │ • Root cause    │
  │ • Page prox.  │    │ • RRF fusion   │    │ • Structured    │
  └───────┬───────┘    └───────┬────────┘    │   diagnosis     │
          │                    │             └────────┬────────┘
          └────────────────────┤                      │
                               ▼                      │
                    ┌──────────────────┐              │
                    │ Retrieval        │              │
                    │ Verifier         │              │
                    │                  │              │
                    │ • Conf. scoring  │              │
                    │ • Query rewrite  │              │
                    │   on low conf.   │              │
                    └──────────┬───────┘              │
                               │                      │
                               ▼                      │
                    ┌──────────────────┐              │
                    │ Synthesizer      │              │
                    │ Agent            │◄─────────────┘
                    │                  │
                    │ • Grounded LLM   │
                    │ • Strict citng.  │
                    │ • Safety rules   │
                    └──────────┬───────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Quality Reviewer │
                    │                  │
                    │ • Halluc. check  │
                    │ • Retry if fail  │
                    └──────────┬───────┘
                               │
                               ▼
                    Final Answer + Citations
                    + Diagram Cards + Conf. Score
```

### Intent → Strategy Routing Table

| Intent | Triggered by | Retrieval Strategy | Special Behaviour |
|---|---|---|---|
| `PROCEDURE` | "How do I...", "Steps to..." | `multimodal` | Fetches diagrams — procedures reference schematics |
| `TROUBLESHOOTING` | "Why is X happening", "Root cause of..." | `multimodal` + Diagnosis Agent | Routes to structured fault-tree agent |
| `DIAGRAM_REQUEST` | "Show me the... diagram" | `image_priority` | CLIP visual search runs first |
| `EMERGENCY` | "Flooding", "Fire", "Immediate actions" | `emergency` | Fast-path — zero retries, max speed |
| `SOP_LOOKUP` | "MARPOL", "SOLAS", "Regulation..." | `text_only` | Regulatory text chunks prioritised |
| `EXPLANATION` | "What is...", "Explain..." | `text_only` | Dense vector search for conceptual content |

---

## Retrieval Engine

The retrieval layer combines four complementary signals before reranking.

```
Query
  │
  ├─── [1] Dense Vector Search ──► paraphrase-multilingual-MiniLM-L12-v2 (384-dim, cosine)
  │                                Qdrant HNSW index
  │
  ├─── [2] Sparse BM25 Search ───► rank-bm25 (TF-IDF term matching)
  │                                In-memory pickle index
  │
  ├─── [3] CLIP Image Search ────► ViT-B-32 (512-dim, visual similarity)
  │                                Separate Qdrant collection
  │
  └─── [4] EasyOCR Payload ──────► Diagram label text extracted at ingestion
                                   Qdrant payload filter match
          │
          ▼
  Reciprocal Rank Fusion (RRF)
  k=60, combines all ranked lists
          │
          ▼
  Cross-Encoder Reranker
  ms-marco-MiniLM-L-12-v2
  Re-scores top-N candidates
          │
          ▼
  Confidence Scoring
  Composite: reranker score + BM25 overlap + section match
          │
          ▼
  Context Expansion
  ±1 adjacent chunks retrieved for continuity
          │
          ▼
  Verified Context → Synthesizer
```

### On Low Confidence — Automatic Query Rewriting

If retrieval confidence falls below the threshold, the system automatically rewrites the query using an LLM before retrying (max 2 retries, non-emergency):

```
Original query: "jacket water pressure fault"
          │
          ▼  [Retrieval Verifier: confidence < threshold]
          │
LLM Rewrite: "jacket cooling water system low pressure causes remedies
              marine diesel engine"
          │
          ▼  [Retry text retrieval with expanded query]
```

---

## Technology Stack

### Backend

| Component | Library / Version | Purpose |
|---|---|---|
| API Framework | FastAPI 0.115 + Uvicorn 0.34 | Async HTTP server, SSE streaming |
| Agent Framework | LangGraph 0.3 + LangChain 0.3 | Stateful multi-agent workflow |
| LLM Providers | Gemini / OpenAI / Ollama | Configurable LLM backend |
| Vector Store | Qdrant 1.9 (local or remote) | Dense + image vector search |
| Text Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | 384-dim multilingual semantic embeddings |
| Image Embeddings | CLIP ViT-B-32 (open-clip-torch) | 512-dim visual embeddings |
| Sparse Search | rank-bm25 | TF-IDF keyword search |
| Reranker | ms-marco-MiniLM-L-12-v2 | Cross-encoder relevance reranking |
| OCR | EasyOCR 1.7 | Diagram label text extraction |
| PDF Processing | PyMuPDF 1.25 + pdfplumber 0.11 | Text and image extraction |
| Token Counting | tiktoken 0.7 | Accurate chunk sizing |
| Language Detection | langdetect 1.0 | Multilingual query routing |
| Caching | Redis 7.2 + redis-py 5.0 | Response caching |
| Auth | PyJWT + bcrypt + slowapi | JWT auth + rate limiting |
| Observability | Arize Phoenix + OpenInference | LangChain trace visualization |
| Data validation | Pydantic v2 + pydantic-settings | Config and schema validation |

### Frontend

| Component | Library / Version | Purpose |
|---|---|---|
| Framework | React 19 + Vite 8 | SPA with HMR dev server |
| Language | TypeScript ~6.0 | Type safety |
| Styling | Tailwind CSS v4 | Utility-first CSS |
| UI Components | shadcn/ui | Accessible component primitives |
| Animation | Framer Motion 12 | Chat animations, lightbox transitions |
| Icons | Lucide React | Icon system |
| Markdown | react-markdown + remark-gfm | Formatted AI responses |
| Routing | React Router v7 | SPA navigation |
| Fonts | Geist Variable | Typography |

### Infrastructure

| Service | Image | Purpose |
|---|---|---|
| Vector DB | `qdrant/qdrant:v1.9.2` | Vector storage |
| Cache | `redis:7.2-alpine` | Response caching (256 MB LRU) |
| Observability | `arizephoenix/phoenix:latest` | Trace monitoring (port 6006) |

---

## Project Structure

```
maritimemind-ai/
│
├── app/                              # Python backend
│   │
│   ├── agents/                       # LangGraph agent nodes
│   │   ├── state.py                  # AgentState TypedDict (shared graph state)
│   │   ├── router.py                 # Context Router — intent + strategy assignment
│   │   ├── visual_specialist.py      # CLIP + OCR image retrieval agent
│   │   ├── verification.py           # Retrieval Verifier — confidence + retry logic
│   │   ├── synthesizer.py            # Synthesizer — grounded LLM response generation
│   │   ├── quality_reviewer.py       # Quality Reviewer — hallucination check
│   │   └── diagnosis_agent.py        # Diagnosis Agent — structured fault analysis
│   │
│   ├── orchestration/
│   │   └── graph.py                  # LangGraph graph definition + run_agent_workflow()
│   │
│   ├── retrieval/                    # Hybrid retrieval engine
│   │   ├── controller.py             # RetrievalController — orchestrates all search paths
│   │   ├── hybrid_search.py          # BM25 + dense vector + RRF fusion
│   │   ├── image_retrieval.py        # CLIP visual search with OCR payload filtering
│   │   ├── reranker.py               # Cross-encoder reranking (ms-marco)
│   │   ├── scoring.py                # Composite confidence scoring
│   │   ├── context_expander.py       # Adjacent chunk expansion (±1 chunk)
│   │   └── query_classifier.py       # Rule + embedding intent classification
│   │
│   ├── services/                     # Singleton service layer (model holders)
│   │   ├── embedding.py              # TextEmbeddingService (multilingual MiniLM, lazy-loaded)
│   │   ├── clip_embedding.py         # ImageEmbeddingService (CLIP, lazy-loaded)
│   │   ├── bm25_index.py             # BM25IndexService (pickle load/save)
│   │   ├── vector_store.py           # Qdrant client wrapper + collection helpers
│   │   ├── llm_service.py            # Multi-provider LLM abstraction
│   │   ├── chunker.py                # Semantic text chunking with tiktoken
│   │   └── association.py            # Text↔image association scoring
│   │
│   ├── ingestion/
│   │   └── pipeline.py               # PDF → extract → chunk → embed → Qdrant
│   │
│   ├── api/
│   │   ├── main.py                   # FastAPI app factory, lifespan, middleware
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── routes/
│   │       ├── query.py              # POST /api/v1/chat/stream (SSE)
│   │       ├── health.py             # GET  /api/v1/health
│   │       ├── sessions.py           # GET  /api/v1/sessions/{id}
│   │       ├── ingestion.py          # POST /api/v1/ingest
│   │       └── auth.py               # POST /api/v1/auth/token
│   │
│   ├── memory/
│   │   └── query_expander.py         # Conversational query expansion from history
│   │
│   ├── configs/
│   │   ├── config.py                 # Settings class (pydantic-settings, .env)
│   │   └── limiter.py                # slowapi rate limiter instance
│   │
│   └── utils/
│       ├── logger.py                 # Structured logging setup
│       └── language.py               # langdetect wrapper + language name mapping
│
├── frontend/src/
│   ├── pages/
│   │   ├── Home.tsx                  # Landing page (hero, feature cards)
│   │   └── Chat.tsx                  # Chat UI (streaming, citations, lightbox)
│   └── components/ui/                # shadcn/ui primitives
│
├── data/
│   ├── raw_pdfs/                     # Source PDFs by department (gitignored)
│   │   ├── engineering/
│   │   ├── deck/
│   │   ├── navigation/
│   │   └── safety/
│   ├── extracted_images/             # CLIP-indexed diagram images (gitignored)
│   ├── demo/
│   │   └── demo_queries.json         # Categorized demo query set
│   └── metadata/
│       ├── ingestion_manifest.json   # Per-manual chunk + image counts
│       └── staged_validation_report.json
│
├── vector_store/                     # Qdrant local storage + BM25 index (gitignored)
│
├── scripts/
│   ├── ingest.py                     # Run ingestion pipeline
│   ├── download_models.py            # Pre-download model weights (~1.5 GB)
│   ├── precache.py                   # Pre-warm Redis with demo queries
│   ├── evaluate.py                   # RAG evaluation against benchmark_queries.json
│   ├── staged_validation.py          # End-to-end retrieval validation suite
│   ├── generate_corpus_report.py     # Print corpus statistics
│   ├── cleanup_images.py             # Remove orphaned extracted images
│   ├── backup.py / restore.py        # Vector store backup and restore
│   └── test_agent_workflow.py        # Manual agent pipeline smoke test
│
├── tests/                            # pytest test suite
│   ├── retrieval/
│   │   ├── test_hybrid.py
│   │   ├── test_scoring.py
│   │   └── test_classifier.py
│   ├── test_vector_store.py
│   ├── test_bm25.py
│   └── test_schemas.py
│
├── logs/                             # Application logs (gitignored)
├── docker-compose.yml                # Full stack: backend + frontend + Qdrant + Redis + Phoenix
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt                  # Python dependencies (pinned)
├── .env.example                      # Environment variable template
└── pytest.ini
```

---

## Getting Started

### Prerequisites

Please ensure the following are installed on your system before proceeding:

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.10 | Tested on 3.11. Required to run the backend API and ingestion scripts. |
| **Node.js** | 18.x | Required to run the React frontend. |
| **Docker & Docker Compose** | Latest stable | Required to run Qdrant (Vector DB) and Redis (Caching). |
| **RAM** | 8 GB | 16 GB recommended when keeping CLIP and reranker models in memory. |
| **Disk Space** | 5 GB free | Required for model weights and vector store persistence. |

---

### Local Development Setup

Follow these exact steps to set up and run the application successfully without any manual missing dependency installations.

#### Step 1: Clone & Configure

Clone the repository and copy the example environment variables:

```bash
git clone https://github.com/danmathews575/ai_chatbot.git
cd ai_chatbot/maritimemind-ai
cp .env.example .env
```

Edit the `.env` file and set your LLM provider and API keys. At a minimum, set:

```env
# Recommended: Gemini (no local GPU required)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here

# Or: Local Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b
```

#### Step 2: Start Infrastructure (Docker)

Start the Qdrant vector database and Redis cache using Docker Compose:

```bash
docker-compose up -d vector-db cache
```

#### Step 3: Install Python Dependencies

Create a virtual environment and install all dependencies from `requirements.txt`:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### Step 4: Download AI Models

Run the download script to fetch the required local models (text embedder, image embedder, and reranker). This will download ~1.5 GB of data.

```bash
python scripts/download_models.py
```

#### Step 5: Document & Image Ingestion Process

Before the chatbot can answer queries, you must ingest the maritime manuals into the vector database.

1. **Prepare your PDFs**: Ensure that your PDF manuals are placed in the correct department subdirectories within `data/raw_pdfs/`. For example:
   - `data/raw_pdfs/engineering/`
   - `data/raw_pdfs/deck/`
   - `data/raw_pdfs/navigation/`
   - `data/raw_pdfs/safety/`
2. **Run the Ingestion Pipeline**: Execute the ingestion script to process all PDFs in these directories.
   ```bash
   python scripts/ingest.py
   ```
   **What this does**:
   - Extracts text and semantically chunks it.
   - Embeds text chunks and stores them in Qdrant (`text_chunks`).
   - Extracts diagrams and OCR labels from images.
   - Embeds images using CLIP and stores them in Qdrant (`image_chunks`).
   - Builds and persists the BM25 sparse index for keyword searches.

*(Optional)* If you want to ingest a specific PDF file individually instead of the whole directory, use:
```bash
python scripts/ingest.py --pdf path/to/your/manual.pdf
```

#### Step 6: Start the Backend API

With the documents ingested, start the FastAPI backend server:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
*Note: On startup, the API pre-warms all models to eliminate cold-start latency. The first startup might take ~15 seconds.*
- **API URL**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`

#### Step 7: Start the Frontend UI

You have two options for the frontend interface. The primary React UI is recommended.

**Option A: React UI (Primary)**
Open a new terminal, navigate to the `frontend` directory, install Node modules, and start the development server:

```bash
cd frontend
npm install
npm run dev
```
- **React App**: `http://localhost:5173`

**Option B: Streamlit UI (Alternative)**
If you prefer a lightweight Python-based interface for testing, ensure your virtual environment is activated and run:

```bash
streamlit run app/ui/streamlit_app.py
```
- **Streamlit App**: `http://localhost:8501`

---

### Option B — Docker Compose (Full Stack)

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and API key

docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| Backend API | `http://localhost:8000` |
| API Docs | `http://localhost:8000/docs` |
| Qdrant Dashboard | `http://localhost:6333/dashboard` |
| Arize Phoenix | `http://localhost:6006` |

---

## Configuration

All settings are loaded from `.env` via `pydantic-settings`. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `ollama` |
| `GEMINI_API_KEY` | — | Comma-separated for rate-limit key rotation |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3:8b` | Ollama model name |
| `QDRANT_HOST` | `local` | `local` (file-based) or server hostname |
| `QDRANT_PATH` | `./vector_store/qdrant_local` | Local Qdrant storage path |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `BM25_INDEX_PATH` | `./vector_store/bm25_index.pkl` | BM25 index pickle path |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `CONFIDENCE_THRESHOLD` | `0.6` | Min retrieval confidence before refusal (0–1) |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `EXTRACTED_IMAGES_DIR` | `./data/extracted_images` | Diagram image output directory |
| `JWT_SECRET_KEY` | — | **Change before any deployment** |
| `TEXT_EMBEDDING_DIM` | `384` | Multilingual MiniLM embedding dimension |
| `IMAGE_EMBEDDING_DIM` | `512` | CLIP embedding dimension |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins |
| `VITE_API_URL` | `http://localhost:8000` | Frontend → backend base URL |

---

## Document Corpus

The system has been validated against the following ingested corpus:

| Manual / Document | Dept. | Text Chunks | Images |
|---|---|---|---|
| Marine Diesel Engine Manual 1 | Engineering | 489 | 60 |
| Marine Diesel Engine Manual 2 | Engineering | 242 | 25 |
| Wärtsilä 26 Maintenance Manual | Engineering | 202 | 3 |
| Ship Cooling System | Engineering | 86 | 3 |
| Ship Fuel System | Engineering | 65 | 1 |
| Radar Manual | Navigation | 429 | 304 |
| ECDIS Handbook | Navigation | 65 | 29 |
| INTERTANKO Anchoring Guidelines | Deck | 166 | 4 |
| POTLL Mooring Manual | Deck | 146 | 61 |
| Framo Ballast Operation Manual | Deck | 22 | 12 |
| Ballast Loss Prevention Article | Deck | 17 | 5 |
| Cargo Handling Manual | Deck | 17 | 6 |
| Engine Room Fires (TSC) | Safety | 18 | 17 |
| Ship Evacuation Guidelines | Safety | 22 | 2 |
| MARPOL Annex I / OPA / VRP | Regulatory | — | 20 |
| USCG Marine Safety Manual Vol III | Regulatory | 1 | — |
| SHIP CON 5 Student Sections | Construction | — | 23 |
| Construction — Stem | Construction | — | 21 |
| **Total** | | **≈ 2,200 chunks** | **≈ 600 images** |

> PDFs that are scanned documents (no extractable text layer) are indexed via CLIP image embeddings and EasyOCR only.

---

## API Reference

### `POST /api/v1/chat/stream`

Main query endpoint. Returns a Server-Sent Events (SSE) stream.

**Request body:**
```json
{
  "query": "What is the procedure for starting the main engine?",
  "session_id": "session-1234567890-abc",
  "filters": {
    "ship_id": "MV_AURORA"
  }
}
```

**SSE event stream:**
```
data: {"type": "metadata", "data": {
         "intent": "procedure",
         "confidence": 0.87,
         "detected_language": "en",
         "citations": [{"manual_name": "...", "page_number": 42, "section_title": "..."}],
         "images": [{"url": "/static/images/...", "caption": "...", "diagram_type": "SCHEMATIC"}]
       }}

data: {"type": "token", "data": "The "}
data: {"type": "token", "data": "main engine "}
data: {"type": "token", "data": "starting procedure..."}
data: {"type": "done"}
```

### Other Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System health, model warm status, Qdrant stats |
| `GET` | `/api/v1/sessions/{session_id}` | Retrieve session conversation history |
| `POST` | `/api/v1/ingest` | Trigger document ingestion pipeline |
| `POST` | `/api/v1/auth/token` | Obtain JWT access token |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

---

## Scripts

```bash
# Ingest all PDFs from data/raw_pdfs/ into the vector store
python scripts/ingest.py

# Pre-download all AI model weights (~1.5 GB)
python scripts/download_models.py

# Pre-warm Redis cache with categorized demo queries
python scripts/precache.py --username admin --password password

# Run RAG evaluation against benchmark queries
python scripts/evaluate.py

# End-to-end retrieval validation across all document categories
python scripts/staged_validation.py

# Print corpus statistics (chunk counts, image counts per manual)
python scripts/generate_corpus_report.py

# Remove orphaned extracted images not present in Qdrant
python scripts/cleanup_images.py

# Backup vector store and BM25 index
python scripts/backup.py

# Restore from backup
python scripts/restore.py

# Manual agent pipeline smoke test
python scripts/test_agent_workflow.py
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v -s

# Run specific modules
pytest tests/retrieval/ -v
pytest tests/test_vector_store.py -v
pytest tests/test_bm25.py -v
```

Test suite covers: retrieval scoring, hybrid search fusion, BM25 index, query classification, vector store operations, and Pydantic schema validation.

---

## Demo Queries

The following categorized queries demonstrate each intent path through the agent pipeline:

| Category | Query | Agent Path |
|---|---|---|
| **Procedure** | *"What is the procedure for starting the main engine?"* | Router → Text Retrieval → Verifier → Synthesizer |
| **Procedure** | *"How do I perform a main engine slow turning before starting?"* | Router → Multimodal Retrieval → Verifier → Synthesizer |
| **Emergency** | *"Engine room flooding — immediate actions"* | Router → Emergency Fast-Path → Synthesizer |
| **Emergency** | *"Main engine fire emergency procedure"* | Router → Emergency Fast-Path → Synthesizer |
| **Troubleshooting** | *"High lube oil temperature alarm on main engine — root cause"* | Router → Diagnosis Agent → Retrieval → Synthesizer |
| **Troubleshooting** | *"The Wärtsilä 26 is showing high EGT on cylinder No. 3 — diagnostic procedure and fuel injection diagram"* | Router → Diagnosis Agent → Visual Specialist → Synthesizer |
| **Diagram** | *"Show me the cooling water system schematic"* | Router → Visual Specialist → Text Retrieval → Synthesizer |
| **Diagram** | *"Main engine fuel oil system diagram"* | Router → Visual Specialist → Synthesizer |
| **Regulatory** | *"What is MARPOL Annex VI NOx Tier III limit?"* | Router → Text Retrieval → Verifier → Synthesizer |
| **Regulatory** | *"SOLAS Chapter II-2 fire detection requirements"* | Router → Text Retrieval → Verifier → Synthesizer |
| **Explanation** | *"Explain the principle of operation of a turbocharger"* | Router → Text Retrieval → Synthesizer |
| **Multilingual** | *"¿Cuál es el procedimiento de arranque del motor principal?"* | Router → Language Detection → Text Retrieval → Synthesizer (replies in Spanish) |

---

## Operational Notes

- **Model warm-up**: The API pre-warms all models at startup. First request after a cold start takes ~15s; subsequent requests are fast.
- **Qdrant local mode**: By default (`QDRANT_HOST=local`) Qdrant runs as an embedded file-based store — no separate Docker service is needed for development.
- **LLM key rotation**: Multiple `GEMINI_API_KEY` values (comma-separated) are rotated automatically to distribute rate limits.
- **Emergency fast-path**: Emergency-intent queries skip all retry loops and proceed directly to synthesis — speed is prioritised over confidence validation.
- **Redis fallback**: If Redis is unavailable the system continues without caching; responses are generated fresh each time.

---

## License

Submitted as an academic and professional portfolio project. All maritime technical documents used are publicly available reference materials.
