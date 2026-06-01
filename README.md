# MaritimeMind AI

**A multimodal, multi-agent RAG system for maritime engineering operations.**

MaritimeMind AI is a production-grade AI assistant built for shipboard and shore-based maritime engineers. It ingests technical manuals, engineering schematics, regulatory documents, and maintenance procedures — then answers complex operational queries with grounded, cited responses and retrieved engineering diagrams.

Built as an **academic and professional showcase** of advanced AI engineering: multi-agent orchestration, multimodal retrieval, hybrid search, and a real-time streaming interface.

---

## Live Demo

| Page | Description |
|---|---|
| **Landing** | Feature overview — Safety Regulations, Engine Diagnostics, Multilingual Q&A |
| **Chat** | Streaming AI responses with intent routing, source citations, confidence scoring, and zoomable diagrams |

**Example query to try:**
> *"The Wärtsilä 26 main engine is showing high exhaust gas temperature on cylinder No. 3. What is the diagnostic procedure and show me the fuel injection system diagram?"*

---

## Architecture

The core is a **6-Agent LangGraph pipeline** that dynamically routes queries, verifies context quality, and synthesizes grounded answers with hallucination prevention.

```
User Query
    │
    ▼
┌──────────────────┐
│  Context Router  │  ← Classifies intent: procedure / troubleshooting /
│      Agent       │    diagram_request / emergency / sop_lookup / explanation
└──────┬───────────┘
       │
   ┌───┴──────────────────────────────────────┐
   │                                          │
   ▼                                          ▼
┌──────────────────┐                ┌──────────────────┐
│ Visual Specialist│                │ Diagnosis Agent  │
│  (CLIP + OCR)    │                │  (Troubleshoot)  │
└──────┬───────────┘                └──────┬───────────┘
       │                                   │
       ▼                                   │
┌──────────────────┐                       │
│  Text Retrieval  │ ◄─────────────────────┘
│ Hybrid BM25+Vec  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Retrieval       │  ← Confidence scoring, retry with LLM query rewriting
│  Verifier        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Synthesizer     │  ← Grounded LLM generation with strict citation rules
│     Agent        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Quality Reviewer │  ← Hallucination check, retry loop if failed
└──────┬───────────┘
       │
       ▼
  Final Answer + Diagram Cards + Citations
```

### Retrieval Layer — Hybrid Multimodal Search

```
Query
  ├── Dense Vector Search   (BAAI/bge-base-en-v1.5, 768-dim)
  ├── Sparse BM25 Search    (rank-bm25, pickle index)
  ├── CLIP Image Search     (ViT-B-32, 512-dim, visual similarity)
  └── EasyOCR Text Match    (diagram label extraction)
          │
          ▼
  Reciprocal Rank Fusion (RRF)
          │
          ▼
  Cross-Encoder Reranker (ms-marco-MiniLM-L-12-v2)
          │
          ▼
  Confidence Scoring + Context Expansion
```

---

## Key Features

- **6-Agent LangGraph Orchestration** — Specialized agents for routing, visual retrieval, diagnostics, synthesis, and quality review
- **Hybrid Multimodal Retrieval** — Dense + sparse + CLIP image search with RRF fusion and cross-encoder reranking
- **Engineering Diagram Intelligence** — CLIP embeddings + EasyOCR extract and index diagrams, schematics, and wiring diagrams from PDFs
- **Hallucination Prevention** — Strict system prompt rules, retrieval verification, quality review loop, and confidence scoring
- **Multilingual Support** — Auto-detects query language (Arabic, Spanish, French, etc.) and responds in the same language
- **Streaming Responses** — Server-Sent Events (SSE) for real-time token-by-token streaming
- **Emergency Fast-Path** — Emergency queries bypass retry loops for immediate response
- **Redis Caching** — Sub-100ms repeated query response times
- **JWT Authentication** — Secure API endpoints with rate limiting

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI 0.115, Uvicorn, Pydantic v2 |
| **Agent Framework** | LangGraph 0.3, LangChain 0.3 |
| **LLM Providers** | Ollama (Llama 3), Google Gemini, OpenAI (configurable) |
| **Vector Store** | Qdrant v1.9 (local or remote) |
| **Text Embeddings** | BAAI/bge-base-en-v1.5 (sentence-transformers) |
| **Image Embeddings** | CLIP ViT-B-32 (open-clip-torch) |
| **Reranker** | ms-marco-MiniLM-L-12-v2 (cross-encoder) |
| **Sparse Search** | BM25 (rank-bm25) |
| **OCR** | EasyOCR |
| **PDF Processing** | PyMuPDF, pdfplumber |
| **Caching** | Redis 7.2 |
| **Observability** | Arize Phoenix (LangChain tracing) |
| **Frontend** | React 19, Vite 8, TypeScript, Tailwind CSS v4 |
| **UI Components** | shadcn/ui, Framer Motion, Lucide React |
| **Auth** | JWT (PyJWT), bcrypt, slowapi rate limiting |
| **Testing** | pytest, pytest-asyncio |

---

## Project Structure

```
maritimemind-ai/
│
├── app/                          # Backend application
│   ├── agents/                   # LangGraph agent nodes
│   │   ├── router.py             # Context Router — intent classification
│   │   ├── visual_specialist.py  # CLIP-based diagram retrieval agent
│   │   ├── verification.py       # Retrieval Verifier — confidence check
│   │   ├── synthesizer.py        # Response Synthesis — grounded LLM generation
│   │   ├── quality_reviewer.py   # Quality Review — hallucination prevention
│   │   ├── diagnosis_agent.py    # Diagnosis Agent — troubleshooting workflows
│   │   └── state.py              # Shared AgentState TypedDict
│   │
│   ├── api/                      # FastAPI application
│   │   ├── main.py               # App factory, lifespan, middleware, CORS
│   │   ├── schemas.py            # Request/response Pydantic models
│   │   └── routes/               # API route handlers
│   │       ├── query.py          # /api/v1/chat/stream  (SSE streaming)
│   │       ├── health.py         # /api/v1/health
│   │       ├── sessions.py       # /api/v1/sessions
│   │       ├── ingestion.py      # /api/v1/ingest
│   │       └── auth.py           # /api/v1/auth/token
│   │
│   ├── retrieval/                # Hybrid retrieval engine
│   │   ├── controller.py         # RetrievalController — orchestrates all search
│   │   ├── hybrid_search.py      # BM25 + dense vector + RRF fusion
│   │   ├── image_retrieval.py    # CLIP-based image search
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   ├── scoring.py            # Confidence scoring
│   │   ├── context_expander.py   # Adjacent chunk retrieval
│   │   └── query_classifier.py   # Query intent classification
│   │
│   ├── services/                 # Singleton service layer
│   │   ├── embedding.py          # Text embedding (BGE)
│   │   ├── clip_embedding.py     # Image embedding (CLIP)
│   │   ├── bm25_index.py         # BM25 index management
│   │   ├── vector_store.py       # Qdrant client wrapper
│   │   ├── llm_service.py        # Multi-provider LLM abstraction
│   │   ├── chunker.py            # Semantic text chunking
│   │   └── association.py        # Text-image association scoring
│   │
│   ├── ingestion/                # Document ingestion pipeline
│   │   └── pipeline.py           # PDF → extract → chunk → embed → store
│   │
│   ├── orchestration/            # LangGraph workflow
│   │   └── graph.py              # Graph definition and run_agent_workflow()
│   │
│   ├── configs/                  # Configuration
│   │   ├── config.py             # Settings (pydantic-settings)
│   │   └── limiter.py            # Rate limiter (slowapi)
│   │
│   └── utils/                    # Utilities
│       ├── logger.py             # Structured logging setup
│       └── language.py           # Language detection utilities
│
├── frontend/                     # React frontend (Vite + TypeScript)
│   └── src/
│       ├── pages/
│       │   ├── Home.tsx          # Landing page
│       │   └── Chat.tsx          # Chat interface with streaming + lightbox
│       └── components/ui/        # shadcn/ui component library
│
├── data/                         # Data directory (partially gitignored)
│   ├── raw_pdfs/                 # Source PDFs by department
│   │   ├── engineering/
│   │   ├── deck/
│   │   ├── navigation/
│   │   └── safety/
│   ├── extracted_images/         # Extracted diagram images (gitignored)
│   └── metadata/                 # Ingestion manifests and validation reports
│
├── vector_store/                 # Qdrant local storage + BM25 index (gitignored)
├── scripts/                      # Operational utilities
│   ├── ingest.py                 # Run ingestion pipeline on raw_pdfs/
│   ├── download_models.py        # Pre-download all AI models
│   ├── precache.py               # Pre-warm Redis cache with demo queries
│   ├── evaluate.py               # RAG evaluation against benchmark queries
│   ├── staged_validation.py      # End-to-end retrieval validation suite
│   └── generate_corpus_report.py # Generate corpus statistics report
│
├── tests/                        # Test suite (pytest)
├── logs/                         # Application logs (gitignored)
├── docker-compose.yml            # Full stack: backend, frontend, Qdrant, Redis, Phoenix
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── .env.example                  # Environment variable template
```

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.11 |
| Node.js | 18+ | For frontend |
| Docker + Docker Compose | Latest | For infrastructure |
| Ollama | Latest | For local LLM inference |
| RAM | 8 GB+ | 16 GB recommended for CLIP + reranker |

---

### Option A — Local Development (Recommended for Demo)

#### 1. Clone and configure

```bash
git clone https://github.com/your-username/maritimemind-ai.git
cd maritimemind-ai
cp .env.example .env
```

Edit `.env` — set your LLM provider and API key:

```env
# For Gemini (recommended — no local GPU needed):
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here

# For local Ollama:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b
```

#### 2. Start infrastructure (Qdrant + Redis)

```bash
docker-compose up -d vector-db cache
```

#### 3. Set up Python environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

#### 4. Download AI models

```bash
python scripts/download_models.py
```

Downloads: BGE text embedder, CLIP ViT-B-32, MS-Marco cross-encoder reranker (~1.5 GB total).

#### 5. Ingest documents

Place PDF manuals in `data/raw_pdfs/{engineering,deck,navigation,safety}/` then run:

```bash
python scripts/ingest.py
```

This extracts text chunks, extracts and indexes images via CLIP, builds the BM25 index, and stores everything in Qdrant.

#### 6. Start the backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

#### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Application available at: `http://localhost:5173`

---

### Option B — Full Docker Compose

```bash
cp .env.example .env
# Edit .env with your LLM provider and API keys

docker-compose up --build
```

Services started:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Qdrant UI: `http://localhost:6333/dashboard`
- Redis: `localhost:6379`
- Arize Phoenix (observability): `http://localhost:6006`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | Yes | `ollama` | `ollama`, `gemini`, or `openai` |
| `GEMINI_API_KEY` | If Gemini | — | Comma-separated keys for rotation |
| `OPENAI_API_KEY` | If OpenAI | — | OpenAI API key |
| `OLLAMA_BASE_URL` | If Ollama | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | If Ollama | `llama3:8b` | Model name |
| `QDRANT_HOST` | No | `local` | `local` (file-based) or hostname |
| `QDRANT_PATH` | No | `./vector_store/qdrant_local` | Local Qdrant storage path |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `CONFIDENCE_THRESHOLD` | No | `0.6` | Minimum retrieval confidence (0–1) |
| `JWT_SECRET_KEY` | Yes | — | Change in production |
| `VITE_API_URL` | No | `http://localhost:8000` | Frontend → backend URL |

---

## Ingested Document Corpus

The system has been tested with the following maritime document types:

| Category | Examples |
|---|---|
| **Engine Manuals** | Marine diesel engine manuals (489 + 242 chunks), Wärtsilä 26 Maintenance Manual (202 chunks) |
| **Safety** | Engine Room Fires TSC (18 chunks, 17 images) |
| **Deck Operations** | Ballast operation manual, Anchoring guidelines, Mooring manual, Cargo handling |
| **Navigation** | Radar manual (429 chunks, 304 images), ECDIS handbook |
| **Ship Construction** | Ship CON 5 Student Sections (23 diagrams), Construction — Stem (21 diagrams) |
| **Regulatory** | MARPOL Annex I, USCG Marine Safety Manual |
| **Systems** | Ship cooling system (86 chunks), Ship fuel system (65 chunks), Framo ballast manual |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/v1/health` | GET | System health, model status, vector store stats |
| `POST /api/v1/chat/stream` | POST | Main query endpoint — SSE streaming response |
| `GET /api/v1/sessions/{id}` | GET | Retrieve session history |
| `POST /api/v1/ingest` | POST | Trigger document ingestion |
| `POST /api/v1/auth/token` | POST | Obtain JWT access token |

Full interactive API documentation: `http://localhost:8000/docs`

### Streaming Response Format (SSE)

```
data: {"type": "metadata", "data": {"intent": "troubleshooting", "confidence": 0.87, "images": [...], "citations": [...]}}
data: {"type": "token", "data": "The "}
data: {"type": "token", "data": "cylinder ..."}
data: {"type": "done"}
```

---

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/retrieval/
pytest tests/test_vector_store.py -v

# With output
pytest -s -v
```

---

## Scripts Reference

| Script | Description |
|---|---|
| `scripts/ingest.py` | Ingest PDFs from `data/raw_pdfs/` into the vector store |
| `scripts/download_models.py` | Pre-download all required AI model weights |
| `scripts/precache.py` | Pre-warm Redis with demo queries for instant responses |
| `scripts/evaluate.py` | Run RAG evaluation against benchmark queries |
| `scripts/staged_validation.py` | End-to-end retrieval validation across all document categories |
| `scripts/generate_corpus_report.py` | Print corpus statistics (chunk counts, image counts per manual) |
| `scripts/cleanup_images.py` | Remove orphaned extracted images not present in vector store |
| `scripts/backup.py` / `restore.py` | Backup and restore vector store and BM25 index |

---

## Operational Notes

- **Cold start**: First request after startup takes ~10–15s as models load. Subsequent requests are fast.
- **Model pre-warming**: The API automatically pre-warms all models on startup (text embedder, CLIP, BM25, reranker).
- **Qdrant local mode**: By default, Qdrant runs in local file-based mode — no separate service required for development.
- **LLM provider fallback**: Multiple `GEMINI_API_KEY` values can be comma-separated for automatic rate limit rotation.

---

## License

This project is submitted as an academic/professional portfolio demonstration. All maritime technical documents used are publicly available reference materials.
