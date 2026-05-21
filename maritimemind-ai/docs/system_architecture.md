# MaritimeMind AI — System Architecture Document

> **Version**: 1.0.0 | **Status**: Active Development | **Classification**: Open-Source

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Core Design Principles](#3-core-design-principles)
4. [Complete Data Flow](#4-complete-data-flow)
5. [Ingestion Pipeline Architecture](#5-ingestion-pipeline-architecture)
6. [Multimodal Pipeline](#6-multimodal-pipeline)
7. [Hybrid Retrieval Architecture](#7-hybrid-retrieval-architecture)
8. [Metadata Architecture](#8-metadata-architecture)
9. [Agent Orchestration Architecture (Future Phase)](#9-agent-orchestration-architecture-future-phase)
10. [Storage Architecture](#10-storage-architecture)
11. [Evaluation Architecture](#11-evaluation-architecture)
12. [Logging & Monitoring Architecture](#12-logging--monitoring-architecture)
13. [Future Scalability](#13-future-scalability)
14. [Project Folder Structure](#14-project-folder-structure)
15. [Security & Reliability Considerations](#15-security--reliability-considerations)

---

## 1. System Overview

### 1.1 Purpose

**MaritimeMind AI** is an offline-capable, multimodal maritime intelligence platform designed to serve as a contextual knowledge assistant aboard vessels and within maritime operations centers. The system allows engineers, officers, and technical personnel to query complex maritime manuals — returning precise technical explanations, step-by-step procedures, troubleshooting guidance, and relevant engineering schematics.

The platform is engineered around the principle that **critical maritime information must remain accessible regardless of network connectivity**, transforming static PDF manuals into a living, queryable knowledge graph.

### 1.2 Multimodal Maritime Intelligence

Maritime technical documentation is inherently multimodal. Engine maintenance manuals contain:
- Dense procedural text and specifications
- Complex engineering schematics and system diagrams
- Tables of parameters, tolerances, and fault codes
- Annotated component diagrams with part numbers

MaritimeMind AI treats text, tables, and images as **first-class retrieval objects**, linking them by spatial proximity and semantic reference. A query for `"cooling pump failure procedure"` can return the relevant procedural text _and_ the associated piping schematic — simultaneously.

### 1.3 Offline-First Architecture Philosophy

The system is architected for **ship deployment constraints**, where satellite internet may be expensive, intermittent, or unavailable:

- All LLM inference runs locally via **Ollama** (e.g. `llama3:8b`)
- All embeddings are computed locally via **SentenceTransformers** and **OpenCLIP**
- All vector storage is managed locally via **ChromaDB** (embedded mode)
- No external API calls are required during operational usage
- The system can be fully initialized, indexed, and queried from an offline workstation or server

---

## 2. High-Level Architecture

### 2.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MaritimeMind AI                             │
│                     System Architecture v1.0                        │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                         Frontend Layer                           │
  │                    Streamlit Web Interface                        │
  │       (Query Entry | Results Display | Image Viewer)             │
  └────────────────────────────┬─────────────────────────────────────┘
                               │ HTTP (FastAPI)
  ┌────────────────────────────▼─────────────────────────────────────┐
  │                          API Layer                               │
  │                     FastAPI REST Backend                         │
  │        (Query Routing | Response Formatting | Sessions)          │
  └──────────┬──────────────────────────────────────┬───────────────┘
             │                                      │
  ┌──────────▼──────────┐             ┌─────────────▼───────────────┐
  │  Orchestration Layer│             │       Ingestion Layer        │
  │  LangGraph Agents   │             │   PDF Parsing / Chunking /   │
  │  (Future Phase)     │             │   Embedding / Manifest       │
  └──────────┬──────────┘             └─────────────┬───────────────┘
             │                                      │
  ┌──────────▼──────────────────────────────────────▼───────────────┐
  │                       Retrieval Layer                            │
  │     BM25 Keyword Search │ ChromaDB Vector Search │ Reranker      │
  └──────────────────────────────────────┬───────────────────────────┘
                                         │
  ┌──────────────────────────────────────▼───────────────────────────┐
  │                     Vector Storage Layer                         │
  │     ChromaDB (Text Collection) │ ChromaDB (Image Collection)     │
  │     BM25 Sparse Index          │ Ingestion Manifest JSON         │
  └──────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Separation

| Layer | Responsibility | Technology |
|---|---|---|
| **Frontend** | User interaction, query entry, result display | Streamlit |
| **API** | Request handling, session management, response formatting | FastAPI |
| **Orchestration** | Query routing, agent memory, iterative reasoning (future) | LangGraph |
| **Ingestion** | PDF parsing, chunking, embedding, storage | PyMuPDF, pdfplumber, SentenceTransformers, OpenCLIP |
| **Retrieval** | Hybrid search, reranking, scoring | ChromaDB, BM25, CrossEncoder |
| **Storage** | Persistent vectors, images, metadata | ChromaDB, Local Filesystem |
| **LLM Inference** | Response synthesis, summarization | Ollama (llama3:8b) |

---

## 3. Core Design Principles

### 3.1 Modularity
Every component is designed as an independently replaceable service. The embedding system, chunking strategy, vector store, and LLM backend can each be swapped without breaking adjacent services. This is enforced by typed service interfaces defined in Pydantic schemas.

### 3.2 Separation of Concerns
Each file and class has a single, clear responsibility:
- **Config** manages environment
- **Schemas** enforce data contracts
- **Services** implement pipeline logic
- **Retrieval** handles query resolution
- **Evaluation** measures system quality

### 3.3 Scalability
The system scales both horizontally (multiple concurrent ingestion workers) and vertically (GPU-accelerated embedding). ChromaDB's embedded mode can be swapped for a server-mode deployment or cloud vector database without altering the retrieval interface.

### 3.4 Retrieval Grounding
All responses are anchored to source documents. Chunk IDs, page numbers, and manual references are always preserved in the response envelope, enabling citation tracking and hallucination prevention.

### 3.5 Multimodal Association
Text chunks maintain explicit references to related images (`related_images`), and images maintain references to nearby or citing text chunks (`linked_chunks`). This bidirectional association enables cross-modal retrieval — retrieving a diagram from a text query and vice versa.

### 3.6 Local-First AI Inference
All language model inference is performed locally via Ollama, ensuring:
- Zero data leaves the vessel
- Zero subscription costs
- Zero latency from remote API calls
- Compliance with maritime data privacy requirements

### 3.7 Maintainability
All files are constrained to 300–400 lines maximum. Docstrings, type hints, and a centralized configuration system ensure the codebase remains readable and extensible by future development teams.

---

## 4. Complete Data Flow

### 4.1 End-to-End Pipeline

```
  ┌──────────────┐
  │  PDF Manual  │  (e.g. engine_room_manual.pdf)
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │  INGESTION LAYER                                     │
  │                                                      │
  │  1. PDF Parser (PyMuPDF + pdfplumber)                │
  │     ├── Extract text blocks with layout info         │
  │     ├── Extract tables → Markdown format             │
  │     └── Extract images → file system + bounding box │
  │                                                      │
  │  2. Layout-Aware Chunker                             │
  │     ├── Detect headings & section hierarchy          │
  │     ├── Build hierarchy_path per chunk               │
  │     ├── Link previous_chunk_id / next_chunk_id       │
  │     └── Bind related images to text chunks           │
  │                                                      │
  │  3. Metadata Resolver                                │
  │     ├── Generate chunk_id (SHA-256)                  │
  │     ├── Associate images ↔ chunks                   │
  │     └── Write ingestion_manifest.json entry          │
  └───────────────────────────────────┬─────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  EMBEDDING LAYER                                     │
  │                                                      │
  │  Text Chunks  → SentenceTransformers                 │
  │  Image Files  → OpenCLIP ViT-B-32 Encoder            │
  └───────────────────────────────────┬─────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  STORAGE LAYER                                       │
  │                                                      │
  │  Text Embeddings → ChromaDB (text_collection)        │
  │  Image Embeddings → ChromaDB (image_collection)      │
  │  Text Corpus → BM25 Sparse Index                     │
  │  File State → ingestion_manifest.json                │
  └───────────────────────────────────┬─────────────────┘
                                      │
                ┌─────────────────────┘
                │  (On Query)
                ▼
  ┌──────────────────────────────────────────────────────┐
  │  RETRIEVAL LAYER                                     │
  │                                                      │
  │  1. Query Classification                             │
  │     └── Intent: EXPLANATION / PROCEDURE / DIAGRAM   │
  │                                                      │
  │  2. Hybrid Search                                    │
  │     ├── BM25 (sparse keyword)                        │
  │     ├── ChromaDB (dense vector)                      │
  │     └── Reciprocal Rank Fusion (RRF)                 │
  │                                                      │
  │  3. Cross-Encoder Reranking                          │
  │     └── MiniLM cross-encoder scores top candidates  │
  │                                                      │
  │  4. Confidence Scoring                               │
  │     └── bm25_score / vector_score / rerank_score     │
  └───────────────────────────────────┬─────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  SYNTHESIS LAYER                                     │
  │                                                      │
  │  Ollama (llama3:8b) - Local LLM                      │
  │  ├── RAG prompt with retrieved context               │
  │  ├── Citation-grounded generation                    │
  │  └── Response + source references                    │
  └──────────────────────────────────────────────────────┘
```

---

## 5. Ingestion Pipeline Architecture

### 5.1 PDF Parsing Flow

The ingestion pipeline begins with `pdf_parser.py`, which processes each PDF using **two complementary libraries**:

- **PyMuPDF (`fitz`)**: Provides high-fidelity layout analysis — character bounding boxes, font weights, page geometry, and raw raster images. This is the primary extraction engine.
- **pdfplumber**: Applied exclusively to pages that PyMuPDF flags as containing tabular data structures. pdfplumber's cell boundary detection is superior for complex multi-column maritime tables, which it converts to Markdown representations.

> **Why two libraries?** Maritime manuals are heterogeneous. Engine room diagrams are best handled by PyMuPDF's pixel-level control, while specification tables require pdfplumber's cell-aware parsing. Using both libraries in a complementary, non-overlapping strategy maximizes extraction quality without redundancy.

### 5.2 Image Extraction

Images are extracted using PyMuPDF's `get_images()` method per page. For every image:
- A unique `image_id` (SHA-256 of raw bytes) is generated to prevent re-processing duplicates
- The image is written to `data/extracted_images/` in PNG format
- The precise **bounding box** (`x0, y0, x1, y1`) on the source page is recorded
- A caption search is performed on nearby text blocks using spatial proximity rules
- Small or decorative images (below `MIN_IMAGE_WIDTH` × `MIN_IMAGE_HEIGHT`) are filtered out

### 5.3 Layout-Aware Chunking Strategy

```
PAGE STRUCTURE ANALYSIS
│
├── Font size ≥ H1 threshold → Chapter heading
│     └── New hierarchy node: ["Chapter 3"]
│
├── Font size ≥ H2 threshold → Section heading
│     └── New hierarchy node: ["Chapter 3", "Section 3.2"]
│
├── Font size ≥ H3 threshold → Subsection heading
│     └── New hierarchy node: ["Chapter 3", "Section 3.2", "3.2.1"]
│
└── Body text → Accumulated into current chunk
      ├── If chunk exceeds CHUNK_SIZE → split with CHUNK_OVERLAP
      ├── Tables → always kept whole (never split)
      └── Numbered procedures → kept whole (never split mid-step)
```

> **Design Rationale**: Purely token-based splitting destroys procedural integrity. If a 10-step maintenance procedure is split at step 6, the retrieved chunk becomes dangerous — it may omit critical safety steps. Layout-aware chunking guarantees procedures are either fully included or fully excluded from any given chunk.

### 5.4 Association Engine

After chunking, an association pass links `ImageMetadata.linked_chunks` to `TextChunk.related_images` using two rules:

1. **Spatial Proximity Rule**: Any image on page N is associated with all chunks originating from page N.
2. **Textual Reference Rule**: Regex patterns scan chunk content for patterns like `"Figure 3"`, `"Diagram A"`, `"see schematic 7-B"`. Matched images are explicitly linked regardless of page.

### 5.5 OCR Placeholder Architecture

An `OcrService` interface is defined now, even though Tesseract is not yet integrated. Its contract:
```
OcrService.extract(image_path: str) → str
```
This stub returns an empty string today but is structurally wired to `ImageMetadata.ocr_text`. When Tesseract OCR or a Vision LLM is integrated in a future phase, zero changes are required downstream — only the service implementation changes.

---

## 6. Multimodal Pipeline

### 6.1 Text Embeddings (SentenceTransformers)

Text chunks are vectorized using **`all-MiniLM-L6-v2`** — a lightweight, high-performance sentence embedding model:
- 384-dimensional dense vectors
- Optimized for semantic similarity
- Runs efficiently on CPU (important for shipboard servers)
- Fully offline, no API calls

### 6.2 Image Embeddings (OpenCLIP)

Engineering diagrams are vectorized using the **OpenCLIP ViT-B-32** model with `laion2b_s34b_b79k` pretrained weights:
- 512-dimensional image embedding space
- Jointly trained with text — enabling cross-modal retrieval
- The image encoder processes `PIL.Image` objects extracted from PDFs
- Stored in a separate ChromaDB collection (`image_collection`)

### 6.3 Cross-Modal Retrieval Architecture

```
  User Query: "cooling pump diagram"
       │
       ▼
  Text Encoder (SentenceTransformers)
       │ Query vector (384-dim)
       ├──────────────────────────────────────────────────┐
       │                                                  │
       ▼                                                  ▼
  ChromaDB Text Search                    ChromaDB Image Search
  (384-dim cosine similarity)             (512-dim cosine similarity)
       │                                                  │
       ▼                                                  ▼
  Top-K Text Chunks                       Top-K Image Results
  with related_images[]                   with linked_chunks[]
       │                                                  │
       └─────────────────┬────────────────────────────────┘
                         │
                         ▼
              Fused Multimodal Result Set
              (Text + Diagrams + Citations)
```

### 6.4 Image-Text Bidirectional Linking

Every `TextChunk` contains a `related_images: List[str]` field referencing `image_id` values. Every `ImageMetadata` contains a `linked_chunks: List[str]` field referencing `chunk_id` values. This bidirectional design enables:

- **Text → Image**: Retrieve text, expand to related schematics
- **Image → Text**: Retrieve diagram, trace back to procedure context
- **Query → Both**: Simultaneous multimodal result assembly

---

## 7. Hybrid Retrieval Architecture

### 7.1 Retrieval Pipeline Overview

```
  User Query
       │
       ▼
  ┌─────────────────────────────┐
  │  Query Classifier           │
  │  EXPLANATION / PROCEDURE /  │
  │  TROUBLESHOOTING / DIAGRAM  │
  └──────────────┬──────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
  BM25 Search        ChromaDB Vector Search
  (Sparse)           (Dense)
  Top-K results      Top-K results
  + bm25_score       + vector_score
       │                    │
       └─────────┬──────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Reciprocal Rank Fusion     │
  │  (RRF)                      │
  │  score = Σ 1/(k + rank_i)   │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Cross-Encoder Reranker     │
  │  ms-marco-MiniLM-L-6-v2    │
  │  → rerank_score             │
  └──────────────┬──────────────┘
                 │
                 ▼
  Top-K RetrievalResult objects
  (with full score profiles)
```

### 7.2 BM25 Keyword Retrieval

BM25 (Best Matching 25) operates on the raw text corpus of all chunks stored as a sparse inverted index (`rank-bm25`). It excels at maritime-specific terminology:

- Part numbers (e.g. `MAN B&W 6S60MC-C`)
- Technical codes (e.g. `ISO 8217`, `MARPOL Annex VI`)
- Procedure identifiers (e.g. `PRO-ENG-0042`)

BM25 retrieves results where dense semantic search may assign low similarity due to technical jargon not well-represented in the embedding space.

### 7.3 Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from BM25 and ChromaDB without requiring score normalization:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k = 60` (standard constant), and `rank_i(d)` is the position of document `d` in retrieval list `i`. Results appearing high in both lists are amplified; results appearing in only one list are naturally discounted.

### 7.4 Cross-Encoder Reranking

After RRF fusion, the top-N candidates are passed through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoders (where query and document are embedded independently), the cross-encoder performs full attention across both, producing more accurate relevance judgements at the cost of slightly higher latency.

### 7.5 Confidence Scoring

Every `RetrievalResult` exposes a `RetrievalScores` object:

| Score | Source | Usage |
|---|---|---|
| `bm25_score` | BM25 rank output | Keyword match strength |
| `vector_score` | ChromaDB cosine distance | Semantic similarity |
| `rerank_score` | Cross-encoder output | Final relevance judgement |
| `final_score` | RRF fusion | Combined ranking signal |
| `confidence_score` | Normalized 0–1 | Threshold gating for agents |

Low `confidence_score` values can trigger automatic retrieval retries in the LangGraph orchestration layer, preventing weak answers from reaching the user.

---

## 8. Metadata Architecture

### 8.1 TextChunk Schema

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | `str` | SHA-256 hash of content + manual context. Stable across re-ingestion. |
| `manual_name` | `str` | Source PDF filename |
| `department` | `str` | Operational context (e.g. `engine`, `navigation`, `safety`) |
| `page_number` | `int` | Source page (1-indexed) for citation display |
| `section_title` | `str` | Immediate parent section heading |
| `content` | `str` | Text content, including Markdown-formatted tables |
| `keywords` | `List[str]` | Auto-extracted or manually assigned technical terms |
| `related_images` | `List[str]` | `image_id` values of spatially or referentially linked diagrams |
| `hierarchy_path` | `List[str]` | Full document tree path (e.g. `["Chapter 3", "Section 3.2", "3.2.1"]`) |
| `previous_chunk_id` | `Optional[str]` | Sequential predecessor (enables "show previous step") |
| `next_chunk_id` | `Optional[str]` | Sequential successor (enables "continue procedure") |
| `embedding_model` | `str` | Embedding model used (for future version tracking) |
| `created_at` | `str` | ISO 8601 UTC timestamp of ingestion |

### 8.2 ImageMetadata Schema

| Field | Type | Purpose |
|---|---|---|
| `image_id` | `str` | SHA-256 hash of raw image bytes |
| `manual_name` | `str` | Source PDF filename |
| `page_number` | `int` | Source page |
| `image_path` | `str` | Filesystem path to the extracted PNG |
| `caption` | `str` | Nearby text identified as figure caption |
| `bbox` | `BoundingBox` | `{x0, y0, x1, y1}` in page units for visual grounding |
| `linked_chunks` | `List[str]` | Reverse-linked `chunk_id` values |
| `ocr_text` | `Optional[str]` | Future: extracted label text from diagram |
| `embedding_model` | `str` | CLIP model used for vectorization |
| `created_at` | `str` | ISO 8601 UTC timestamp |

### 8.3 RetrievalResult Schema

The output of the retrieval layer is a structured `RetrievalResult`, which contains:
- Full text chunk content
- Source reference (manual, page, section)
- Complete document hierarchy path for navigation
- References to linked image files
- A `RetrievalScores` sub-object with all scoring components

### 8.4 Hierarchy Tracking & Navigation

The `hierarchy_path` and `previous_chunk_id` / `next_chunk_id` fields collectively form a **doubly-linked hierarchical graph** embedded inside the flat vector store. This enables future agents to:
- Navigate to the parent section for broader context
- Walk forward or backward through procedural steps
- Reconstruct the full procedure from any entry point

---

## 9. Agent Orchestration Architecture (Future Phase)

### 9.1 LangGraph Multi-Agent Design

```
  User Query
       │
       ▼
  ┌──────────────────────────────────┐
  │        QUERY ROUTER AGENT        │
  │  Classifies intent:              │
  │  EXPLANATION / PROCEDURE /       │
  │  TROUBLESHOOTING / EMERGENCY /   │
  │  DIAGRAM_REQUEST                 │
  └──────────────┬───────────────────┘
                 │
       ┌─────────┴──────────┬──────────────────────────┐
       │                    │                          │
       ▼                    ▼                          ▼
  TEXT RETRIEVAL      VISUAL RETRIEVAL          SAFETY AGENT
  AGENT               AGENT                    (Emergency
  (Hybrid Search +    (OpenCLIP Image           Protocol
  Reranking)          Search + Diagram          Injection)
       │              Expansion)                    │
       └──────────────┬──────────────────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ RESPONSE         │
             │ SYNTHESIZER      │
             │ (Ollama LLM +    │
             │ RAG Prompt)      │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ QUALITY REVIEWER │
             │ Confidence gate: │
             │ if score < 0.6:  │
             │ → retry retrieval│
             └────────┬─────────┘
                      │
                      ▼
               Final Response
```

### 9.2 Agent Responsibilities

| Agent | Responsibility | Trigger |
|---|---|---|
| **Query Router** | Classifies intent and selects retrieval strategy | Every query |
| **Text Retrieval Agent** | Executes hybrid BM25 + vector search | Text-oriented intents |
| **Visual Retrieval Agent** | Executes image collection search and diagram expansion | `DIAGRAM_REQUEST` intent |
| **Safety Agent** | Injects emergency protocols for critical fault queries | `EMERGENCY` intent |
| **Response Synthesizer** | Constructs grounded, cited LLM response | After retrieval |
| **Quality Reviewer** | Validates response confidence; triggers retry loops | Always |

---

## 10. Storage Architecture

### 10.1 ChromaDB Collections

Two persistent ChromaDB collections are maintained:

**`maritime_text_chunks`**
- Distance metric: Cosine
- Embedding dimension: 384 (SentenceTransformers)
- Document: `TextChunk.content`
- Metadata: `manual_name`, `page_number`, `section_title`, `hierarchy_path`, `related_images`

**`maritime_image_chunks`**
- Distance metric: Cosine
- Embedding dimension: 512 (OpenCLIP ViT-B-32)
- Document: Caption text (or OCR placeholder)
- Metadata: `manual_name`, `page_number`, `image_path`, `linked_chunks`, `bbox`

### 10.2 BM25 Sparse Index

The BM25 index (`rank-bm25`) is built from the tokenized `content` field of all `TextChunk` objects. It is serialized to disk alongside the ChromaDB persistence directory using Python's `pickle` module, enabling fast reload without re-indexing.

### 10.3 Ingestion Manifest

`data/metadata/ingestion_manifest.json` maintains a per-file processing ledger:

```json
{
  "engine_room_manual.pdf": {
    "status": "COMPLETED",
    "processed_date": "2025-05-21T17:00:00Z",
    "chunk_count": 342,
    "image_count": 87,
    "errors": []
  }
}
```

This manifest prevents duplicate ingestion, enables resumable processing after failures, and provides operational visibility into the knowledge base composition.

### 10.4 Local Storage Strategy

```
data/
├── raw_pdfs/           # Input PDFs (source of truth)
├── extracted_text/     # Intermediate raw text blocks (debugging)
├── extracted_images/   # Extracted PNG schematics
│   └── {manual_name}/
│       └── {image_id}.png
├── processed_chunks/   # Serialized TextChunk objects (optional cache)
└── metadata/
    └── ingestion_manifest.json

vector_store/
└── chromadb/           # ChromaDB embedded persistent storage
    ├── maritime_text_chunks/
    └── maritime_image_chunks/
```

---

## 11. Evaluation Architecture

### 11.1 Benchmark Dataset

A structured JSON benchmark dataset (`app/evaluation/benchmark_queries.json`) drives automated evaluation:

```json
[
  {
    "query_id": "Q001",
    "query": "show cooling pump diagram",
    "intent": "DIAGRAM_REQUEST",
    "expected_manual": "engine_room_manual.pdf",
    "expected_page": 17,
    "expected_chunk_ids": ["abc123", "def456"],
    "expected_image_id": "img_017_a"
  }
]
```

### 11.2 Retrieval Evaluation Metrics

| Metric | Description |
|---|---|
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of relevant documents found in top-K |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank for first relevant result |
| **MAP (Mean Average Precision)** | Mean of precision scores at each relevant result |
| **NDCG@K** | Normalized Discounted Cumulative Gain for ranked relevance |

### 11.3 Image Retrieval Evaluation

The `image_retrieval_metrics.py` module evaluates:
- Diagram-to-text match accuracy
- Cross-modal retrieval precision (text query → correct image)
- Caption-embedding alignment scores

### 11.4 Grounding Accuracy & Hallucination Prevention

Every response includes a source citation envelope (manual name, page number, chunk ID). The evaluation pipeline validates:
- Whether the LLM response content can be traced to a specific retrieved chunk
- Whether no new technical facts are introduced beyond the retrieved context
- Whether `confidence_score` correlates with factual accuracy across the benchmark

---

## 12. Logging & Monitoring Architecture

### 12.1 Centralized Logging

All modules instantiate loggers via a shared factory:

```python
from app.utils.logger import setup_logger
logger = setup_logger(__name__)
```

This ensures consistent formatting, rotation, and file routing across the entire system.

### 12.2 Log Structure

```
logs/
└── maritimemind.log   # Rotated at 10MB, 30 backups retained
```

Log format:
```
2025-05-21 17:00:00 - app.services.pdf_parser - INFO - Processing: engine_room_manual.pdf
2025-05-21 17:01:23 - app.services.chunker - INFO - Generated 342 chunks
2025-05-21 17:01:45 - app.services.embedding - INFO - Embedded 342 text chunks [model: all-MiniLM-L6-v2]
2025-05-21 17:02:11 - app.services.vector_store - ERROR - Failed to write chunk abc123: [reason]
```

### 12.3 Monitoring Dimensions

| Area | What is Logged |
|---|---|
| **Ingestion** | File start/complete, chunk count, image count, errors |
| **Extraction** | Pages processed, tables found, images extracted, filtered |
| **Embedding** | Model name, batch sizes, processing time |
| **Retrieval** | Query text, intent classification, top-K scores, latency |
| **Synthesis** | LLM prompt length, response tokens, total latency |
| **Failures** | Full exception traces, file names, recovery actions |

---

## 13. Future Scalability

### 13.1 OCR Integration (Phase 2)
Tesseract OCR or a Vision LLM (e.g., `LLaVA` via Ollama) will populate `ImageMetadata.ocr_text` with label text extracted from engineering diagrams — dramatically improving retrieval accuracy for queries involving valve identifiers, pipe codes, and component labels.

### 13.2 Multilingual Retrieval
`paraphrase-multilingual-MiniLM-L12-v2` can replace the default text embedding model to support Spanish, Norwegian, and other maritime industry languages without re-architecting any pipeline components.

### 13.3 GPU Acceleration
The `DEVICE` configuration field in `config.py` is the sole change required to route all tensor operations to a CUDA-capable GPU, enabling 10x+ embedding throughput for large-scale ingestion.

### 13.4 Neo4j Knowledge Graph
The hierarchical chunk structure (`hierarchy_path`, `previous_chunk_id`, `next_chunk_id`) is architected to project directly into a Neo4j graph database, enabling graph-based traversal queries — ideal for complex fault tracing across interconnected maritime systems.

### 13.5 Ship-Specific Personalization
A per-vessel configuration layer (fleet codes, equipment serial numbers, maintenance histories) can be injected at query time to filter and prioritize manual sections relevant to the specific vessel's installed equipment.

### 13.6 Voice Interface
The FastAPI backend is stateless and can be fronted with a speech-to-text preprocessing layer (e.g. `whisper.cpp`) to enable voice queries from engine room personnel without keyboard access.

### 13.7 Anomaly Diagnosis Workflows
The LangGraph orchestration layer is designed to support multi-step diagnostic workflows where the agent iteratively queries different sections of different manuals and synthesizes a fault tree — critical for real-time machinery troubleshooting.

---

## 14. Project Folder Structure

```
maritimemind-ai/
│
├── app/                          # Core application source code
│   ├── __init__.py
│   ├── configs/
│   │   ├── __init__.py
│   │   └── config.py             # Centralized pydantic-settings config
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # All Pydantic data models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # PDF layout parsing & asset extraction
│   │   ├── ocr.py                # OCR service stub/interface
│   │   ├── chunker.py            # Hierarchical, layout-aware chunking
│   │   ├── embedding.py          # SentenceTransformer + OpenCLIP wrappers
│   │   ├── vector_store.py       # ChromaDB + BM25 persistence layer
│   │   └── retrieval.py          # Hybrid search + reranking controller
│   │
│   ├── agents/                   # [Future] LangGraph agent definitions
│   ├── orchestration/            # [Future] LangGraph graph definitions
│   ├── memory/                   # [Future] Conversation state management
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py             # Centralized logging factory
│   │
│   ├── api/                      # FastAPI route definitions
│   ├── ui/                       # Streamlit pages
│   └── evaluation/
│       ├── __init__.py
│       ├── benchmark_queries.json
│       ├── retrieval_metrics.py
│       ├── image_retrieval_metrics.py
│       └── evaluation_runner.py
│
├── data/
│   ├── raw_pdfs/                 # Source PDF manuals
│   ├── extracted_text/           # Intermediate extraction output
│   ├── extracted_images/         # Exported schematic PNGs
│   ├── processed_chunks/         # Serialized chunk cache
│   └── metadata/
│       └── ingestion_manifest.json
│
├── vector_store/
│   └── chromadb/                 # ChromaDB embedded persistence
│
├── logs/
│   └── maritimemind.log          # Rotating application logs
│
├── notebooks/                    # Jupyter exploration notebooks
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_schemas.py
│   └── test_evaluation.py
│
├── configs/
│   └── deployment.yaml           # [Future] Docker/server deployment config
│
├── scripts/
│   ├── ingest.py                 # CLI ingestion entry point
│   └── evaluate.py              # CLI evaluation runner
│
├── docs/
│   └── system_architecture.md   # This document
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 15. Security & Reliability Considerations

### 15.1 Local-First Privacy
All data — PDFs, extracted text, images, embeddings, and LLM inference — remain on the local machine or vessel server. No third-party APIs are called during operation. This complies with maritime data protection requirements and eliminates exposure of proprietary technical manuals.

### 15.2 Offline Operation
The system is designed to operate with zero network access after initial setup. Model weights (SentenceTransformers, OpenCLIP, CrossEncoder, Ollama) are downloaded once and cached locally. ChromaDB runs in embedded mode requiring no server process.

### 15.3 Hallucination Mitigation

MaritimeMind AI applies three layers of hallucination prevention:

1. **Retrieval Grounding**: The LLM receives only retrieved context — it cannot introduce facts not present in the manual.
2. **Citation Tracking**: Every `RetrievalResult` includes source references. The Streamlit UI displays citations alongside every response.
3. **Confidence Thresholds**: Low-confidence retrievals (below configurable `confidence_score` thresholds) suppress LLM synthesis and return `"Insufficient information found"` responses.

### 15.4 Fail-Safe Design

The ingestion manifest (`ingestion_manifest.json`) tracks processing state per file. If ingestion fails mid-document, only completed pages are committed to the vector store. Re-running the ingestion pipeline detects and resumes from the last verified state — preventing corrupted or partial knowledge bases.

### 15.5 Citation Tracking

Every `TextChunk` carries its `manual_name`, `page_number`, and `hierarchy_path`. Every `RetrievalResult` preserves and surfaces these fields. The response synthesis prompt instructs the LLM to reference specific sections — creating an auditable, traceable evidence chain from answer back to source manual page.

---

*Document generated for MaritimeMind AI v1.0.0 — Open-Source Maritime Intelligence Platform*
