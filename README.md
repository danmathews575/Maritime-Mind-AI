# MaritimeMind AI

## Agentic Multimodal Maritime Intelligence Platform for Shipboard Operations and Engineering Support

MaritimeMind AI is an AI-powered maritime intelligence platform designed to assist marine engineers, deck cadets, ship operators, and safety personnel by providing contextual technical explanations, operational procedures, troubleshooting guidance, and relevant visual diagrams directly from maritime manuals and shipboard documentation.

Unlike traditional chatbots that only retrieve text, MaritimeMind AI combines **Hybrid Retrieval**, **Cross-Modal Image Search**, and **Multi-Agent Orchestration** to deliver a complete shipboard knowledge experience with both textual and visual intelligence.

---

# Problem Statement

Modern ships operate as highly complex engineering ecosystems containing:

* Marine propulsion systems
* Fuel and cooling systems
* Electrical networks
* Navigation systems
* Cargo and ballast operations
* Safety and emergency procedures

Crew members and cadets often rely on hundreds of pages of technical manuals, troubleshooting guides, and operational documentation to diagnose issues and understand procedures.

Traditional manual searching is:

* Time-consuming
* Difficult during emergencies
* Diagram-heavy
* Operationally inefficient

Existing AI chatbots primarily retrieve text and fail to provide contextual engineering visuals such as:

* Wiring diagrams
* Piping schematics
* Flow charts
* Engine layouts
* Safety maps

MaritimeMind AI addresses this challenge through an intelligent multimodal retrieval architecture capable of retrieving both technical knowledge and relevant visual references from maritime documentation.

---

# Project Objectives

* Build an AI-powered maritime knowledge intelligence platform
* Support ship-wide operational workflows and engineering systems
* Implement Hybrid Retrieval (Semantic + Keyword Search)
* Implement Cross-Modal Image Retrieval
* Build a Multi-Agent LangGraph workflow
* Provide contextual visual grounding for maritime procedures
* Support deck cadets and marine engineers with procedural learning
* Reduce manual search complexity in shipboard environments

---

# Core Features

## Hybrid Maritime Retrieval

Combines:

* Vector Search (semantic understanding)
* BM25 Keyword Search (exact technical matching)

Supports:

* Alarm codes
* Technical terminology
* Operational procedures
* Safety instructions

---

## Multimodal Diagram Retrieval

Retrieves:

* Wiring diagrams
* Piping schematics
* Ballast layouts
* Engine cross-sections
* Emergency evacuation maps
* Navigation system visuals

The system automatically displays relevant diagrams alongside technical explanations.

---

## Multi-Agent Architecture

Built using LangGraph with specialized AI agents.

### Planned Agents

| Agent                      | Responsibility                            |
| -------------------------- | ----------------------------------------- |
| Intent Router Agent        | Determines query domain and routing       |
| Retrieval Agent            | Performs hybrid document retrieval        |
| Visual Intelligence Agent  | Retrieves relevant diagrams/images        |
| Maritime Operations Agent  | Handles operational workflows             |
| Safety Compliance Agent    | Handles emergency and compliance queries  |
| Response Synthesizer Agent | Generates grounded final responses        |
| Quality Verification Agent | Validates retrieval quality and grounding |

---

## Stateful Maritime Conversations

Supports conversational memory.

Example:

* User asks about a cooling system
* Follow-up queries can reference previous diagrams and components contextually

---

## Cross-Department Intelligence

Supports multiple shipboard domains:

### Engineering

* Marine diesel engines
* Cooling systems
* Fuel systems
* Pumps
* Electrical systems

### Deck Operations

* Ballast systems
* Cargo handling
* Mooring procedures
* Anchoring workflows

### Navigation

* Radar systems
* ECDIS procedures
* Bridge operations

### Safety & Emergency

* Engine room fire response
* Evacuation procedures
* SOLAS workflows
* Safety compliance

---

# System Architecture

```plaintext
User Query
    ↓
Intent Router Agent
    ↓
Hybrid Retrieval Pipeline
    ├── Semantic Vector Search
    ├── BM25 Keyword Search
    └── Metadata Filtering
    ↓
Visual Intelligence Agent
    ↓
Response Synthesizer
    ↓
Quality Verification Agent
    ↓
Final Multimodal Maritime Response
```

---

# Tech Stack

## Backend

* Python
* FastAPI

## AI Orchestration

* LangGraph
* LangChain

## LLMs

* Ollama (Local LLM Runtime)
* DeepSeek / Llama / Qwen

## Vector Database

* ChromaDB

## Embeddings

* Sentence Transformers
* BGE Embeddings
* OpenCLIP (Image Embeddings)

## Retrieval

* Hybrid Search
* BM25
* Semantic Search

## PDF Processing

* PyMuPDF
* pdfplumber

## Frontend

* Streamlit

---

# Dataset Categories

The platform uses maritime manuals and operational documents including:

## Engineering Manuals

* Wärtsilä maintenance manuals
* Marine diesel engine manuals
* Cooling system documentation
* Fuel system documentation
* Pump maintenance guides

## Deck Operations

* Ballast operation manuals
* Cargo handling procedures
* Mooring operation guides
* Anchoring procedures

## Navigation Systems

* Radar manuals
* ECDIS training handbooks

## Safety & Emergency

* Engine room fire procedures
* Ship evacuation guidelines
* SOLAS operational procedures
* Maritime safety documentation

---

# Example Use Cases

## Engineering Troubleshooting

**Query:**
“Why is cooling water pressure low?”

**System Response:**

* Retrieves troubleshooting procedures
* Displays cooling pipeline schematic
* Explains probable causes
* Suggests inspection sequence

---

## Deck Cadet Training

**Query:**
“Explain ballast tank operation.”

**System Response:**

* Explains ballast workflow
* Retrieves ballast system diagram
* Describes stability management process

---

## Emergency Assistance

**Query:**
“What is the engine room fire response procedure?”

**System Response:**

* Retrieves emergency SOP
* Displays evacuation/safety layout
* Explains emergency sequence

---

# Planned Future Enhancements

* Visual component highlighting
* Voice-based shipboard assistant
* Real-time onboard document indexing
* OCR for scanned maritime manuals
* Crew-specific role personalization
* AR-assisted engineering support
* Predictive maintenance intelligence

---

# Project Structure

```plaintext
maritime-ai/
│
├── app/
│   ├── agents/
│   ├── retrieval/
│   ├── ingestion/
│   ├── memory/
│   ├── ui/
│   └── utils/
│
├── data/
│   ├── engineering/
│   ├── deck_operations/
│   ├── safety/
│   ├── navigation/
│   ├── emergency/
│   ├── images/
│   └── metadata/
│
├── vector_store/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

# Key Innovations

* Agentic maritime workflow orchestration
* Multimodal maritime retrieval
* Hybrid semantic + keyword retrieval
* Cross-modal text-to-image search
* Stateful maritime intelligence
* Diagram-grounded technical explanations

---

# Vision

MaritimeMind AI aims to become an intelligent shipboard copilot capable of assisting marine engineers, deck cadets, and maritime operators with contextual operational intelligence, technical troubleshooting, and multimodal maritime knowledge retrieval.

---

# Status

Current Phase:

* Dataset Collection & Architecture Design

Upcoming:

* PDF ingestion pipeline
* Multimodal embedding generation
* Hybrid retrieval implementation
* Agent workflow orchestration
* Streamlit-based UI development
