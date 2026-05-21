# MaritimeMind AI

MaritimeMind AI is an advanced AI-powered decision support and retrieval-augmented generation (RAG) system designed for the maritime domain.

## Project Structure

```text
maritimemind-ai/
├── app/                  # Main application source code
│   ├── agents/          # LLM Agents and planning strategies
│   ├── retrieval/       # Search and retrieval mechanics
│   ├── ingestion/       # PDF and document parsing pipelines
│   ├── memory/          # Conversational memory models
│   ├── orchestration/   # Agent workflows and state machines
│   ├── models/          # Model connectors (OpenAI, Anthropic, Gemini, local)
│   ├── services/        # External services (API integrations, DBs)
│   ├── ui/              # User interface components
│   ├── api/             # FastAPI backend endpoints
│   ├── evaluation/      # Evaluation harnesses (Ragas, TruLens)
│   └── utils/           # Shared utility functions
├── data/                 # Data assets
│   ├── raw_pdfs/        # Input PDF documents (maritime rules, manuals)
│   ├── extracted_text/  # Plain text outputs from parser
│   ├── extracted_images/# Extracted images and figures
│   ├── processed_chunks/# Chunked text files ready for embedding
│   └── metadata/        # Metadata JSON files
├── vector_store/         # Local vector databases
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/                # Automated test suite
│   ├── ingestion/
│   ├── retrieval/
│   ├── agents/
│   └── evaluation/
├── configs/              # System and agent config files
├── scripts/              # Setup, ingestion, and deployment scripts
└── docs/                 # Documentation and architecture diagrams
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd maritimemind-ai
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## Development and Testing

- Run tests using:
  ```bash
  pytest
  ```

## License

This project is licensed under the MIT License.
