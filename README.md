# Production-Grade Document Q&A System

A production-oriented Retrieval-Augmented Generation (RAG)
application for conversational question answering over PDF documents.

## Features

- 10+ PDF document support
- Page-aware PDF extraction
- Recursive document chunking
- OpenAI embeddings
- Pinecone vector database
- Semantic similarity search
- Configurable Top-K retrieval
- Metadata filtering
- Similarity threshold
- Grounded LLM responses
- Document/page-level citations
- Answer-not-found handling
- Multi-turn conversation history
- FastAPI backend
- Streamlit frontend
- RAGAS evaluation
- Faithfulness evaluation
- Answer relevancy evaluation
- Context precision evaluation
- Context recall evaluation
- Latency tracking
- Token usage tracking
- Estimated LLM cost tracking
- Structured application logging
- Dockerized deployment

## Architecture

PDFs
→ PyMuPDF
→ Chunking
→ Embeddings
→ Pinecone
→ Retrieval
→ LLM
→ Answer + Citations

## Technology Stack

- Python
- FastAPI
- LangChain
- OpenAI
- Pinecone
- PyMuPDF
- RAGAS
- Streamlit
- Docker

## Project Structure

production-rag-document-qa/

├── app/
│ ├── api/
│ ├── core/
│ ├── ingestion/
│ ├── retrieval/
│ ├── generation/
│ ├── evaluation/
│ ├── monitoring/
│ └── main.py
│
├── data/
│ └── documents/
│
├── evaluation/
│ ├── dataset.json
│ ├── evaluate_rag.py
│ └── results/
│
├── scripts/
│
├── ui/
│ ├── streamlit_app.py
│ └── evaluation_dashboard.py
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## Setup

Create virtual environment:

python -m venv .venv

Activate environment and install dependencies:

pip install -r requirements.txt

Create environment file:

cp .env.example .env

Add API credentials to `.env`.

## Run API

uvicorn app.main:app --reload

API:

http://localhost:8000

Swagger:

http://localhost:8000/docs

## Run Streamlit

streamlit run ui/streamlit_app.py

## Run Evaluation

python evaluation/evaluate_rag.py

## Run Docker

docker compose up --build

API:

http://localhost:8000

UI:

http://localhost:8501

## Evaluation

The system evaluates:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Target average RAG quality:

> = 0.75

Actual scores should be generated from the evaluation dataset and
reported rather than hard-coded.

## Production Observability

The application tracks:

- Request ID
- HTTP status
- Retrieval latency
- LLM latency
- Total latency
- Input tokens
- Output tokens
- Total tokens
- Estimated LLM cost
- Errors

## Security

API keys are loaded from environment variables.

`.env` is excluded from Git.

User documents are excluded from the Docker image.

Production deployments should restrict CORS origins and store
secrets using the cloud provider's secret-management system.

## Future Improvements

- Hybrid BM25 + vector search
- Cross-encoder reranking
- Redis conversation storage
- PostgreSQL persistence
- Authentication
- Rate limiting
- Background document ingestion
- OpenTelemetry
- Prometheus/Grafana monitoring
