# Production-Grade Document Q&A System

A production-oriented Retrieval-Augmented Generation (RAG) system for
question answering over multiple PDF documents.

## Project Goals

- Process 10+ PDF documents
- Extract and chunk document content
- Generate embeddings
- Store vectors in Pinecone
- Retrieve relevant context
- Generate grounded answers using an LLM
- Provide document and page-level citations
- Support metadata filtering
- Support configurable Top-K retrieval
- Maintain conversation history
- Detect questions not supported by the knowledge base
- Evaluate the RAG pipeline using RAGAS
- Track latency, token usage, and cost
- Provide a FastAPI backend
- Provide a Streamlit interface
- Dockerize and deploy the application

## Architecture

PDFs
→ Document Processing
→ Chunking
→ Embeddings
→ Pinecone
→ Retrieval
→ LLM
→ Answer + Citations

## Current Phase

Phase 1 - Project Setup
