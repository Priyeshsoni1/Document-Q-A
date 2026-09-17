from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import create_chunks
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import PineconeVectorStore


def main():
    settings = get_settings()

    documents_directory = Path("data/documents")

    print("=" * 60)
    print("RAG VECTOR INDEXING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load PDFs
    # ---------------------------------------------------------

    print("\n[1/5] Loading PDF documents...")

    pages = load_all_pdfs(
        str(documents_directory)
    )

    print(f"Pages loaded: {len(pages)}")

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------

    print("\n[2/5] Creating chunks...")

    chunks = create_chunks(
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created.")

    # ---------------------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------------------

    print("\n[3/5] Generating embeddings...")

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_service.embed_documents(
        texts=texts
    )

    print(
        f"Embeddings generated: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: {len(embeddings[0])}"
    )

    # ---------------------------------------------------------
    # 4. Create Pinecone index
    # ---------------------------------------------------------

    print("\n[4/5] Setting up Pinecone...")

    vector_store = PineconeVectorStore()

    dimension = len(embeddings[0])

    vector_store.create_index(
        dimension=dimension
    )

    # ---------------------------------------------------------
    # 5. Upload vectors
    # ---------------------------------------------------------

    print("\n[5/5] Uploading vectors to Pinecone...")

    vectors = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        metadata = {
            **chunk["metadata"],
            "text": chunk["text"],
        }

        vector = {
            "id": str(uuid4()),
            "values": embedding,
            "metadata": metadata,
        }

        vectors.append(vector)

    # Pinecone supports batching. Keep batches manageable.
    batch_size = 100

    for start in range(
        0,
        len(vectors),
        batch_size,
    ):
        batch = vectors[
            start:start + batch_size
        ]

        vector_store.upsert(
            vectors=batch,
            namespace="documents",
        )

    print("\nIndex statistics:")

    stats = vector_store.describe_index()

    print(stats)

    print("\n" + "=" * 60)
    print("VECTOR INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()