from pathlib import Path

from app.core.config import get_settings
from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import create_chunks


def main():
    settings = get_settings()

    documents_directory = Path("data/documents")

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    print(f"\nDocument directory: {documents_directory}")

    # Load PDFs
    pages = load_all_pdfs(str(documents_directory))

    print(f"Pages extracted: {len(pages)}")

    # Create chunks
    chunks = create_chunks(
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"Chunks created: {len(chunks)}")

    print("\nSample chunk:")
    print("-" * 60)

    if chunks:
        sample = chunks[0]

        print(sample["text"][:500])
        print("\nMetadata:")
        print(sample["metadata"])

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()