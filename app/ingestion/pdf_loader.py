from pathlib import Path
from typing import List, Dict, Any

import fitz


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF page-by-page.

    Each returned item represents one PDF page and contains
    the page text and metadata required for downstream RAG.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    pages = []

    document = fitz.open(file_path)

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_path": str(path),
                        "page": page_number,
                        "document": path.stem,
                    },
                }
            )
    finally:
        document.close()

    return pages


def load_all_pdfs(directory: str) -> List[Dict[str, Any]]:
    """
    Load all PDF files from a directory.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    pdf_files = sorted(directory_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in directory: {directory}"
        )

    all_pages = []

    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)

    return all_pages