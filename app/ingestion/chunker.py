from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split page-level documents into smaller chunks while
    preserving document and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk": chunk_index,
                    },
                }
            )

    return chunks