from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import PineconeVectorStore


class Retriever:
    """Retrieve relevant document chunks from Pinecone."""

    def __init__(self):
        settings = get_settings()

        self.top_k = settings.top_k
        self.similarity_threshold = settings.similarity_threshold

        self.embedding_service = EmbeddingService()
        self.vector_store = PineconeVectorStore()
        self.index = self.vector_store.get_index()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document: Optional[str] = None,
        source: Optional[str] = None,
        page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search Pinecone for relevant document chunks."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = top_k or self.top_k

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = (
            self.embedding_service.embed_text(
                query.strip()
            )
        )

        metadata_filter = self._build_filter(
            document=document,
            source=source,
            page=page,
        )

        query_kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": "documents",
        }

        if metadata_filter:
            query_kwargs["filter"] = metadata_filter

        response = self.index.query(
            **query_kwargs
        )

        results = []

        for match in response.matches:

            score = float(match.score)

            # Ignore weak semantic matches.
            if score < self.similarity_threshold:
                continue

            metadata = match.metadata or {}

            results.append(
                {
                    "id": match.id,
                    "score": score,
                    "text": metadata.get(
                        "text",
                        "",
                    ),
                    "metadata": {
                        "source": metadata.get(
                            "source"
                        ),
                        "document": metadata.get(
                            "document"
                        ),
                        "page": metadata.get(
                            "page"
                        ),
                        "chunk": metadata.get(
                            "chunk"
                        ),
                        "file_path": metadata.get(
                            "file_path"
                        ),
                    },
                }
            )

        return results

    @staticmethod
    def _build_filter(
        document: Optional[str] = None,
        source: Optional[str] = None,
        page: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build Pinecone metadata filter."""

        conditions = []

        if document:
            conditions.append(
                {
                    "document": {
                        "$eq": document
                    }
                }
            )

        if source:
            conditions.append(
                {
                    "source": {
                        "$eq": source
                    }
                }
            )

        if page is not None:
            conditions.append(
                {
                    "page": {
                        "$eq": page
                    }
                }
            )

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }