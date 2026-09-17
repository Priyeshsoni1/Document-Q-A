from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings


class PineconeVectorStore:
    """Pinecone vector database service."""

    def __init__(self):
        settings = get_settings()

        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not configured.")

        self.client = Pinecone(
            api_key=settings.pinecone_api_key
        )

        self.index_name = settings.pinecone_index_name

    def create_index(self, dimension: int):
        """Create the Pinecone index if it does not already exist."""

        existing_indexes = self.client.list_indexes()

        index_names = [
            index["name"]
            for index in existing_indexes
        ]

        if self.index_name in index_names:
            print(
                f"Pinecone index '{self.index_name}' already exists."
            )
            return

        self.client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print(
            f"Created Pinecone index: {self.index_name}"
        )

    def get_index(self):
        """Return the Pinecone index object."""

        return self.client.Index(self.index_name)

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "documents",
    ):
        """Insert or update vectors in Pinecone."""

        index = self.get_index()

        index.upsert(
            vectors=vectors,
            namespace=namespace,
        )

        print(
            f"Upserted {len(vectors)} vectors "
            f"into namespace '{namespace}'."
        )

    def describe_index(self):
        """Return Pinecone index statistics."""

        index = self.get_index()

        return index.describe_index_stats()