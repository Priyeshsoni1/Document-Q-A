from typing import List

from openai import OpenAI

from app.core.config import get_settings


class EmbeddingService:
    """Generate embeddings using OpenAI."""

    def __init__(self):
        settings = get_settings()

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=settings.openai_api_key,base_url="https://openrouter.ai/api/v1",)
        self.model = settings.embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Generate embeddings for multiple documents in batches."""

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            embeddings = [
                item.embedding
                for item in response.data
            ]

            all_embeddings.extend(embeddings)

        return all_embeddings