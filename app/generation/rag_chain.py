from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.generation.prompts import RAG_SYSTEM_PROMPT
from app.retrieval.retriever import Retriever


class RAGService:
    """RAG service for grounded document question answering."""

    def __init__(self):
        settings = get_settings()

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.retriever = Retriever()

        self.llm = ChatOpenAI(
            model="inclusionai/ling-3.0-flash-vl:free",
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document: str | None = None,
        source: str | None = None,
        page: int | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer using retrieved documents.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # -----------------------------------------------------
        # 1. Retrieve relevant chunks
        # -----------------------------------------------------

        results = self.retriever.search(
            query=question,
            top_k=top_k,
            document=document,
            source=source,
            page=page,
        )

        # -----------------------------------------------------
        # 2. Handle no relevant documents
        # -----------------------------------------------------

        if not results:
            return {
                "answer": (
                    "I couldn't find this information "
                    "in the provided documents."
                ),
                "found": False,
                "citations": [],
                "retrieved_chunks": 0,
            }

        # -----------------------------------------------------
        # 3. Build context
        # -----------------------------------------------------

        context = self._build_context(results)

        # -----------------------------------------------------
        # 4. Create prompt
        # -----------------------------------------------------

        system_prompt = RAG_SYSTEM_PROMPT.format(
            context=context
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=question
            ),
        ]

        # -----------------------------------------------------
        # 5. Generate answer
        # -----------------------------------------------------

        response = self.llm.invoke(messages)

        answer = response.content

        # -----------------------------------------------------
        # 6. Generate citations from retrieved metadata
        # -----------------------------------------------------

        citations = self._build_citations(results)

        return {
            "answer": answer,
            "found": True,
            "citations": citations,
            "retrieved_chunks": len(results),
        }

    @staticmethod
    def _build_context(
        results: List[Dict[str, Any]]
    ) -> str:
        """Convert retrieved chunks into LLM context."""

        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            document = metadata.get(
                "document",
                "Unknown",
            )

            context_parts.append(
                f"""
--- CONTEXT {index} ---
Document: {document}
Source: {source}
Page: {page}

Content:
{result["text"]}
"""
            )

        return "\n".join(context_parts)

    @staticmethod
    def _build_citations(
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create structured citations from retrieved metadata."""

        citations = []

        seen = set()

        for result in results:
            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            citation_key = (
                source,
                page,
            )

            if citation_key in seen:
                continue

            seen.add(citation_key)

            citations.append(
                {
                    "source": source,
                    "page": page,
                    "document": metadata.get(
                        "document"
                    ),
                    "score": result["score"],
                }
            )

        return citations