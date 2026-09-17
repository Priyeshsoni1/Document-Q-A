from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.generation.conversation import ConversationManager
from app.generation.prompts import RAG_SYSTEM_PROMPT
from app.retrieval.retriever import Retriever


class RAGService:
    """RAG service with conversational history."""

    def __init__(
        self,
        conversation_manager: ConversationManager | None = None,
    ):
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

        self.conversation_manager = (
            conversation_manager
            or ConversationManager()
        )

    def answer(
        self,
        question: str,
        session_id: str = "default",
        top_k: int | None = None,
        document: str | None = None,
        source: str | None = None,
        page: int | None = None,
    ) -> Dict[str, Any]:
        """Generate a grounded answer using conversation history."""

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # -----------------------------------------------------
        # 1. Get previous conversation
        # -----------------------------------------------------

        history = self.conversation_manager.get_history(
            session_id
        )

        # -----------------------------------------------------
        # 2. Retrieve relevant documents
        # -----------------------------------------------------

        results = self.retriever.search(
            query=question,
            top_k=top_k,
            document=document,
            source=source,
            page=page,
        )

        # -----------------------------------------------------
        # 3. Handle no relevant documents
        # -----------------------------------------------------

        if not results:

            answer = (
                "I couldn't find this information "
                "in the provided documents."
            )

            self.conversation_manager.add_message(
                session_id=session_id,
                role="user",
                content=question,
            )

            self.conversation_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
            )

            return {
                "answer": answer,
                "found": False,
                "citations": [],
                "retrieved_chunks": 0,
                "session_id": session_id,
            }

        # -----------------------------------------------------
        # 4. Build document context
        # -----------------------------------------------------

        context = self._build_context(
            results
        )

        # -----------------------------------------------------
        # 5. Build system prompt
        # -----------------------------------------------------

        system_prompt = RAG_SYSTEM_PROMPT.format(
            context=context
        )

        messages = [
            SystemMessage(
                content=system_prompt
            )
        ]

        # -----------------------------------------------------
        # 6. Add previous conversation
        # -----------------------------------------------------

        for message in history:

            if message["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=message["content"]
                    )
                )

            elif message["role"] == "assistant":

                messages.append(
                    AIMessage(
                        content=message["content"]
                    )
                )

        # -----------------------------------------------------
        # 7. Add current question
        # -----------------------------------------------------

        messages.append(
            HumanMessage(
                content=question
            )
        )

        # -----------------------------------------------------
        # 8. Generate answer
        # -----------------------------------------------------

        response = self.llm.invoke(messages)

        answer = response.content

        # -----------------------------------------------------
        # 9. Save conversation
        # -----------------------------------------------------

        self.conversation_manager.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        self.conversation_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # -----------------------------------------------------
        # 10. Build citations
        # -----------------------------------------------------

        citations = self._build_citations(
            results
        )

        return {
            "answer": answer,
            "found": True,
            "citations": citations,
            "retrieved_chunks": len(results),
            "session_id": session_id,
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

        return "\n".join(
            context_parts
        )

    @staticmethod
    def _build_citations(
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create structured citations."""

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