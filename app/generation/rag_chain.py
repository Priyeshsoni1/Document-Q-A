import time
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
from app.monitoring.logger import logger
from app.monitoring.metrics import (
    calculate_cost,
    extract_usage,
)
from app.retrieval.retriever import Retriever


class RAGService:
    """RAG service with conversation and observability."""

    def __init__(
        self,
        conversation_manager=None,
    ):
        settings = get_settings()

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.settings = settings

        self.retriever = Retriever()

        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=settings.llm_model,
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

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        total_start = time.perf_counter()

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        retrieval_start = time.perf_counter()

        results = self.retriever.search(
            query=question,
            top_k=top_k,
            document=document,
            source=source,
            page=page,
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # -------------------------------------------------
        # No results
        # -------------------------------------------------

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

            total_latency_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            logger.info(
                "rag_no_context session_id=%s "
                "retrieval_latency_ms=%.2f "
                "total_latency_ms=%.2f",
                session_id,
                retrieval_latency_ms,
                total_latency_ms,
            )

            return {
                "answer": answer,
                "found": False,
                "citations": [],
                "retrieved_chunks": 0,
                "session_id": session_id,
                "metrics": {
                    "retrieval_latency_ms": round(
                        retrieval_latency_ms,
                        2,
                    ),
                    "llm_latency_ms": 0.0,
                    "total_latency_ms": round(
                        total_latency_ms,
                        2,
                    ),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            }

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = self._build_context(
            results
        )

        system_prompt = RAG_SYSTEM_PROMPT.format(
            context=context
        )

        messages = [
            SystemMessage(
                content=system_prompt
            )
        ]

        # -------------------------------------------------
        # Conversation history
        # -------------------------------------------------

        history = (
            self.conversation_manager.get_history(
                session_id
            )
        )

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

        messages.append(
            HumanMessage(
                content=question
            )
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        llm_start = time.perf_counter()

        response = self.llm.invoke(
            messages
        )

        llm_latency_ms = (
            time.perf_counter()
            - llm_start
        ) * 1000

        answer = response.content

        # -------------------------------------------------
        # Token usage
        # -------------------------------------------------

        usage = extract_usage(
            response
        )

        estimated_cost = calculate_cost(
            model=self.settings.llm_model,
            input_tokens=usage[
                "input_tokens"
            ],
            output_tokens=usage[
                "output_tokens"
            ],
        )

        # -------------------------------------------------
        # Save conversation
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Citations
        # -------------------------------------------------

        citations = self._build_citations(
            results
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        logger.info(
            "rag_request session_id=%s "
            "retrieved_chunks=%s "
            "retrieval_latency_ms=%.2f "
            "llm_latency_ms=%.2f "
            "total_latency_ms=%.2f "
            "input_tokens=%s "
            "output_tokens=%s "
            "total_tokens=%s "
            "estimated_cost_usd=%.8f",
            session_id,
            len(results),
            retrieval_latency_ms,
            llm_latency_ms,
            total_latency_ms,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
            estimated_cost,
        )

        return {
            "answer": answer,
            "found": True,
            "citations": citations,
            "retrieved_chunks": len(results),
            "session_id": session_id,
            "metrics": {
                "retrieval_latency_ms": round(
                    retrieval_latency_ms,
                    2,
                ),
                "llm_latency_ms": round(
                    llm_latency_ms,
                    2,
                ),
                "total_latency_ms": round(
                    total_latency_ms,
                    2,
                ),
                "input_tokens": usage[
                    "input_tokens"
                ],
                "output_tokens": usage[
                    "output_tokens"
                ],
                "total_tokens": usage[
                    "total_tokens"
                ],
                "estimated_cost_usd": estimated_cost,
            },
        }

    @staticmethod
    def _build_context(
        results: List[Dict[str, Any]]
    ) -> str:

        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            metadata = result["metadata"]

            context_parts.append(
                f"""
--- CONTEXT {index} ---
Document: {metadata.get("document", "Unknown")}
Source: {metadata.get("source", "Unknown")}
Page: {metadata.get("page", "Unknown")}

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

            key = (
                source,
                page,
            )

            if key in seen:
                continue

            seen.add(key)

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