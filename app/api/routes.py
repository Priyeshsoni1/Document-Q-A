from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MessageResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.generation.rag_chain import RAGService


router = APIRouter(
    prefix="/api/v1",
    tags=["RAG"],
)


# Create one RAG service for the application process.
rag_service = RAGService()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health_check():
    """Check whether the RAG API is running."""

    return {
        "status": "healthy",
        "service": "rag-api",
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    """
    Ask a question and receive a grounded answer
    with document citations.
    """

    try:
        result = rag_service.answer(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            document=request.document,
            source=request.source,
            page=request.page,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc


@router.post(
    "/search",
    response_model=SearchResponse,
)
async def search(request: SearchRequest):
    """
    Search the document knowledge base without
    generating an LLM answer.
    """

    try:
        results = rag_service.retriever.search(
            query=request.query,
            top_k=request.top_k,
            document=request.document,
            source=request.source,
            page=request.page,
        )

        formatted_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                text=result["text"],
                metadata=result["metadata"],
            )
            for result in results
        ]

        return {
            "results": formatted_results,
            "count": len(formatted_results),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to search documents.",
        ) from exc


@router.delete(
    "/chat/{session_id}",
    response_model=MessageResponse,
)
async def clear_chat(session_id: str):
    """Clear conversation history for a session."""

    rag_service.conversation_manager.clear_session(
        session_id
    )

    return {
        "message": (
            f"Conversation '{session_id}' "
            "cleared successfully."
        )
    }