from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    page: int | str
    document: Optional[str] = None
    score: float


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Question about the uploaded documents.",
    )

    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Conversation session identifier.",
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of documents to retrieve.",
    )

    document: Optional[str] = None

    source: Optional[str] = None

    page: Optional[int] = Field(
        default=None,
        ge=1,
    )


class ChatResponse(BaseModel):
    answer: str
    found: bool
    citations: List[Citation]
    retrieved_chunks: int
    session_id: str
    metrics: Metrics


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
    )

    document: Optional[str] = None

    source: Optional[str] = None

    page: Optional[int] = Field(
        default=None,
        ge=1,
    )


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    service: str


class MessageResponse(BaseModel):
    message: str

class Metrics(BaseModel):
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float

    input_tokens: int
    output_tokens: int
    total_tokens: int

    estimated_cost_usd: float