from fastapi import FastAPI

from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-grade RAG Document Q&A API",
)


@app.get("/")
async def root():
    """Basic API information."""
    return {
        "application": settings.app_name,
        "version": "0.1.0",
        "environment": settings.app_env,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-api",
    }