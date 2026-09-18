from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.monitoring.middleware import (
    request_logging_middleware,
)


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Production-grade RAG Document Q&A API "
        "with retrieval, citations, evaluation, "
        "and observability."
    ),
)


# ---------------------------------------------------------
# Middleware
# ---------------------------------------------------------

app.middleware(
    "http"
)(request_logging_middleware)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(router)


@app.get("/")
async def root():

    return {
        "application": settings.app_name,
        "version": "1.0.0",
        "environment": settings.app_env,
        "status": "running",
    }