from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from contextlib import asynccontextmanager

from app.services.rag_service import RAGService
from app.api.dependencies import set_rag_service

from app.api.routes import query, health ,ingest, admin
from app.core.logger import get_logger

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing Enterprise RAG system...")

    rag_service = RAGService()

    # store globally
    set_rag_service(rag_service)

    logger.info("Enterprise RAG system ready.")

    yield

    logger.info("Shutting down API...")


def create_app() -> FastAPI:
    """
    Application factory for the Enterprise RAG API.
    """

    logger.info("Creating FastAPI application...")

    app = FastAPI(
        title="Enterprise RAG API",
        description="Production-ready Retrieval Augmented Generation API",
        version="1.0.0",
        lifespan=lifespan
    )

    # Register API routes
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(admin.router)

    logger.info("FastAPI application created successfully.")

    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse("frontend/index.html")
    # -----------------------------------

    logger.info("FastAPI application created successfully.")

    return app


app = create_app() 