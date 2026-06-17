from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from app.services.rag_service import RAGService
from app.api.dependencies import set_rag_service

from app.api.routes import query, health ,ingest, admin
from app.core.logger import get_logger

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing Enterprise RAG system...")

    # 1. Initialize RAG Service
    rag_service = RAGService()
    
    # 2. Pre-initialize retriever (loads heavy models)
    logger.info("Pre-loading RAG models (Retriever/Embeddings)...")
    rag_service.rag.initialize_retriever()

    # 3. Auto-ingest if index is empty
    if rag_service.rag.is_index_empty():
        logger.info("RAG index is empty. Triggering automatic local ingestion from data/raw...")
        rag_service.ingest("data/raw")

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

    # Restrict cross-origin calls to the company site (plus localhost for dev).
    # The wildcard "*" let any website on the internet call the API from a
    # user's browser; "*" + allow_credentials=True is also rejected by browsers.
    allowed_origins = [
        "https://quickfoxconsulting.com",
        "https://www.quickfoxconsulting.com",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
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