from fastapi import APIRouter, Depends

from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)


@router.post("/reindex")
def reindex(rag_service: RAGService = Depends(get_rag_service)):
    """
    Rebuild the entire RAG index from scratch.

    This removes existing FAISS and BM25 indexes
    and runs the ingestion pipeline again.
    """

    logger.info("Admin triggered full reindex.")

    rag_service.reindex()

    return {
        "status": "success",
        "message": "Reindexing completed successfully."
    }