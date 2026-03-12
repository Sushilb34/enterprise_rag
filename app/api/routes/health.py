from fastapi import APIRouter
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/")
def health_check():
    """
    Health check endpoint.

    Used to verify that the API service is running.
    This does NOT trigger the RAG pipeline.
    """

    logger.info("Health check endpoint called.")

    return {
        "status": "healthy",
        "service": "Enterprise RAG API"
    }