from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_rag_service
from app.services.rag_service import RAGService
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/")
def health_check():
    """
    Liveness check.

    Verifies the API process is up and serving. Static — does NOT touch the
    RAG pipeline, the index, or the LLM backend (use /health/ready for that).
    """

    logger.info("Health check endpoint called.")

    return {
        "status": "healthy",
        "service": "Enterprise RAG API"
    }


@router.get("/ready")
def readiness_check(rag_service: RAGService = Depends(get_rag_service)):
    """
    Readiness check.

    Confirms the service can actually serve a query: the retrieval index is
    loaded and the LLM backend is reachable. Returns 200 when ready, 503
    otherwise, with a per-check breakdown so failures are diagnosable.
    """

    logger.info("Readiness check endpoint called.")

    checks = rag_service.readiness()
    ready = all(checks.values())

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        },
    )