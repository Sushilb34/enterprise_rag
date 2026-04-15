from fastapi import APIRouter, Depends
from app.schemas.query import QueryRequest, QueryResponse
from app.api.dependencies import get_rag_service
from app.core.logger import get_logger
from app.services.rag_service import RAGService

logger = get_logger()

router = APIRouter(
    prefix="/query",
    tags=["RAG Query"]
)

@router.post("/", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Query the Enterprise RAG system directly.
    """
    logger.info(f"API Query Received: {request.query}")

    # Step 1: Call RAG pipeline directly
    answer, sources = rag_service.query(request.query)

    return QueryResponse(answer=answer, sources=sources)