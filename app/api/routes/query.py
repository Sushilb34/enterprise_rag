from fastapi import APIRouter, Depends

from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logger import get_logger

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
    Query the Enterprise RAG system.

    This endpoint:
    1. Receives a user question
    2. Retrieves relevant documents
    3. Generates an answer using the LLM
    4. Returns answer + sources
    """

    logger.info(f"API Query Received: {request.query}")

    answer, sources = rag_service.query(request.query)

    return QueryResponse(
        answer=answer,
        sources=sources
    )