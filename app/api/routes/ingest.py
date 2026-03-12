from fastapi import APIRouter, Depends

from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(
    prefix="/ingest",
    tags=["Document Ingestion"]
)


@router.post("/", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Trigger document ingestion for the RAG system.
    """

    logger.info("API ingestion request received.")

    count = rag_service.ingest(request.data_path)

    return IngestResponse(
        message="Documents ingested successfully",
        documents_processed=count
    )