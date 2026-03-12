from app.services.rag_service import RAGService
from app.core.logger import get_logger

logger = get_logger()

rag_service: RAGService | None = None


def set_rag_service(service: RAGService):
    """
    Store global RAGService instance during app startup.
    """
    global rag_service
    rag_service = service


def get_rag_service() -> RAGService:
    """
    Dependency that returns the initialized RAG service.
    """

    if rag_service is None:
        raise RuntimeError("RAGService has not been initialized.")

    return rag_service