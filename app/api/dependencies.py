from fastapi import Depends
from app.intent_router.router import IntentRouter
from app.services.rag_service import RAGService
from app.llm.llm_provider import LLMProvider

from app.core.logger import get_logger

logger = get_logger()

rag_service: RAGService | None = None
llm_provider: LLMProvider | None = None


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

# -----------------------------
# New dependencies for IntentRouter
# -----------------------------
def get_llm_provider() -> LLMProvider:
    """
    Returns an initialized LLMProvider.
    """
    return LLMProvider()

def get_intent_router(
    rag_service: RAGService = Depends(get_rag_service),
    llm_provider: LLMProvider = Depends(get_llm_provider)
) -> IntentRouter:
    """
    Returns an initialized IntentRouter for FastAPI dependency injection.
    """
    return IntentRouter(llm_provider=llm_provider, rag_service=rag_service)