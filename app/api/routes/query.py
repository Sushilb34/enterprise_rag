from fastapi import APIRouter, Depends
from app.schemas.query import QueryRequest, QueryResponse
from app.api.dependencies import get_intent_router
from app.core.logger import get_logger
from app.intent_router.router import IntentRouter

logger = get_logger()

router = APIRouter(
    prefix="/query",
    tags=["RAG Query"]
)

@router.post("/", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    intent_router: IntentRouter = Depends(get_intent_router)
):
    """
    Query the Enterprise RAG system via IntentRouter.

    This endpoint:
    1. Detects the intent of the query (small-talk or RAG)
    2. Routes small-talk queries to LLM directly
    3. Routes knowledge queries to RAG pipeline
    4. Returns answer + sources
    """
    logger.info(f"API Query Received: {request.query}")

    # Step 1: Let IntentRouter handle the query
    response = intent_router.handle_query(request.query)

    return response