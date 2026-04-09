from typing import List

from app.llm.llm_provider import LLMProvider
from app.services.rag_service import RAGService
from app.core.logger import get_logger

from app.intent_router.intent_prompt import INTENT_CLASSIFICATION_PROMPT
from app.schemas.query import QueryResponse


logger = get_logger()


class IntentRouter:
    """
    Routes user queries based on detected intent.
    Small-talk handled directly via LLM.
    Knowledge queries routed to RAGService.
    """

    def __init__(self, llm_provider: LLMProvider, rag_service: RAGService):
        """
        Initialize IntentRouter with dependencies.
        """
        self.llm_provider = llm_provider
        self.rag_service = rag_service
        self.SMALL_TALK_INTENTS = ["GREETING", "IDENTITY", "THANKS", "GOODBYE", "CAPABILITY"]

    def detect_intent_llm(self, query: str) -> str:
        """
        Detects intent without using documents.
        Returns one of: GREETING, IDENTITY, THANKS, GOODBYE, CAPABILITY, RAG
        """

        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
            # Using LLMProvider to generate intent
            response = self.llm_provider.generate_simple_response(prompt)
            intent = response.strip().upper()

            if intent not in self.SMALL_TALK_INTENTS + ["RAG"]:
                logger.warning(f"Unexpected intent from LLM: {intent}. Defaulting to RAG.")
                return "RAG"

            logger.info(f"LLM detected intent: {intent}")
            return intent

        except Exception as e:
            logger.error(f"LLM intent detection error: {e}. Defaulting to RAG.")
            return "RAG"

    def handle_query(self, query: str) -> QueryResponse:
        """
        Routes query to small talk or RAG using LLM-based intent detection.
        """
        try:
            # Step 1: Detect intent using LLM
            intent = self.detect_intent_llm(query)
            logger.info(f"Detected intent via LLM: {intent}")

            # Step 2: Handle small-talk intents directly
            if intent in self.SMALL_TALK_INTENTS:
                # Generate professional small-talk response via LLM
                prompt = f"""
                You are a professional AI assistant.

                Respond to the user message with:
                    1. A polite greeting reply.
                    2. A follow-up offer to help.

                User message: "{query}"

                Response:
                """
                response = self.llm_provider.generate_simple_response(prompt)
                logger.info(f"Small talk response: {response}")
                return QueryResponse(
                    answer=response,
                    sources=[]
                )

            # Step 3: Otherwise, route to RAG pipeline
            logger.info("Routing to RAG pipeline...")
            answer, sources = self.rag_service.query(query)
            return QueryResponse(answer=answer, sources=sources)

        except Exception as e:
            # Step 4: Fallback to RAG in case of error
            logger.error(f"Error in handle_query: {e}. Routing to RAG as fallback.")
            answer, sources = self.rag_service.query(query)
            return QueryResponse(answer=answer, sources=sources)