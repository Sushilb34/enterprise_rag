from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class LLMProvider:
    """
    Enterprise LLM Provider

    Responsibilities:
    - Load LLM from configuration
    - Format prompts safely
    - Inject retrieved context
    - Generate final answer
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

        if self.provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a professional AI assistant for a company website.

Use ONLY the provided context to answer the user's question.
If the answer is not found in the context, say:
"I could not find this information in the provided documents."

Be precise, clear, and professional.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        logger.info("LLM Provider initialized successfully.")

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate final answer using retrieved context.
        """

        if not documents:
            logger.warning("No documents provided to LLM.")
            return "I could not find relevant information in the documents."

        logger.info("Generating answer from LLM...")

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page_number')}]\n{doc.page_content}"
            for doc in documents
        )

        prompt = self.prompt_template.format(
            context=context,
            question=query
        )

        response = self.llm.invoke(prompt)

        logger.info("Answer generated successfully.")

        return response.content