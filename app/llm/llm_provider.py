from typing import List
from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()

# Load environment variables
load_dotenv()

# Try to import Gemini (OpenAI-compatible) client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMProvider:
    """
    Enterprise LLM Provider

    Responsibilities:
    - Load LLM from configuration (.env)
    - Supports OpenAI or Gemini 3.0 Flash
    - Format prompts safely
    - Inject retrieved context
    - Generate final answer
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.LLM_MODEL
        logger.info(f"Initializing LLM Provider | provider={self.provider}")

        # -----------------------------
        # OpenAI setup
        # -----------------------------
        if self.provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )
            logger.info("OpenAI LLM initialized successfully.")

        # -----------------------------
        # Gemini 3 Flash setup
        # -----------------------------
        elif self.provider == "gemini":
            if OpenAI is None:
                raise ImportError("Please install 'openai' package to use Gemini 3.0 Flash")

            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in .env")

            self.llm = OpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/",
            )
            logger.info("Gemini 3.0 Flash LLM initialized successfully.")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # -----------------------------
        # Prompt template
        # -----------------------------
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

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate final answer using retrieved context.
        """

        if not documents:
            logger.warning("No documents provided to LLM.")
            return "I could not find relevant information in the documents."

        logger.info("Generating answer from LLM...")

        # Prepare context with file name and page number
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page_number')}]\n{doc.page_content}"
            for doc in documents
        )

        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )

        # -----------------------------
        # Call the correct provider
        # -----------------------------
        try:
            if self.provider == "openai":
                response = self.llm.invoke(prompt)
                return response.content

            elif self.provider == "gemini":
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message["content"]

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating answer: {e}"