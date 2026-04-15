from typing import List
import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings
from app.core.logger import get_logger
from app.llm.local_llm_client import LocalLLMClient

logger = get_logger()
settings = get_settings()

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
        self.provider = "local" if settings.USE_LOCAL_LLM else settings.LLM_PROVIDER.lower()
        
        self.use_local = settings.USE_LOCAL_LLM
        logger.info(f"Initializing LLM Provider | use_local={self.use_local}")
        if self.use_local:
            logger.info("Using Local LLM Client...")
            self.llm = LocalLLMClient(
                model_name=settings.LOCAL_LLM_MODEL,
                api_url=settings.LOCAL_LLM_API_URL,
                max_tokens=settings.LOCAL_LLM_MAX_TOKENS,
                temperature=settings.LOCAL_LLM_TEMPERATURE,
            )
            
        else:
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
You are Lucy, the virtual assistant at Quickfox Consulting.

USER GUIDELINES:
1. GREETING/IDENTITY: If the user is just saying hello, asking who you are, or making small talk, respond professionally and friendly as Lucy.
2. COMPANY QUESTIONS: If the user asks about Quickfox, its services, or capabilities, use ONLY the provided Context below.
3. STRICTNESS: If the question is RAG-related but the information is missing from the Context, say: "I could not find this information in the provided documents."
4. NO HALLUCINATION: Do not make up facts about the company from your own knowledge.
5. FORMATTING: If the context contains lists or requirements, reproduce them clearly.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def generate_answer(self, query: str, documents: List[Document], stop: list = None) -> str:
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
            if self.provider == "local":
                return self.llm.generate(prompt, stop=stop)
            
            elif self.provider == "openai":
                response = self.llm.invoke(prompt)
                return response.content

            elif self.provider == "gemini":
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating answer: {e}"

    def generate_simple_response(self, query: str, stop: list = None) -> str:
        """
        Lightweight LLM call (no RAG context).
        Used for:
        - Intent classification
        - Small talk responses
        """
        logger.info("Generating simple LLM response (no context)...")

        try:
            if self.provider == "local":
                return self.llm.generate(query, stop=stop)
            
            elif self.provider == "openai":
                response = self.llm.invoke(str(query))
                return response.content.strip()

            elif self.provider == "gemini":
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Simple LLM generation error: {e}")
            return "Hello! How can I assist you today?"