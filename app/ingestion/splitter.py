from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import tiktoken

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class DocumentSplitter:
    """
    Enterprise Recursive Text Splitter (token-based)

    Responsibilities:
    - Split documents into manageable chunks
    - Preserve metadata
    - Add chunk-level identifiers
    - Respect config-driven chunk settings
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        encoding = tiktoken.get_encoding("cl100k_base")

        def token_length(text: str) -> int:
            return len(encoding.encode(text))

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=token_length,
            separators=["\n\n", "\n", ".", " ",""],
        )

        logger.info(
            f"Splitter initialized | chunk_size={self.chunk_size} | overlap={self.chunk_overlap} | token-based"
        )

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for splitting.")
            return []

        split_docs = []
        chunk_counter = 0

        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                    },
                )

                split_docs.append(new_doc)
                chunk_counter += 1

        logger.info(f"Total chunks created: {chunk_counter}")

        return split_docs