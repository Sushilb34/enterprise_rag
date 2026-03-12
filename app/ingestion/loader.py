from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader

from app.core.logger import get_logger

logger = get_logger()


class PDFLoader:
    """
    Enterprise PDF Loader
 
    Responsibilities:
    - Load PDFs from directory
    - Extract page-wise text
    - Attach metadata (file name, page number, source path)
    - Return LangChain Document objects
    """

    def __init__(self, data_dir: str):
        self.data_path = Path(data_dir)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def load_pdfs(self) -> List[Document]:
        documents = []

        pdf_files = list(self.data_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in directory.")

        logger.info(f"Found {len(pdf_files)} PDF files.")

        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing file: {pdf_path.name}")
                reader = PdfReader(pdf_path)

                for page_number, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if not text or not text.strip():
                        continue

                    doc = Document(
                        page_content=text,
                        metadata={
                            "file_name": pdf_path.name,
                            "page_number": page_number + 1,
                            "source": str(pdf_path.resolve()),
                        },
                    )

                    documents.append(doc)

                logger.info(f"Completed file: {pdf_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")

        logger.info(f"Total extracted pages: {len(documents)}")

        return documents