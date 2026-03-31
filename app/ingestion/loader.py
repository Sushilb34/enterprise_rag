from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

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

    def __init__(self, data_dir: str = "data/raw"):
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
                loader = PyMuPDFLoader(str(pdf_path))
                pages = loader.load() # return list of document objects

                for page_number, page in enumerate(pages, start=1):
                    if not page.page_content or not page.page_content.strip():
                        continue


                    page.metadata.update({
                            "file_name": pdf_path.name,
                            "page_number": page_number,
                            "source": str(pdf_path.resolve()),
                        }
                    )

                    documents.append(page)

                logger.info(f"Completed file: {pdf_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")

        logger.info(f"Total extracted pages: {len(documents)}")

        return documents