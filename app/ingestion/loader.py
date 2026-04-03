from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from app.core.logger import get_logger

logger = get_logger()


class DocumentLoader:
    """
    Enterprise Document Loader

    Responsibilities:
    - Scan data/raw directory for documents (PDFs and Markdown files)
    - Extract text with metadata
    - Attach metadata (file name, page number/section, source path)
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

    def load_markdowns(self) -> List[Document]:
        """
        Load Markdown files from directory.
        
        Responsibilities:
        - Load markdown files
        - Extract text content
        - Attach metadata (file name, source path)
        - Return LangChain Document objects
        """
        documents = []

        md_files = list(self.data_path.glob("*.md"))

        if not md_files:
            logger.warning("No Markdown files found in directory.")

        logger.info(f"Found {len(md_files)} Markdown files.")

        for md_path in md_files:
            try:
                logger.info(f"Processing file: {md_path.name}")
                
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content or not content.strip():
                    logger.warning(f"Markdown file is empty: {md_path.name}")
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "file_name": md_path.name,
                        "source": str(md_path.resolve()),
                        "file_type": "markdown"
                    }
                )

                documents.append(doc)
                logger.info(f"Completed file: {md_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {md_path.name}: {str(e)}")

        logger.info(f"Total markdown documents: {len(documents)}")

        return documents

    def load(self) -> List[Document]:
        """
        Load all documents (PDFs and Markdown files) from directory.
        
        Returns:
            Combined list of Document objects from both PDFs and Markdown files
        """
        logger.info("Starting document loading from all sources...")
        
        all_documents = []
        
        # Load PDFs
        pdf_docs = self.load_pdfs()
        all_documents.extend(pdf_docs)
        
        # Load Markdown files
        md_docs = self.load_markdowns()
        all_documents.extend(md_docs)
        
        logger.info(f"Total documents loaded: {len(all_documents)} (PDFs: {len(pdf_docs)}, Markdown: {len(md_docs)})")
        
        return all_documents