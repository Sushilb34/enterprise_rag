from typing import List, Dict, Optional
from langchain_core.documents import Document


class RAGASMapper:
    """
    Responsible for transforming RAG outputs into RAGAS-compatible format.

    This layer ensures:
    - Clean separation between RAG pipeline and evaluation logic
    - Standardized structure for downstream evaluation
    """

    @staticmethod
    def to_ragas_format(
        query: str,
        answer: str,
        documents: List[Document],
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Convert a single RAG response into RAGAS expected format.
        """

        contexts = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                contexts.append(doc.page_content)
            elif hasattr(doc, "content"):
                contexts.append(doc.content)
            elif hasattr(doc, "text"):
                contexts.append(doc.text)
            else:
                contexts.append(str(doc))

        return {
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth or ""
        }