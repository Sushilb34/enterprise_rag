from typing import List, Tuple, Dict
from langchain_core.documents import Document


class SourceDeduplicator:
    """
    Removes duplicate sources based on (file_name + page_number).
    Keeps the highest ranked chunk.
    """

    def deduplicate(self, docs: List[Document], max_sources: int = 5) -> List[Document]:

        unique_map: Dict[Tuple[str, int], Document] = {}

        for doc in docs:

            file_name = doc.metadata.get("file_name")
            page = doc.metadata.get("page_number")

            if file_name is None or page is None:
                continue

            key = (file_name, page)

            # keep first occurrence (already reranked order)
            if key not in unique_map:
                unique_map[key] = doc

        deduped_docs = list(unique_map.values())

        return deduped_docs[:max_sources]