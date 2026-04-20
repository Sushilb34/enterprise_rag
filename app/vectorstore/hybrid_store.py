import hashlib
from collections import defaultdict
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger
from app.vectorstore.bm25_store import BM25Store
from app.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger()
settings = get_settings()


class HybridRetriever:
    """
    Enterprise Hybrid Retriever using Reciprocal Rank Fusion (RRF).
    
    This version is resilient to typos and score scaling issues by 
    combining the relative rankings of Vector search and BM25.
    """

    def __init__(self, documents: List[Document] | None = None):
        # alpha = weight for vector results (e.g., 0.7)
        # 1-alpha = weight for BM25 results (e.g., 0.3)
        self.alpha = settings.HYBRID_ALPHA  
        self.top_k = settings.TOP_K
        
        # RRF constant (Standard is 60). 
        # It smoothes the impact of high-ranking documents.
        self.rrf_k = 60 

        logger.info(f"Initializing RRF Hybrid Retriever | alpha={self.alpha} | top_k={self.top_k}")

        self.vector_store = FAISSVectorStore()
        self.bm25_store = BM25Store()

        if documents:
            self.vector_store.add_documents(documents)
            self.bm25_store.build_index(documents)
        else:
            self._load_indexes()

    def add_documents(self, documents: List[Document]):
        """
        Add documents to both FAISS and BM25 indexes.
        Reuses existing models to save memory.
        """
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} new documents to hybrid retriever.")
        
        # 1. Update FAISS (this also saves to disk internally)
        self.vector_store.add_documents(documents)
        
        # 2. Update BM25 (rebuilds with current + new docs or just creates if empty)
        # BM25 build_index logic will handle disk check/merge logic
        self.bm25_store.build_index(documents)
        
        logger.info("Hybrid indexes updated successfully.")


    def _load_indexes(self):
        """Loads existing indexes for query-only mode."""
        try:
            self.bm25_store._load_from_disk()
            logger.info("BM25 index loaded from disk.")
        except Exception:
            try:
                ds = self.vector_store.vectorstore.docstore
                docs_for_bm25 = list(getattr(ds, "_dict", {}).values())
                self.bm25_store.build_index(docs_for_bm25)
                logger.info(f"Rebuilt BM25 from FAISS docstore ({len(docs_for_bm25)} docs).")
            except Exception as e:
                logger.warning(f"BM25 initialization failed: {e}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion.
        """
        logger.info(f"Running RRF hybrid retrieval for: {query}")

        # 1. Fetch more results than top_k to ensure we have a good overlap for RRF
        fetch_k = self.top_k * 2

        # 2. Get Vector search results (Ranked by similarity)
        # FAISS similarity_search_with_score returns (doc, distance)
        vector_results = self.vector_store.vectorstore.similarity_search_with_score(
            query, k=fetch_k
        )

        # 3. Get BM25 results (Ranked by keyword relevance)
        bm25_results = self.bm25_store.search(query, k=fetch_k)

        # 4. RRF Scoring Logic
        # rrf_score = sum( weight / (rrf_k + rank) )
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        # Process Vector Ranks
        for rank, (doc, _score) in enumerate(vector_results, start=1):
            doc_id = self._doc_identifier(doc)
            doc_map[doc_id] = doc
            # Apply alpha weight to vector ranking
            rrf_scores[doc_id] += self.alpha * (1.0 / (self.rrf_k + rank))

        # Process BM25 Ranks
        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            doc_id = self._doc_identifier(doc)
            doc_map[doc_id] = doc
            # Apply (1-alpha) weight to keyword ranking
            rrf_scores[doc_id] += (1 - self.alpha) * (1.0 / (self.rrf_k + rank))

        # 5. Sort by aggregated RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_docs = []
        for rank, (doc_id, combined_score) in enumerate(sorted_docs[:self.top_k], start=1):
            doc = doc_map[doc_id]
            
            # Metadata for debugging
            doc.metadata["hybrid_score"] = float(combined_score)
            doc.metadata["retrieval_rank"] = rank
            doc.metadata["retrieval_method"] = "rrf_hybrid"
            
            final_docs.append(doc)

        if not final_docs:
            logger.warning("No documents retrieved for query.")
            return []

        logger.info(
            f"RRF Complete | top_score={final_docs[0].metadata['hybrid_score']:.6f} | "
            f"docs_returned={len(final_docs)}"
        )

        return final_docs

    def _doc_identifier(self, doc: Document) -> str:
        """
        Generates a unique ID for a document to prevent duplicates 
        during fusion. Uses metadata if available, falls back to content hash.
        """
        # Attempt to use specific metadata
        file_name = doc.metadata.get('file_name', '')
        chunk_id = doc.metadata.get('chunk_id', '')
        
        if file_name and chunk_id:
            return f"{file_name}_{chunk_id}"
        
        # Fallback: Hash the page content if metadata is missing
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def is_empty(self) -> bool:
        """
        Check if both FAISS and BM25 indexes are empty.
        """
        faiss_empty = self.vector_store.is_empty()
        bm25_empty = self.bm25_store.is_empty()

        return faiss_empty and bm25_empty