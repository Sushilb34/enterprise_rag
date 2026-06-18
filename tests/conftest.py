"""Shared fixtures for the API smoke tests (L3).

These tests exercise the FastAPI routes with the heavy RAG pipeline
(embeddings / FAISS / reranker / LLM backend) replaced by a lightweight fake,
so they run fast and fully offline.

The TestClient is created WITHOUT the ``with`` context manager on purpose:
entering the context would run the app lifespan, which builds the real
RAGService and loads models on startup. Skipping it keeps the tests light, and
we override ``get_rag_service`` so the routes use the fake instead.
"""

import pytest
from fastapi.testclient import TestClient


class FakeRAGService:
    """Stand-in for ``RAGService`` used by the smoke tests.

    ``query()`` mirrors the real signature ``(answer, sources)``. Tests can set
    ``answer`` / ``sources`` to simulate any response (normal answer, refusal,
    no sources) and read ``last_question`` to assert what the route forwarded.
    """

    def __init__(self):
        # Defaults represent a normal, in-scope answer and a ready service.
        self.answer = "Quickfox Consulting provides software consulting services."
        self.sources = [{"file_name": "services.md", "page_number": 1}]
        self.last_question = None
        self.readiness_checks = {"index_loaded": True, "llm_backend": True}

    def query(self, question: str):
        self.last_question = question
        return self.answer, self.sources

    def readiness(self):
        return self.readiness_checks


@pytest.fixture
def client():
    """A TestClient whose ``/query`` route is backed by a FakeRAGService.

    The fake is attached as ``client.fake`` so a test can configure the
    response and inspect what was received.
    """
    from app.api.server import app
    from app.api.dependencies import get_rag_service

    fake = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: fake

    test_client = TestClient(app)
    test_client.fake = fake
    yield test_client

    app.dependency_overrides.clear()
