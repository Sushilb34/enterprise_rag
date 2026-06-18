"""Smoke tests for the public API surface (L3).

Covers:
- ``/health`` returns a healthy status without touching the RAG pipeline.
- ``/query`` happy path returns the service answer + sources and forwards the
  question unchanged.
- Query-length validation (the H1 cap): empty / missing / oversized queries are
  rejected with 422; a max-length query is accepted.
- Scope-refusal passthrough: when the service returns Lucy's out-of-scope
  refusal with no sources, the route surfaces it as-is with an empty source list.

The scope-refusal test is a *contract* test (the RAG service is faked); it does
not exercise the real LLM guardrail. A live-backend integration test is left as
a follow-up.
"""

# Canonical fragment of Lucy's out-of-scope refusal (see the prompt in
# app/llm/llm_provider.py).
REFUSAL_FRAGMENT = "only help with questions about Quickfox"


def test_health_ok(client):
    resp = client.get("/health/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "Enterprise RAG API"


def test_query_happy_path(client):
    client.fake.answer = "Quickfox offers RAG and AI consulting."
    client.fake.sources = [{"file_name": "about.md", "page_number": 2}]

    resp = client.post("/query/", json={"query": "What does Quickfox do?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Quickfox offers RAG and AI consulting."
    assert body["sources"] == [{"file_name": "about.md", "page_number": 2}]
    # The route forwarded the user's question to the service unchanged.
    assert client.fake.last_question == "What does Quickfox do?"


def test_query_empty_is_rejected(client):
    resp = client.post("/query/", json={"query": ""})
    assert resp.status_code == 422


def test_query_missing_field_is_rejected(client):
    resp = client.post("/query/", json={})
    assert resp.status_code == 422


def test_query_too_long_is_rejected(client):
    resp = client.post("/query/", json={"query": "x" * 2001})
    assert resp.status_code == 422


def test_query_at_max_length_is_accepted(client):
    resp = client.post("/query/", json={"query": "x" * 2000})
    assert resp.status_code == 200


def test_scope_refusal_passthrough(client):
    # Simulate the RAG service returning Lucy's out-of-scope refusal.
    client.fake.answer = (
        "I'm Lucy, the Quickfox Consulting assistant, so I can "
        "only help with questions about Quickfox. 😊"
    )
    client.fake.sources = []

    resp = client.post("/query/", json={"query": "Write me a poem about cats."})

    assert resp.status_code == 200
    body = resp.json()
    assert REFUSAL_FRAGMENT in body["answer"]
    # No sources should be surfaced for an out-of-scope refusal.
    assert body["sources"] == []
