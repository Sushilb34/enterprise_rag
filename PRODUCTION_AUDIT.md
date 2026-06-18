
# Production Readiness Audit — Enterprise RAG

**Date:** 2026-06-17
**Context:** RAG assistant ("Lucy") intended to be embedded in the public Quickfox Consulting website.
**Reviewed:** API layer, LLM/prompt layer, ingestion, config, logging, frontend, deployment posture.

This is a prioritized punch-list. Work top-down — **🔴 Critical** items should be resolved
before any public exposure; **🟠 High** before a real launch; **🟡 Medium / ⚪ Loose ends**
can be scheduled after. Each item notes *what*, *where*, *why it matters*, and a *suggested fix*.

---

## 🔴 Critical — fix before production

### C1. No authentication on any endpoint (esp. destructive admin)
- **Where:** [app/api/routes/admin.py](app/api/routes/admin.py#L15), [ingest.py](app/api/routes/ingest.py#L16), [query.py](app/api/routes/query.py#L14)
- **Why:** `POST /admin/reindex` **deletes the FAISS + BM25 indexes and rebuilds from scratch**
  ([rag_service.py:43-80](app/services/rag_service.py#L43-L80)). Anyone who can reach the API can
  wipe your knowledge base or trigger an expensive re-ingest. `/ingest` and `/query` are also fully open.
- **Fix:** Require an API key / token on `/admin/*` and `/ingest` at minimum (FastAPI dependency
  checking a secret header). Ideally keep admin/ingest off the public-facing deployment entirely
  (internal-only network or separate service). `/query` should at least be rate-limited (see H3).

### C2. Served over plain HTTP — will break when embedded in the HTTPS website
- **Where:** Deployment (`uvicorn ... --host 0.0.0.0 --port 8001`), frontend `fetch` calls.
- **Why:** The company website is HTTPS. A browser on an HTTPS page **blocks** requests to an
  HTTP API (mixed-content) and rejects insecure cookies. Also all queries/answers travel in cleartext on the network.
- **Fix:** Put the API behind a reverse proxy (Nginx/Caddy/Traefik) terminating TLS, or serve via
  the company's existing HTTPS gateway. Frontend should call an HTTPS origin (already relative, so this is mostly infra).

### C3. CORS is fully open (`allow_origins=["*"]`)  ✅ RESOLVED (2026-06-17)
- **Where:** [app/api/server.py](app/api/server.py#L58-L66) (added during debugging).
- **Why:** Any website on the internet can call your API from a user's browser. For a public-but-branded
  assistant you want to restrict who can invoke it. `allow_origins=["*"]` + `allow_credentials=True` is
  also a spec contradiction (browsers reject it when credentials are sent).
- **Fix:** Set `allow_origins` to the explicit company domain(s), e.g. `["https://quickfoxconsulting.com"]`.
  Drop `allow_credentials=True` unless you actually use cookies.
- **Done:** Replaced `["*"]` with an explicit allow-list (`quickfoxconsulting.com` + `www`, plus
  localhost:8001 for dev), set `allow_credentials=False`, and narrowed methods to `GET, POST` and
  headers to `Content-Type`. *(If origins differ per environment later, consider making the list a setting.)*

### C4. No timeout on the LLM HTTP call — a hung backend freezes workers  ✅ RESOLVED (2026-06-17)
- **Where:** [app/llm/local_llm_client.py:76](app/llm/local_llm_client.py#L76) (`requests.post(..., json=payload)` — no `timeout=`).
- **Why:** If the vLLM box (`192.168.1.135:8000`) hangs or is slow, the request blocks the worker
  thread **forever**. A handful of these exhaust the server and the whole site assistant goes down.
- **Fix:** Add `timeout=(connect, read)` (e.g. `timeout=(5, 60)`) and handle `requests.Timeout` /
  `ConnectionError` gracefully with a user-facing fallback message.
- **Done:** Configurable `(connect, read)` timeout (default `(5, 60)`s via new `LOCAL_LLM_CONNECT_TIMEOUT` /
  `LOCAL_LLM_READ_TIMEOUT` settings) applied to the `requests.post` call. Timeout/connection errors now
  propagate up and are caught by H2's generic fallback.

### C5. Rotate the API keys that were exposed
- **Where:** [.env](.env#L6) (OpenAI `sk-proj-...`), [.env:11](.env#L11) (Gemini `AIza...`).
- **Why:** `.env` is correctly gitignored (✅ not in git history), but both keys were pasted into chat
  and sit in plaintext on the laptop. Treat them as compromised.
- **Fix:** Rotate both keys in their consoles. For production use a real secrets manager / environment
  injection rather than a committed-adjacent `.env`. Add a `.env.example` with blank placeholders.

---

## 🟠 High — fix before real launch

### H1. Unbounded query length → cost & DoS abuse  ✅ RESOLVED (2026-06-17)
- **Where:** [app/schemas/query.py:10](app/schemas/query.py#L10) (`query: str` with no `max_length`).
- **Why:** A user can paste megabytes of text. It flows into retrieval, reranking (GPU), and the LLM
  prompt (token cost / latency). Cheap way to abuse or DoS the system.
- **Fix:** `query: str = Field(..., min_length=1, max_length=2000)` (tune the cap). Reject early with 422.
- **Done:** Added `min_length=1, max_length=2000` to the `query` field — empty and oversized queries now
  rejected with 422 before retrieval/GPU/LLM.

### H2. Error details leaked to the client  ✅ RESOLVED (2026-06-17)
- **Where:** [app/llm/llm_provider.py:158-160](app/llm/llm_provider.py#L158-L160) returns `f"Error generating answer: {e}"` as the answer.
- **Why:** Raw exception strings (URLs, stack context, backend internals) can reach end users on a public site.
- **Fix:** Log the real exception server-side; return a generic message ("Sorry, I'm having trouble right
  now — please try again.") to the user. Consider a global FastAPI exception handler.
- **Done:** `generate_answer` now logs via `logger.exception(...)` (full traceback, server-side only) and
  returns a fixed generic message. Also fixed the Windows console log sink to UTF-8 in
  [app/core/logger.py](app/core/logger.py) so exception tracebacks don't crash the print sink.
  *(Note: a global FastAPI exception handler is still worth adding later for non-LLM error paths.)*

### H3. No rate limiting / concurrency control
- **Where:** API layer generally; single uvicorn process, sync endpoints, GPU-bound rerank + ~15s LLM calls.
- **Why:** A public endpoint with no throttle invites scraping/abuse and cost spikes; concurrent requests
  queue on one GPU and degrade for everyone.
- **Fix:** Add rate limiting (e.g. `slowapi`, or at the reverse proxy). Decide a concurrency cap aligned
  with the single-GPU backend.

### H4. Frontend "reindex" button calls a non-existent route
- **Where:** [frontend/index.html:408](frontend/index.html#L408) calls `POST /reindex/`, but the route is
  `POST /admin/reindex` ([admin.py:15](app/api/routes/admin.py#L15)).
- **Why:** The reindex action silently 404s — feature is broken. Also, a reindex trigger should never be
  exposed in the public client at all (ties into C1).
- **Fix:** Remove the reindex control from the public UI; keep reindex an authenticated, internal-only action.

### H5. `/ingest` ignores its `data_path` and the endpoint is misleading
- **Where:** [app/services/rag_service.py:30-41](app/services/rag_service.py#L30-L41) — `ingest(data_dir)`
  accepts a path but calls `self.rag.ingest_documents()` which uses the **configured** dir, ignoring the argument.
- **Why:** The API contract ([ingest.py](app/api/routes/ingest.py#L27)) advertises a `data_path` that does
  nothing — confusing, and `documents_processed` returns `True` (a bool), not a count. (Silver lining: because
  the path is ignored, there's no path-traversal exploit — but fix the contract.)
- **Fix:** Either honor `data_path` safely (validate it's within an allowed base dir) or remove the parameter
  and return the real processed count.

---

## 🟡 Medium — schedule soon

### M1. Logs contain full user queries and full document content  ✅ RESOLVED (2026-06-18)
- **Where:** [query.py:22](app/api/routes/query.py#L22), [llm_provider reranker logs](app/retrieval/reranker.py),
  [logger.py](app/core/logger.py) (file sink, 10-day retention).
- **Why:** User-submitted text on a public site may include PII; logging full questions + retrieved doc
  bodies at INFO creates a privacy footprint and bloats logs.
- **Fix:** Log lengths/IDs instead of full content at INFO; gate verbose content behind DEBUG. Define a
  retention/PII policy.
- **Done:** All four request-path query logs now emit only `length=N chars` at INFO (full text at DEBUG):
  [query.py](app/api/routes/query.py#L22), [rag_service.py](app/services/rag_service.py#L95),
  [main.py](app/main.py#L89), [hybrid_store.py](app/vectorstore/hybrid_store.py#L83). The reranker logs
  source IDs (`file#pN`) at INFO and full doc bodies only at DEBUG
  ([reranker.py](app/retrieval/reranker.py#L73)). Per the requirement that developers still see full
  detail, the project log file (`logs/rag_system.log`) captures DEBUG while the console stays at the
  configured `LOG_LEVEL`; PII/retention policy documented in [logger.py](app/core/logger.py).
  *(Out of scope: [single_ragas_evaluator.py:62](app/evaluation/single_ragas_evaluator.py#L62) logs the
  full question, but it's an offline eval harness on a curated test set, not the public path.)*

### M2. Hardcoded private backend IP for the LLM
- **Where:** [.env:65](.env#L65) `LOCAL_LLM_API_URL = http://192.168.1.135:8000/...`
- **Why:** A DHCP LAN IP is brittle for production; if the box reboots/changes IP the assistant breaks.
  Also unauthenticated and HTTP.
- **Fix:** Use a stable hostname / static reservation, secure the vLLM endpoint, and keep it on a trusted network segment.

### M4. Auto-ingest on startup can surprise you
- **Where:** [app/api/server.py:29-31](app/api/server.py#L29-L31) — if the index is empty on boot, it ingests
  `data/raw` automatically during startup.
- **Why:** In production a missing/empty index would trigger a long blocking ingest on boot (slow startup,
  possible partial state). 
- **Fix:** Make ingestion an explicit, monitored operation; on boot, fail fast / warn if the index is missing
  rather than silently rebuilding.

---

## ⚪ Loose ends / lower priority

- **L1. Dead code: `IntentRouter` is never wired in.** ✅ RESOLVED (2026-06-18) — the `app/intent_router/`
  package (`router.py`, `intent_prompt.py`) was fully implemented but instantiated nowhere; per decision the
  intent router is not needed, so the whole package was deleted. This also removes the unscoped small-talk
  prompt that would have been an injection surface. Verified the app still imports/builds cleanly afterward.
- **L2. Single uvicorn worker + `--reload`** is a dev configuration. For production use a process manager
  (multiple workers behind the proxy, no `--reload`), mindful of the single-GPU backend.
- **L3. No automated tests / CI.** `test_llm_switch.py`, `test_local_llm.py` are ad-hoc scripts. Add at least
  smoke tests for `/health`, `/query`, and the scope-refusal behavior.
  - **Tests ✅ DONE (2026-06-18):** Added a `tests/` pytest suite ([tests/test_smoke.py](tests/test_smoke.py),
    [tests/conftest.py](tests/conftest.py)) with 7 fast, offline smoke tests — `/health`, `/query` happy path,
    query-length validation (empty/missing/oversized → 422, max-length OK, exercises the H1 cap), and
    scope-refusal passthrough. The RAG pipeline is faked via dependency override and the lifespan is skipped
    (no model loading). Added [pytest.ini](pytest.ini) and [requirements-dev.txt](requirements-dev.txt).
    Run with `pytest`.
  - **Still open:** CI (e.g. a GitHub Actions workflow) and a live-backend integration test for the *real*
    LLM scope-refusal (the current refusal test is a contract/passthrough test with the service faked). The
    ad-hoc `test_llm_switch.py` / `test_local_llm.py` root scripts are still present and could be folded in.
- **L4. Stale `docker-compose.yml`** in the repo serves `TinyLlama`, but production serves `Qwen/Qwen3-8B`.
  Sync it so the committed compose matches reality. ✅ RESOLVED (2026-06-18) — compose now serves
  `Qwen/Qwen3-8B` with `--served-model-name qwen3-8b` (matches `LOCAL_LLM_MODEL` in `.env`), port 8000,
  and `--max-model-len 16384` (fits the client's prompt budget + `LOCAL_LLM_MAX_TOKENS=8000`). Service/
  container renamed off `tinyllama`. *Infra-tunable values (`--gpu-memory-utilization 0.90`,
  `--enforce-eager`, `--dtype float16`) are best-guess defaults — confirm against the real GPU box.*
- **L5. Pydantic v2 deprecation:** `Field(..., example=...)` in [query.py](app/schemas/query.py#L10) should be
  `json_schema_extra={"example": ...}`. Cosmetic. ✅ RESOLVED (2026-06-17) — switched to `json_schema_extra` while fixing H1.
- **L6. `documents_processed` returns `True` not an int** ([rag_service.py:41](app/services/rag_service.py#L41)) — type mismatch with the `IngestResponse` int field.
- **L7. No `/health` depth.** Health check returns static OK without verifying the LLM backend or index are
  actually reachable. Consider a `/health/ready` that pings the vLLM endpoint and confirms the index loaded.

---

## Suggested order of attack
1. **C5** (rotate keys) — do immediately, takes minutes.
2. **C4 + H2** (timeout + error handling) — small, high-value robustness.
3. **C1 + H4** (auth + remove public reindex) — close the destructive surface.
4. **C3 + H1** (lock CORS, cap query length) — small edits.
5. **C2** (TLS/reverse proxy) — infra task, coordinate with whoever owns the website.
6. Then work the 🟡 / ⚪ list as time allows.

> We'll tackle these one at a time — tell me which item to start with and I'll implement + verify it.
