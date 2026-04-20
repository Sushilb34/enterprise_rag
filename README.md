# Hybrid RAG System

A high-performance, enterprise-grade Retrieval-Augmented Generation (RAG) system featuring hybrid retrieval (Semantic + Keyword), cross-encoder reranking, and website-aware ingestion.

## 🚀 Overview

This system is designed to provide robust, fact-based answers by combining the strengths of vector-based semantic search with traditional keyword matching. It includes specialized logic to handle user typos and provides a clean, API-first architecture.

### Key Features
- **Hybrid Retrieval**: Combines FAISS (Vector Store) and BM25 (Keyword Store) using Reciprocal Rank Fusion (RRF).
- **Advanced Reranking**: Uses a Cross-Encoder model (`ms-marco-MiniLM`) to re-score retrieved documents for maximum relevance.
- **Typo-Tolerance**: LLM-guided prompts specifically designed to handle spelling mistakes (e.g., "vacciencies" -> "vacancies").
- **Multi-Source Ingestion**: Supports local PDFs, Markdown files, and automated website crawling via Crawl4AI.
- **Flexible LLM Support**: Integrates with OpenAI, Gemini 1.5/2.0, or Local LLMs (via OpenAI-compatible APIs).

---

## 🛠️ Installation

### 1. Environment Setup
It is **strongly recommended** to use a 64-bit Python 3.10+ environment to avoid architecture mismatches with PyTorch (Commonly seen as `WinError 193`).

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory. This system follows strict typing via Pydantic; if a required variable is missing, the system will not start.


# SYSTEM PATHS
FAISS_INDEX_PATH=data/vectorstore/faiss_index
BM25_INDEX_PATH=data/vectorstore/bm25_index.pkl
DATA_DIR=data/raw


---

## 🏃 Running the System

### 1. Start the Backend API
Always run the backend from your virtual environment to ensure `torch` dependencies load correctly.

```powershell
.\venv\Scripts\uvicorn app.api.server:app --reload --port 8001
```

### 2. Access Documentation
Once started, you can access the interactive API documentation (Swagger UI) at:
👉 **[http://localhost:8001/docs](http://localhost:8001/docs)**

### 3. Start the Frontend
In a separate terminal, navigate to your website directory and run:
```bash
npm run dev
```

---

## 📥 Ingestion Feature

The ingestion feature allows you to populate or update your RAG knowledge base with new documents.

### How to Ingest
Run the following command in your terminal:
```bash
python run.py --ingest
```

### Why Ingest?
- **Knowledge Update**: You have added new documents (PDFs, Markdown) to your `data/raw` folder and want them to be searchable.
- **Initial Setup**: You are running the system for the first time and need to build the initial index.

### What Happens During Ingestion?
1.  **File Loading**: The system scans the configured `DATA_DIR` for supported file types.
2.  **Document Splitting**: Large files are broken down into smaller, overlapping chunks (based on your `.env` settings).
3.  **Vector Store Update**: Chunks are processed by the embedding model and added to the FAISS index.
4.  **Keyword Index Update**: The BM25 matrix is refreshed to include the new content, ensuring keyword search works across all documents.

---

## 🔄 Reindexing Feature

The reindexing feature allows you to rebuild your knowledge base from scratch.

### How to Reindex
Run the following command in your terminal:
```bash
python run.py --reindex
```

### Why Reindex?
- **Config Changes**: You changed `CHUNK_SIZE` or `CHUNK_OVERLAP` in your `.env`.
- **Model Upgrades**: You switched your `EMBEDDING_MODEL` to a more powerful one.
- **Data Integrity**: If you suspect the current index is corrupted or out of sync.

### What Happens During Reindexing?
1. **Index Clearance**: The existing FAISS and BM25 index files on disk are bypassed/overwritten.
2. **Fresh Scan**: The system re-scans all files in your `DATA_DIR`.
3. **Re-embedding**: Every document chunk is sent through the embedding model to generate new vectors (this is the most time-consuming part).
4. **Keyword Update**: The BM25 matrix is recalculated from scratch.

> [!WARNING]
> Reindexing can be resource-intensive and take several minutes depending on the number of documents and the speed of your embedding model/device.

---

## ⚠️ Troubleshooting & Port Management

### Port 8001 Conflict
If you see an error stating the port is already in use, you likely have a "ghost" uvicorn process.
- **Windows**: `netstat -ano | findstr :8001` then `taskkill /PID <PID> /F`
- **Linux**: `fuser -k 8001/tcp`

### WinError 193 (%1 is not a valid Win32 application)
This happens if you try to run the system using a global Python (like Anaconda) that has a 32-bit dependency mismatch with PyTorch.
- **Fix**: Always prefix your commands with `.\venv\Scripts/python` or `.\venv\Scripts/uvicorn`.
