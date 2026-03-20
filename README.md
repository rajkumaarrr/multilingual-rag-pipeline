# 🌏 Multilingual RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built for Southeast Asian languages, powered by **SEA-LION**, **BGE-M3**, and **Qdrant**. Supports both a terminal chat interface and a Streamlit web UI.

---

## ✨ Features

- **Multilingual** — Thai, Indonesian, Vietnamese, Filipino, Malay, Hindi, Tamil, Bengali, English, and more
- **Hybrid retriever** — keyword-exact search for structured Excel data + vector similarity for prose documents
- **BGE-M3 embeddings** — 1024-dim multilingual embeddings via Ollama
- **SEA-LION 8B LLM** — Llama-based model fine-tuned for Southeast Asian languages
- **BGE Reranker v2-m3** — cross-encoder reranker trims retrieved chunks to top-3 before LLM inference
- **Two-layer Redis cache** — exact SHA-256 cache + semantic cosine similarity cache with language-directive guard
- **LangSmith tracing** — full observability for every retrieval, rerank, and LLM call
- **Streamlit UI** — web interface with Connect / Load & Index / Upload / Scrape options
- **Terminal chat** — lightweight CLI interface via `python main.py`

---

## 🗂 Repository Structure

```
├── main.py                 # All pipeline logic + terminal chat loop
├── app.py                  # Streamlit web UI (imports nothing from main.py)
├── web_scraper.py          # URL scraping and table extraction
├── setup_logger.py         # Loguru logger (terminal + file logging)
├── docker-compose.yml      # Redis + Qdrant services
├── requirements.txt        # Python dependencies
└── data/                   # Put your documents here (create this folder)
```

---

## 🖥 System Requirements

| Component | Requirement |

| Python | 3.11.3 or higher |
| OS | Windows 10/11, macOS, or Linux |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | Optional but strongly recommended for LLM inference |
| Docker | Required for Redis and Qdrant |

---

## Prerequisites — Install Before Running

### 1. Docker Desktop
Download and install from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

### 2. Ollama
Download and install from [https://ollama.com](https://ollama.com)

After installing, pull the required models:
```bash
ollama pull bge-m3
ollama pull aisingapore/Llama-SEA-LION-v3.5-8B-R
```

### 3. Tesseract OCR *(only needed for scanned PDFs and image files)*

**Windows:**
Download the installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`



### 4. Poppler *(only needed for scanned PDF OCR)*

**Windows:**
Download from [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), extract, and note the `bin/` folder path.


## Setup & Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `FlagEmbedding` (the reranker) is optional. The pipeline runs without it — the reranker is automatically disabled if it's not installed.

### Step 4 — Start Redis and Qdrant
```bash
docker compose up -d
```

Verify services are running:
```bash
docker ps
```
You should see `rag-redis` and `rag-qdrant` listed as running.

### Step 5 — Configure paths and API keys

Open `main.py` (or `app.py`) and update these values near the top:

```python
# Tesseract path (Windows only — skip on macOS/Linux)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# LangSmith API key (optional — get one at https://smith.langchain.com)
os.environ.setdefault("LANGCHAIN_API_KEY", "YOUR_LANGSMITH_API_KEY_HERE")
```

For **Windows Poppler** (scanned PDF OCR only), also update in `app.py`:
```python
POPPLER_PATH = r"C:\path\to\poppler\Library\bin"
```

> macOS and Linux users do not need to set `POPPLER_PATH` — it is found automatically.




## ▶️ Running the Pipeline

### Option A — Terminal chat
```bash
python main.py
```
Loads all documents from `./data`, builds the index, then starts a CLI chat loop.

**Chat commands:**
- Type a question → get an answer
- Prefix with `!` → bypass cache and force fresh retrieval (e.g. `!what is X?`)
- Paste a URL → scrape and index that website
- Type `exit` or `quit` → shut down

### Option B — Streamlit web UI
```bash
streamlit run app.py
```
Opens at [http://localhost:8501](http://localhost:8501)

The UI starts instantly (no auto-indexing). Use the sidebar to choose:

| Button | What it does |

| **⚡ Connect** | Attach to an existing Qdrant index — use this on restarts |
| **📂 Load & Index** | Load all files from a directory and build a fresh index |
| **📤 Upload Files** | Upload individual files and add them to the current index |
| **🌐 Scrape Website** | Scrape a URL and add it to the current index |

**Typical workflow:**
1. First run: click **📂 Load & Index** → wait for indexing to complete
2. Every subsequent run: click **⚡ Connect** → start chatting immediately

---

## 🌐 Language Directives

You can force the response language by appending a directive to your question:

| Directive | Response language |
|---|---|
| `answer in english` | English |
| `answer in indonesian` / `jawab dalam bahasa indonesia` | Indonesian |
| `answer in thai` / `ตอบเป็นภาษาไทย` | Thai |
| `answer in vietnamese` | Vietnamese |
| `answer in malay` | Malay |
| `answer in filipino` / `answer in tagalog` | Filipino |
| `answer in hindi` | Hindi |

**Example:** `ประชากรของประเทศไทยในปี 2567 มีกี่คน answer in english`

---

## 🧠 How It Works

```
User question
      │
      ▼
Redis cache lookup (exact SHA-256 → semantic cosine similarity)
      │ cache miss
      ▼
Hybrid Retriever
  ├── Keyword pass  → exact match on Excel row IDs (e.g. DSAI-NODE-1001)
  └── Vector pass   → Qdrant cosine similarity (BGE-M3, 1024-dim)
      │
      ▼
BGE Reranker v2-m3  →  top-5 chunks re-scored → top-3 kept
      │
      ▼
SEA-LION 8B LLM  →  answer generated with reranked context
      │
      ▼
Redis cache store (exact + semantic)
      │
      ▼
Answer returned to user
```

---

## 📊 LangSmith Tracing (Optional)

Get a free API key at [https://smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.

Set it in `main.py` / `app.py`:
```python
os.environ.setdefault("LANGCHAIN_API_KEY", "lsv2_pt_...")
```

Each query creates a trace showing:
- Retrieval scores (all Qdrant candidates)
- Reranker before/after scores and rank deltas
- LLM input context and output
- Total latency breakdown

---

## 🐳 Docker Services

| Service | Port | Purpose |
|---|---|---|
| Qdrant | 6333 (HTTP), 6334 (gRPC) | Vector store for document embeddings |
| Redis | 6379 | Exact + semantic answer cache |

**Useful Docker commands:**
```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Clear the Redis cache
docker exec rag-redis redis-cli FLUSHDB

# View logs
docker logs rag-redis
docker logs rag-qdrant
```


## 📁 Supported File Types

| Extension | Loader | Notes |
|---|---|---|
| `.pdf` | PyMuPDF | Falls back to Tesseract OCR for scanned PDFs |
| `.docx` | Unstructured | |
| `.txt` | TextLoader | UTF-8 encoding |
| `.md` | Unstructured | |
| `.xlsx` / `.xls` | pandas | Each row becomes a separate document for keyword search |
| `.csv` | CSVLoader | |
| `.png` / `.jpg` / `.jpeg` | Tesseract OCR | Requires Tesseract installed |

---

## 🔧 Configuration

All tunable parameters are at the top of `main.py`:

```python
COLLECTION_NAME     = "testing_the_hydre"   # Qdrant collection name
CHUNK_SIZE          = 800                    # Characters per chunk
CHUNK_OVERLAP       = 150                    # Overlap between chunks
MIN_RELEVANCE_SCORE = 0.40                   # Min Qdrant score to keep a chunk
MMR_FETCH_K         = 30                     # Candidates fetched from Qdrant
MMR_K               = 5                      # Kept after score filter
RERANKER_TOP_N      = 3                      # Kept after reranking
SEMANTIC_THRESHOLD  = 0.85                   # Cosine similarity for cache hit
CACHE_TTL           = 86400                  # Cache expiry in seconds (24h)
RERANKER_ENABLED    = True                   # Set False to disable reranker
RERANKER_USE_FP16   = True                   # Set False for CPU-only machines
```

---

## 🛠 Troubleshooting

**Qdrant / Redis not connecting**
```bash
docker compose up -d          # make sure containers are running
docker ps                     # verify rag-redis and rag-qdrant are listed
```

**Tesseract not found**
Update the path in `main.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Reranker disabled on startup**
Install FlagEmbedding:
```bash
pip install FlagEmbedding
```

**Cache returning wrong language**
Clear stale cache entries after updating the code:
```bash
docker exec rag-redis redis-cli FLUSHDB
```
