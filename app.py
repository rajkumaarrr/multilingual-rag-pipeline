"""
app.py — Streamlit UI for the Multilingual RAG Pipeline
Run with: streamlit run app.py

Startup is fast — only connects to Qdrant/Redis/models (cached once).
NO documents are loaded or indexed automatically.

Users choose one of these options from the sidebar:
  ⚡ Connect         — attach to an already-existing Qdrant index (fastest)
  📂 Load & Index   — load files from a directory and build a fresh index
  📤 Upload Files   — upload individual files and add them to the index
  🌐 Scrape URL     — scrape a website and add it to the index
"""

import os
import time
import json
import hashlib
from glob import glob
from pathlib import Path

import streamlit as st

# ── Page config — must be the very first Streamlit call ──────────
st.set_page_config(
    page_title="Multilingual RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LangSmith env vars — must be set before any LangChain import ─
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_API_KEY",     "YOUR_LANGSMITH_API_KEY")
os.environ.setdefault("LANGCHAIN_PROJECT",     "rag-sea-lion")

from setup_logger import setup_logger
logger = setup_logger()

import numpy as np
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\Admin\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

from langchain_core.documents import Document
from PIL import Image
from pdf2image import convert_from_path
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import redis as _redis
from web_scraper import scrape_url, is_valid_url

try:
    from FlagEmbedding import FlagReranker as _FlagReranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
COLLECTION_NAME     = "testing_the_hydre"
CACHE_TTL           = 60 * 60 * 24
SEMANTIC_THRESHOLD  = 0.85
MIN_RELEVANCE_SCORE = 0.40
MMR_FETCH_K         = 30
MMR_K               = 5
RERANKER_MODEL      = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N      = 3
RERANKER_USE_FP16   = True
DATA_DIR            = "./data"
SUPPORTED           = {".pdf", ".txt", ".docx", ".md", ".xlsx", ".xls",
                       ".csv", ".png", ".jpeg", ".jpg"}

REDIS_PREFIX_EXACT  = "rag:exact:"
REDIS_PREFIX_VEC    = "rag:vec:"
REDIS_VEC_INDEX_KEY = "rag:vec:index"

_LANG_DIRECTIVES = sorted({
    "answer in english", "answer in indonesian", "answer in thai",
    "answer in vietnamese", "answer in tagalog", "answer in malay",
    "answer in filipino", "answer in hindi", "answer in tamil",
    "answer in bengali", "answer in chinese", "answer in japanese",
    "in english", "in indonesian", "in thai",
    "in vietnamese", "in tagalog", "in malay", "in filipino",
    "jawab dalam bahasa indonesia", "jawab dalam bahasa inggris",
    "ตอบเป็นภาษาไทย", "ตอบเป็นภาษาอังกฤษ",
}, key=len, reverse=True)

import re as _re
_STRUCTURED_ID_PATTERN = _re.compile(
    r"(?:[A-Z]{2,}-[A-Z0-9]{2,}-[0-9]+|[A-Z]{2,}-[0-9]{3,}|[A-Z][0-9]{4,})",
    _re.IGNORECASE,
)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", "।", "॥", "ฯ", ". ", " ", ""]
)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _strip_lang_directive(text: str) -> str:
    lowered = text.strip().lower()
    for d in _LANG_DIRECTIVES:
        if lowered.endswith(d):
            lowered = lowered[:-len(d)].rstrip(" ,;")
            break
    return lowered


def _extract_directive(text: str) -> str:
    """Unicode-safe: uses endswith(), never string slicing by len()."""
    lowered = text.strip().lower()
    for d in _LANG_DIRECTIVES:
        if lowered.endswith(d):
            return d
    return ""


def cosine_similarity(a, b):
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0


def make_exact_key(q):
    return REDIS_PREFIX_EXACT + hashlib.sha256(q.strip().lower().encode()).hexdigest()


def make_vec_key(q):
    return REDIS_PREFIX_VEC + hashlib.sha256(q.strip().lower().encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════
class FilteredRetriever(BaseRetriever):
    vectorstore: QdrantVectorStore
    excel_docs:  list
    k:           int   = MMR_K
    fetch_k:     int   = MMR_FETCH_K
    min_score:   float = MIN_RELEVANCE_SCORE

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _extract_tokens(query):
        clean = _strip_lang_directive(query)
        if not _STRUCTURED_ID_PATTERN.search(clean):
            return []
        codes   = _STRUCTURED_ID_PATTERN.findall(clean)
        numbers = _re.findall(r"[0-9]{3,}", clean)
        tokens  = [t.lower() for t in codes] + numbers
        seen, unique = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t); unique.append(t)
        return unique

    def _keyword_search(self, query):
        if not self.excel_docs: return []
        tokens = self._extract_tokens(query)
        if not tokens: return []
        scored = []
        for doc in self.excel_docs:
            score = sum(1 for t in tokens if t in doc.page_content.lower())
            if score > 0: scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[RETRIEVAL] Keyword: {len(scored)}/{len(self.excel_docs)} rows matched")
        return [d for d, _ in scored[:self.k]]

    def _vector_search(self, query):
        t0      = time.perf_counter()
        results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        if not results:
            logger.warning("[RETRIEVAL] Qdrant returned 0 results")
            return []
        logger.log("TIMING", f"[TIMER] {'Qdrant search':<45} {round(time.perf_counter()-t0,3)}s")
        filtered = [(d, s) for d, s in results if s >= self.min_score]
        if not filtered:
            filtered = results
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def _get_relevant_documents(self, query, *, run_manager=None):
        kw_docs  = self._keyword_search(query)
        vec_docs = [d for d, _ in self._vector_search(query)[:self.k]]
        seen, merged = set(), []
        for doc in kw_docs + vec_docs:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h); merged.append(doc)
            if len(merged) >= self.k: break
        kw_n  = sum(1 for d in merged if d.metadata.get("doc_type") == "excel_row")
        logger.info(f"[RETRIEVAL] {len(merged)} chunks → {kw_n} keyword + {len(merged)-kw_n} vector")
        return merged


# ══════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════
class RAGCache:
    def __init__(self, r, emb):
        self.r = r; self.emb = emb; self._mem = {}

    def _embed(self, text):
        k = text.strip().lower()
        if k not in self._mem:
            t0 = time.perf_counter()
            self._mem[k] = self.emb.embed_query(text)
            logger.log("TIMING", f"[TIMER] {'BGE-M3 embed':<45} {round(time.perf_counter()-t0,3)}s")
        return self._mem[k]

    def get(self, question) -> str | None:
        # Exact
        raw = self.r.get(make_exact_key(question))
        if raw:
            logger.info(f"[CACHE] Exact HIT")
            return raw.decode("utf-8")
        # Semantic
        qv    = self._embed(question)
        q_dir = _extract_directive(question)
        keys  = self.r.lrange(REDIS_VEC_INDEX_KEY, 0, -1)
        if not keys: return None
        best_score, best_ans = -1.0, None
        for rk in keys:
            k  = rk.decode() if isinstance(rk, bytes) else rk
            rv = self.r.get(k)
            if not rv: continue
            try:
                e          = json.loads(rv.decode())
                cached_dir = _extract_directive(e.get("question", ""))
                if q_dir != cached_dir: continue          # directive guard
                score = cosine_similarity(qv, e["embedding"])
                if score > best_score:
                    best_score = score
                    best_ans   = e.get("answer") if score >= SEMANTIC_THRESHOLD else None
            except Exception:
                pass
        if best_ans:
            logger.info(f"[CACHE] Semantic HIT sim={best_score:.4f}")
        else:
            logger.info(f"[CACHE] Semantic MISS best_sim={best_score:.4f}")
        return best_ans

    def set(self, question, answer):
        if CACHE_TTL:
            self.r.setex(make_exact_key(question), CACHE_TTL, answer)
        else:
            self.r.set(make_exact_key(question), answer)
        emb  = self._embed(question)
        vk   = make_vec_key(question)
        payload = json.dumps({"question": question, "embedding": emb, "answer": answer})
        if CACHE_TTL:
            self.r.setex(vk, CACHE_TTL, payload)
        else:
            self.r.set(vk, payload)
        existing = [k.decode() if isinstance(k, bytes) else k
                    for k in self.r.lrange(REDIS_VEC_INDEX_KEY, 0, -1)]
        if vk not in existing:
            self.r.rpush(REDIS_VEC_INDEX_KEY, vk)
            if CACHE_TTL: self.r.expire(REDIS_VEC_INDEX_KEY, CACHE_TTL)
        logger.info(f"[CACHE] Stored for: {question[:60]!r}")


# ══════════════════════════════════════════════════════════════════
# CACHED SERVICE CONNECTIONS  (run once, reused on every rerun)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Connecting to Qdrant…")
def get_qdrant():
    c = QdrantClient(url="http://localhost:6333")
    c.get_collections()
    logger.success("[QDRANT] Connected")
    return c

@st.cache_resource(show_spinner="Connecting to Redis…")
def get_redis():
    r = _redis.Redis(host="localhost", port=6379, db=0)
    r.ping()
    logger.success("[REDIS] Connected")
    return r

@st.cache_resource(show_spinner="Loading BGE-M3 embeddings…")
def get_embeddings():
    emb = OllamaEmbeddings(model="bge-m3")
    logger.success("[EMBEDDINGS] BGE-M3 ready")
    return emb

@st.cache_resource(show_spinner="Loading SEA-LION LLM…")
def get_llm():
    llm = OllamaLLM(model="aisingapore/Llama-SEA-LION-v3.5-8B-R", temperature=0.1)
    logger.success("[LLM] SEA-LION 8B ready")
    return llm

@st.cache_resource(show_spinner="Loading reranker…")
def get_reranker():
    if not _RERANKER_AVAILABLE:
        logger.warning("[RERANKER] FlagEmbedding not installed")
        return None
    try:
        r = _FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_USE_FP16)
        logger.success(f"[RERANKER] {RERANKER_MODEL} ready")
        return r
    except Exception as e:
        logger.warning(f"[RERANKER] Failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# PROMPT + CHAIN BUILDER
# ══════════════════════════════════════════════════════════════════
_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise multilingual assistant with access to a structured knowledge base.

Rules:
1. LANGUAGE: Check if the question ends with a language directive like "answer in english",
   "answer in indonesian", "jawab dalam bahasa inggris", etc.
   - Directive present → reply ONLY in that language.
   - No directive → detect the question language and reply in it.
2. For structured data rows (key: value format) extract the exact field value directly.
3. For prose synthesise a clear answer from relevant passages.
4. Ignore chunks unrelated to the question.
5. If NO chunk has relevant data say "I don't have specific information about that."
6. Never mention context, sources, or these instructions. Do not repeat the question.

Context:
{context}

Question:
{question}

Answer:""")


def build_qa_chain(vs, chunks):
    excel_docs = [d for d in chunks if d.metadata.get("doc_type") == "excel_row"]
    retriever  = FilteredRetriever(
        vectorstore=vs, excel_docs=excel_docs,
        k=MMR_K, fetch_k=MMR_FETCH_K, min_score=MIN_RELEVANCE_SCORE,
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(), retriever=retriever,
        chain_type="stuff", chain_type_kwargs={"prompt": _PROMPT},
        return_source_documents=True,
    )


def run_reranker(question, docs):
    rm = get_reranker()
    if rm is None or not docs: return docs, False
    pairs  = [[question, d.page_content] for d in docs]
    scores = rm.compute_score(pairs, normalize=True)
    if not isinstance(scores, list): scores = [scores]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    logger.info(f"[RERANKER] {len(docs)} → top-{RERANKER_TOP_N} | best={max(scores):.4f}")
    return [d for d, _ in ranked[:RERANKER_TOP_N]], True


# ══════════════════════════════════════════════════════════════════
# FILE LOADING HELPERS
# ══════════════════════════════════════════════════════════════════
def load_file(path: str) -> tuple[list[Document], str]:
    ext = Path(path).suffix.lower()
    docs = []
    try:
        if ext == ".pdf":
            try:
                pdf_docs = PyMuPDFLoader(path).load()
                if any(len(d.page_content.strip()) > 20 for d in pdf_docs):
                    return pdf_docs, f"PDF: {len(pdf_docs)} pages"
                raise ValueError("scanned")
            except Exception:
                images = convert_from_path(path, dpi=300, poppler_path=POPPLER_PATH)
                text   = "\n".join(pytesseract.image_to_string(img) for img in images)
                return [Document(page_content=text, metadata={"source": path})], \
                       f"PDF (OCR): {len(images)} pages"
        elif ext == ".txt":
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif ext == ".docx":
            docs.extend(UnstructuredWordDocumentLoader(path).load())
        elif ext == ".md":
            docs.extend(UnstructuredMarkdownLoader(path).load())
        elif ext == ".csv":
            docs.extend(CSVLoader(file_path=path, encoding="utf-8").load())
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = pytesseract.image_to_string(Image.open(path))
            return [Document(page_content=text, metadata={"source": path})], "Image (OCR)"
        elif ext in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet).fillna("")
                for idx, row in df.iterrows():
                    content = "\n".join(f"{c}: {row[c]}" for c in df.columns)
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "sheet": sheet,
                                  "row_index": int(idx), "doc_type": "excel_row"}
                    ))
            return docs, f"Excel: {len(docs)} rows"
        return docs, f"{ext.upper()}: {len(docs)} doc(s)"
    except Exception as e:
        return [], f"FAILED: {e}"


def chunk_docs(docs):
    result = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src.endswith((".xlsx", ".xls")) or doc.metadata.get("doc_type") == "excel_row":
            result.append(doc)
        else:
            result.extend(_SPLITTER.split_documents([doc]))
    return result


def upsert_chunks(new_chunks, force_recreate=False):
    """Embed and upsert chunks into Qdrant. Returns the QdrantVectorStore."""
    qdrant = get_qdrant()
    emb    = get_embeddings()
    exists = qdrant.collection_exists(COLLECTION_NAME)
    if not exists:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        )
    return QdrantVectorStore.from_documents(
        new_chunks, emb,
        url="http://localhost:6333",
        collection_name=COLLECTION_NAME,
        force_recreate=force_recreate,
    )


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    "pipeline_ready": False,
    "vectorstore":    None,
    "cache":          None,
    "qa_chain":       None,
    "all_chunks":     [],
    "scraped_urls":   set(),
    "messages":       [],
    "query_count":    0,
    "load_log":       [],
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _mark_ready(vs, chunks):
    """Called after any indexing operation to activate the chat."""
    r   = get_redis()
    emb = get_embeddings()
    st.session_state.vectorstore    = vs
    st.session_state.cache          = RAGCache(r, emb)
    st.session_state.all_chunks     = list(chunks)
    st.session_state.qa_chain       = build_qa_chain(vs, chunks)
    st.session_state.pipeline_ready = True


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔍 RAG Pipeline")
    st.caption("Multilingual · SEA-LION · BGE-M3")

    # ── Service status ────────────────────────────────────────────
    st.subheader("Services")
    c1, c2 = st.columns(2)
    try:
        get_qdrant().get_collections(); c1.success("Qdrant ✓")
    except Exception:
        c1.error("Qdrant ✗")
    try:
        get_redis().ping(); c2.success("Redis ✓")
    except Exception:
        c2.error("Redis ✗")

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # OPTION 1 — Connect to existing index (fastest, no re-indexing)
    # ══════════════════════════════════════════════════════════════
    st.subheader("⚡ Connect to Existing Index")
    st.caption("Already indexed? Connect without re-loading any files.")

    if st.button("Connect", use_container_width=True, key="btn_connect"):
        with st.spinner("Connecting to existing index…"):
            try:
                emb = get_embeddings()
                vs  = QdrantVectorStore(
                    client=get_qdrant(),
                    collection_name=COLLECTION_NAME,
                    embedding=emb,
                )
                _mark_ready(vs, [])   # no local chunks — keyword search disabled
                logger.success("[APP] Connected to existing Qdrant index")
                st.success("Connected! Start chatting.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # OPTION 2 — Load & Index from a directory
    # ══════════════════════════════════════════════════════════════
    st.subheader("📂 Load & Index Directory")
    data_dir_input = st.text_input("Data directory", value=DATA_DIR, key="data_dir")
    force_reindex  = st.checkbox("Force full reindex", value=False, key="force_reindex")

    if st.button("Load & Index", use_container_width=True, key="btn_load"):
        with st.spinner("Loading documents…"):
            file_paths = [
                p for p in glob(os.path.join(data_dir_input, "**", "*"), recursive=True)
                if os.path.isfile(p) and Path(p).suffix.lower() in SUPPORTED
            ]
            if not file_paths:
                st.warning(f"No supported files found in '{data_dir_input}'")
                st.stop()

            all_docs, load_log = [], []
            prog = st.progress(0, text="Loading files…")
            for i, path in enumerate(file_paths):
                docs, msg = load_file(path)
                all_docs.extend(docs)
                load_log.append(f"{'✓' if docs else '✗'} {Path(path).name} — {msg}")
                prog.progress((i+1) / len(file_paths),
                              text=f"Loading {Path(path).name}…")
            prog.empty()

            new_chunks = chunk_docs(all_docs)
            st.info(f"Chunked into {len(new_chunks)} pieces. Embedding & indexing…")

            t0 = time.perf_counter()
            vs = upsert_chunks(new_chunks, force_recreate=force_reindex)
            elapsed = round(time.perf_counter() - t0, 2)

            _mark_ready(vs, new_chunks)
            st.session_state.load_log = load_log
            logger.success(f"[APP] Indexed {len(new_chunks)} chunks in {elapsed}s")

        st.success(f"✓ {len(new_chunks)} chunks indexed in {elapsed}s")

    if st.session_state.load_log:
        with st.expander("Load details", expanded=False):
            for line in st.session_state.load_log:
                st.text(line)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # OPTION 3 — Upload individual files
    # ══════════════════════════════════════════════════════════════
    st.subheader("📤 Upload Files")
    uploaded = st.file_uploader(
        "Drop files here", accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md", "xlsx", "xls", "csv", "png", "jpg", "jpeg"],
        key="uploader",
    )

    if uploaded and st.button("Index Uploaded Files", use_container_width=True, key="btn_upload"):
        with st.spinner("Processing uploads…"):
            upload_dir = Path("./data/_uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            new_docs = []
            for uf in uploaded:
                save_path = upload_dir / uf.name
                save_path.write_bytes(uf.read())
                docs, msg = load_file(str(save_path))
                new_docs.extend(docs)
                st.write(f"✓ {uf.name} — {msg}")

            new_chunks = chunk_docs(new_docs)
            t0 = time.perf_counter()
            vs = upsert_chunks(new_chunks, force_recreate=False)
            elapsed = round(time.perf_counter() - t0, 2)

            # Merge with existing chunks if pipeline was already ready
            all_chunks = st.session_state.all_chunks + new_chunks
            _mark_ready(vs, all_chunks)
            logger.success(f"[APP] Upload: indexed {len(new_chunks)} chunks in {elapsed}s")

        st.success(f"✓ {len(new_chunks)} chunks indexed in {elapsed}s")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────
    st.subheader("Settings")
    use_cache    = st.toggle("Use answer cache",  value=True)
    use_reranker = st.toggle("Use reranker",       value=True)
    show_sources = st.toggle("Show sources",       value=True)
    show_timing  = st.toggle("Show response time", value=True)

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────
    if st.session_state.pipeline_ready:
        st.subheader("Index Stats")
        st.metric("Chunks indexed",   len(st.session_state.all_chunks))
        st.metric("Queries answered", st.session_state.query_count)
        st.metric("Excel rows", sum(
            1 for d in st.session_state.all_chunks
            if d.metadata.get("doc_type") == "excel_row"
        ))

    if st.button("🗑 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
st.title("🌏 Multilingual RAG Chatbot")
st.caption("English · Thai · Hindi · Tamil · Bengali · Filipino · Malay · Vietnamese · Indonesian")

if not st.session_state.pipeline_ready:
    st.info(
        "👈 Choose an option in the sidebar to get started:\n\n"
        "- **⚡ Connect** — attach to an existing Qdrant index instantly\n"
        "- **📂 Load & Index** — load documents from a folder and build an index\n"
        "- **📤 Upload Files** — upload files directly and index them",
        icon="ℹ️",
    )

# ── OPTION 4 — Scrape a website ───────────────────────────────────
with st.expander("🌐 Scrape a Website", expanded=False):
    url_col, btn_col = st.columns([4, 1])
    url_input = url_col.text_input("URL", label_visibility="collapsed",
                                   placeholder="https://example.com")
    if btn_col.button("Scrape", use_container_width=True):
        if not is_valid_url(url_input):
            st.error("Please enter a valid http/https URL.")
        elif not st.session_state.pipeline_ready:
            st.warning("Connect to an index first (use sidebar options).")
        else:
            with st.spinner(f"Scraping {url_input} …"):
                t0       = time.perf_counter()
                web_docs = scrape_url(url_input, st.session_state.scraped_urls, max_depth=1)
                if not web_docs:
                    st.warning("No content extracted.")
                else:
                    web_chunks = []
                    for doc in web_docs:
                        if doc.metadata.get("type") == "table":
                            web_chunks.append(doc)
                        else:
                            web_chunks.extend(_SPLITTER.split_documents([doc]))
                    vs = upsert_chunks(web_chunks, force_recreate=False)
                    all_chunks = st.session_state.all_chunks + web_chunks
                    _mark_ready(vs, all_chunks)
                    elapsed = round(time.perf_counter() - t0, 2)
                    pages   = sum(1 for d in web_docs if d.metadata.get("type") == "webpage")
                    tables  = sum(1 for d in web_docs if d.metadata.get("type") == "table")
                    st.success(
                        f"✓ {len(web_chunks)} chunks in {elapsed}s "
                        f"— {pages} pages, {tables} tables")

st.divider()

# ── Chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"📄 {src}")
        if msg.get("timing") and show_timing:
            st.caption(f"⏱ {msg['timing']}")
        if msg.get("cached"):
            st.caption("⚡ Served from cache")
        if msg.get("reranked"):
            st.caption("🎯 Reranker applied")

# ── Chat input ────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask a question… (prefix ! to bypass cache)",
    disabled=not st.session_state.pipeline_ready,
):
    bypass_cache = question.startswith("!")
    clean_q      = question[1:].strip() if bypass_cache else question

    if not clean_q:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.query_count += 1
    qnum = st.session_state.query_count
    logger.info("─" * 60)
    logger.info(f"[CHAT] Q#{qnum}: {clean_q!r} | bypass={bypass_cache}")

    with st.chat_message("user"):
        st.markdown(clean_q)
    st.session_state.messages.append({"role": "user", "content": clean_q})

    with st.chat_message("assistant"):
        status_ph = st.empty()
        answer_ph = st.empty()

        t_start       = time.perf_counter()
        cached_answer = None
        sources       = []
        reranked      = False
        timing        = ""

        # ── Cache lookup ──────────────────────────────────────────
        if not bypass_cache and st.session_state.cache:
            status_ph.info("🔍 Checking cache…")
            cached_answer = st.session_state.cache.get(clean_q)

        if cached_answer:
            answer = cached_answer
            status_ph.empty()
            answer_ph.markdown(answer)
            wall   = round(time.perf_counter() - t_start, 3)
            timing = f"{wall}s (cached)"
            st.caption("⚡ Served from cache")
            if show_timing: st.caption(f"⏱ {timing}")
            logger.info(f"[CHAT] Cache hit in {wall}s")

        else:
            # ── Retrieval ─────────────────────────────────────────
            status_ph.info("🔎 Retrieving relevant chunks…")
            t0          = time.perf_counter()
            result      = st.session_state.qa_chain.invoke({"query": clean_q})
            source_docs = result["source_documents"]
            logger.info(f"[RETRIEVAL] Done in {round(time.perf_counter()-t0,2)}s")

            # ── Reranker ──────────────────────────────────────────
            if use_reranker:
                status_ph.info("🎯 Reranking results…")
                top_docs, reranked = run_reranker(clean_q, source_docs)
                if reranked:
                    status_ph.info("💬 Generating answer with reranked context…")
                    t_llm = time.perf_counter()
                    rr    = st.session_state.qa_chain.combine_documents_chain.invoke({
                        "input_documents": top_docs, "question": clean_q,
                    })
                    logger.log("TIMING",
                        f"[TIMER] {'LLM (reranked)':<45} {round(time.perf_counter()-t_llm,3)}s")
                    answer      = rr.get("output_text", result["result"])
                    source_docs = top_docs
                else:
                    answer = result["result"]
            else:
                status_ph.info("💬 Generating answer…")
                answer = result["result"]

            if "<think>" in answer:
                think  = answer.split("</think>")[0].replace("<think>", "").strip()
                answer = answer.split("</think>")[-1].strip()
                logger.debug(f"[LLM] <think> stripped ({len(think)} chars)")

            status_ph.empty()
            answer_ph.markdown(answer)

            sources = list(dict.fromkeys(
                d.metadata.get("source", "unknown") for d in source_docs
            ))
            if show_sources and sources:
                with st.expander("Sources", expanded=False):
                    for src in sources:
                        st.caption(f"📄 {src}")
            if reranked:
                st.caption("🎯 Reranker applied")

            wall   = round(time.perf_counter() - t_start, 2)
            timing = f"{wall}s"
            if show_timing: st.caption(f"⏱ {timing}")

            logger.info(f"[LLM] Answer: {answer[:150]!r}")
            logger.info(f"[CHAT] Total: {wall}s")

            if st.session_state.cache:
                try:
                    st.session_state.cache.set(clean_q, answer)
                except Exception as ce:
                    logger.warning(f"[CACHE] Store failed: {ce}")

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "sources":  sources,
        "timing":   timing,
        "cached":   bool(cached_answer),
        "reranked": reranked,
    })