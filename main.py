import os
import time
import json
import hashlib
import numpy as np
from glob import glob
from contextlib import contextmanager

# ── LangSmith (must be set before any LangChain import) ───────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_API_KEY",    "YOUR_LANGSMITH_KEY")
os.environ.setdefault("LANGCHAIN_PROJECT",    "rag-sea-lion")

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import pandas as pd
from langchain_core.documents import Document
from PIL import Image
from pdf2image import convert_from_path

from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from setup_logger import setup_logger

import redis
from langsmith import traceable

try:
    from FlagEmbedding import FlagReranker as _FlagReranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

from web_scraper import scrape_url, is_valid_url

logger = setup_logger()

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
COLLECTION_NAME     = "testing_multilingual"
CACHE_TTL           = 60 * 60 * 24
SEMANTIC_THRESHOLD  = 0.85
MIN_RELEVANCE_SCORE = 0.40
MMR_FETCH_K         = 30
MMR_K               = 5
RERANKER_ENABLED    = True
RERANKER_MODEL      = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N      = 3
RERANKER_USE_FP16   = True
SUPPORTED           = {".pdf", ".txt", ".docx", ".md", ".xlsx", ".xls",
                       ".csv", ".png", ".jpeg", ".jpg"}

REDIS_PREFIX_EXACT  = "rag:exact:"
REDIS_PREFIX_VEC    = "rag:vec:"
REDIS_VEC_INDEX_KEY = "rag:vec:index"

# Sorted longest-first so "answer in english" always matched before "in english"
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


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.log("TIMING", f"[TIMER] {label:<45} {elapsed:.3f}s")


def _strip_lang_directive(text: str) -> str:
    """Remove the trailing language directive from a question string."""
    lowered = text.strip().lower()
    for d in _LANG_DIRECTIVES:          # already sorted longest-first
        if lowered.endswith(d):
            lowered = lowered[:-len(d)].rstrip(" ,;")
            break
    return lowered


def _extract_directive(text: str) -> str:
    """
    Return the trailing language directive, lowercased, or '' if none.

    e.g.  "What is X? answer in english"        -> "answer in english"
          "ประชากร...มีกี่คน answer in english"  -> "answer in english"
          "What is X?"                           -> ""

    Unicode-safe: uses endswith() on sorted directives, NOT string slicing
    by len() which breaks with Thai/CJK text.
    """
    lowered = text.strip().lower()
    for d in _LANG_DIRECTIVES:          # longest-first prevents partial match
        if lowered.endswith(d):
            return d
    return ""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def make_exact_key(question: str) -> str:
    return REDIS_PREFIX_EXACT + hashlib.sha256(question.strip().lower().encode()).hexdigest()


def make_vec_key(question: str) -> str:
    return REDIS_PREFIX_VEC + hashlib.sha256(question.strip().lower().encode()).hexdigest()


import re as _re

_KEYWORD_STOP_WORDS: frozenset = frozenset({
    "answer", "english", "indonesian", "thai", "vietnamese",
    "tagalog", "filipino", "malay", "chinese", "japanese", "korean",
    "the", "for", "what", "when", "where", "who", "how", "why",
    "is", "are", "was", "were", "did", "does", "and", "or",
    "in", "on", "at", "of", "to", "a", "an",
    "please", "tell", "me", "give", "find", "show",
    "bahasa", "dalam", "jawab", "apa", "kapan", "siapa",
    "dari", "yang", "dengan", "ada", "ini", "itu",
    "been", "have", "has", "had", "this", "that", "with",
    "year", "quarter", "month", "week", "time", "about",
})

_STRUCTURED_ID_PATTERN = _re.compile(
    r"(?:[A-Z]{2,}-[A-Z0-9]{2,}-[0-9]+|[A-Z]{2,}-[0-9]{3,}|[A-Z][0-9]{4,})",
    _re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════
class FilteredRetriever(BaseRetriever):
    """Hybrid retriever: keyword-exact for Excel rows + vector for prose."""

    vectorstore:  QdrantVectorStore
    excel_docs:   list
    k:            int   = MMR_K
    fetch_k:      int   = MMR_FETCH_K
    min_score:    float = MIN_RELEVANCE_SCORE

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _extract_tokens(query: str) -> list[str]:
        clean = _strip_lang_directive(query)
        if not _STRUCTURED_ID_PATTERN.search(clean):
            logger.debug("[RETRIEVAL] No structured ID — skipping keyword pass")
            return []
        codes   = _STRUCTURED_ID_PATTERN.findall(clean)
        numbers = _re.findall(r"[0-9]{3,}", clean)
        tokens  = [t.lower() for t in codes] + numbers
        seen, unique = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t); unique.append(t)
        return unique

    def _keyword_search(self, query: str) -> list[Document]:
        if not self.excel_docs:
            return []
        tokens = self._extract_tokens(query)
        if not tokens:
            return []
        logger.debug(f"[RETRIEVAL] Keyword tokens: {tokens}")
        scored = []
        for doc in self.excel_docs:
            score = sum(1 for t in tokens if t in doc.page_content.lower())
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        for doc, sc in scored[:5]:
            logger.debug(f"[RETRIEVAL] KW hit score={sc} | {doc.page_content.strip()[:120].replace(chr(10),' ')!r}")
        logger.info(f"[RETRIEVAL] Keyword: {len(scored)}/{len(self.excel_docs)} rows matched")
        return [doc for doc, _ in scored[:self.k]]

    def _vector_search(self, query: str) -> list[tuple]:
        with timer("Qdrant similarity_search_with_score"):
            results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        if not results:
            logger.warning("[RETRIEVAL] Qdrant returned 0 results")
            return []
        logger.debug("[RETRIEVAL] ── All vector scores ──────────────────────")
        for rank, (doc, score) in enumerate(results, 1):
            src   = doc.metadata.get("source", "?")
            sheet = doc.metadata.get("sheet", "")
            label = f" [sheet:{sheet}]" if sheet else ""
            logger.debug(f"[RETRIEVAL]   #{rank:02d} score={score:.4f}  {src}{label}")
        filtered = [(doc, score) for doc, score in results if score >= self.min_score]
        if not filtered:
            logger.warning(
                f"[RETRIEVAL] All scores below min_score={self.min_score} "
                f"(top={results[0][1]:.4f}). Using top-{self.k} as fallback."
            )
            filtered = results
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None,
    ) -> list[Document]:
        kw_docs     = self._keyword_search(query)
        vec_results = self._vector_search(query)
        vec_docs    = [doc for doc, _ in vec_results[:self.k]]
        seen_hashes: set = set()
        merged: list[Document] = []
        for doc in kw_docs + vec_docs:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h); merged.append(doc)
            if len(merged) >= self.k:
                break
        kw_count  = sum(1 for d in merged if d.metadata.get("doc_type") == "excel_row")
        vec_count = len(merged) - kw_count
        logger.info(f"[RETRIEVAL] {len(merged)} chunks → {kw_count} keyword + {vec_count} vector")
        return merged


# ══════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════
class RAGCache:
    def __init__(self, redis_client: redis.Redis, embeddings: OllamaEmbeddings):
        self.r          = redis_client
        self.emb        = embeddings
        self._emb_cache: dict = {}

    def _embed(self, text: str) -> list[float]:
        key = text.strip().lower()
        if key not in self._emb_cache:
            with timer(f"BGE-M3 embed ({len(text)} chars)"):
                self._emb_cache[key] = self.emb.embed_query(text)
            logger.debug(f"[CACHE] Embedding computed for: {text[:60]!r}")
        else:
            logger.debug(f"[CACHE] Embedding from in-memory cache: {text[:60]!r}")
        return self._emb_cache[key]

    def _exact_get(self, question: str) -> str | None:
        key = make_exact_key(question)
        with timer("Redis GET (exact cache)"):
            raw = self.r.get(key)
        if raw:
            logger.info(f"[CACHE] Exact HIT  → key=...{key[-8:]}")
            return raw.decode("utf-8")
        logger.debug(f"[CACHE] Exact MISS → key=...{key[-8:]}")
        return None

    def _exact_set(self, question: str, answer: str) -> None:
        key = make_exact_key(question)
        with timer("Redis SET (exact cache)"):
            if CACHE_TTL:
                self.r.setex(key, CACHE_TTL, answer)
            else:
                self.r.set(key, answer)
        logger.debug(f"[CACHE] Exact stored → key=...{key[-8:]} | TTL={CACHE_TTL}s")

    def _semantic_get(self, question: str) -> str | None:
        query_vec = self._embed(question)
        # Unicode-safe directive extraction — no string slicing
        q_dir     = _extract_directive(question)

        with timer("Redis LRANGE (fetch vec index)"):
            all_keys = self.r.lrange(REDIS_VEC_INDEX_KEY, 0, -1)
        if not all_keys:
            logger.debug("[CACHE] Semantic cache is empty")
            return None

        logger.debug(f"[CACHE] Scanning {len(all_keys)} vectors | q_dir={q_dir!r}")
        best_score, best_answer, best_key = -1.0, None, None
        scan_start = time.perf_counter()

        for raw_key in all_keys:
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            raw = self.r.get(key)
            if not raw:
                continue
            try:
                entry    = json.loads(raw.decode("utf-8"))
                cached_q = entry["question"]

                # ── Directive guard (Unicode-safe) ────────────────
                # Directives must match exactly.
                # "X? answer in english" (q_dir="answer in english")
                # vs cached "X?" (cached_dir="")  →  SKIP  ✓
                cached_dir = _extract_directive(cached_q)
                if q_dir != cached_dir:
                    logger.debug(
                        f"[CACHE]   SKIP directive mismatch: "
                        f"want={q_dir!r} cached={cached_dir!r} | q={cached_q[:50]!r}"
                    )
                    continue

                score = cosine_similarity(query_vec, entry["embedding"])
                logger.debug(f"[CACHE]   sim={score:.4f} | q={cached_q[:60]!r}")
                if score > best_score:
                    best_score  = score
                    best_key    = key
                    best_answer = entry["answer"] if score >= SEMANTIC_THRESHOLD else None
            except Exception as e:
                logger.warning(f"[CACHE] Corrupt entry {key}: {e}")

        scan_elapsed = time.perf_counter() - scan_start
        logger.log("TIMING",
            f"[TIMER] {'Semantic cache scan':<45} {scan_elapsed:.3f}s ({len(all_keys)} vectors)")
        if best_answer:
            logger.info(
                f"[CACHE] Semantic HIT  → sim={best_score:.4f} "
                f"(threshold={SEMANTIC_THRESHOLD}) | key=...{best_key[-8:]}")
        else:
            logger.info(
                f"[CACHE] Semantic MISS → best_sim={best_score:.4f} "
                f"(threshold={SEMANTIC_THRESHOLD})")
        return best_answer

    def _semantic_set(self, question: str, answer: str) -> None:
        embedding = self._embed(question)
        vec_key   = make_vec_key(question)
        entry = json.dumps({"question": question, "embedding": embedding, "answer": answer})
        with timer("Redis SET (semantic cache)"):
            if CACHE_TTL:
                self.r.setex(vec_key, CACHE_TTL, entry)
            else:
                self.r.set(vec_key, entry)
        existing = [
            k.decode() if isinstance(k, bytes) else k
            for k in self.r.lrange(REDIS_VEC_INDEX_KEY, 0, -1)
        ]
        if vec_key not in existing:
            self.r.rpush(REDIS_VEC_INDEX_KEY, vec_key)
            if CACHE_TTL:
                self.r.expire(REDIS_VEC_INDEX_KEY, CACHE_TTL)
        logger.debug(f"[CACHE] Semantic stored → key=...{vec_key[-8:]} | TTL={CACHE_TTL}s")

    def get(self, question: str) -> str | None:
        logger.debug(f"[CACHE] Looking up: {question!r}")
        answer = self._exact_get(question)
        if answer:
            return answer
        return self._semantic_get(question)

    def set(self, question: str, answer: str) -> None:
        logger.debug(f"[CACHE] Storing answer for: {question!r}")
        self._exact_set(question, answer)
        self._semantic_set(question, answer)
        logger.info("[CACHE] Answer stored (exact + semantic)")


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════
_pipeline_start = time.perf_counter()

logger.info("=" * 60)
logger.info("RAG pipeline starting up")
logger.info(f"Collection: {COLLECTION_NAME}")
logger.info("=" * 60)

logger.info("[QDRANT] Connecting to http://localhost:6333 ...")
with timer("Qdrant connection"):
    try:
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        logger.success("[QDRANT] Connected")
    except Exception as e:
        logger.critical(f"[QDRANT] Failed: {e}"); raise

logger.info("[REDIS] Connecting to localhost:6379 ...")
with timer("Redis connection"):
    try:
        redis_client = redis.Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        logger.success("[REDIS] Connected")
    except Exception as e:
        logger.critical(f"[REDIS] Failed: {e}"); raise

logger.info("[LANGSMITH] Validating connection...")
try:
    from langsmith import Client as LangSmithClient
    _ls_client  = LangSmithClient()
    _ls_project = os.environ.get("LANGCHAIN_PROJECT", "rag-sea-lion")
    if not _ls_client.has_project(_ls_project):
        _ls_client.create_project(_ls_project, description="RAG SEA-LION pipeline")
    logger.success(
        f"[LANGSMITH] Connected — project={_ls_project!r} | "
        f"https://smith.langchain.com/projects/{_ls_project}"
    )
except Exception as ls_err:
    logger.warning(f"[LANGSMITH] Could not connect ({ls_err}). Tracing disabled.")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Load documents ────────────────────────────────────────────────
logger.info("[LOADER] Discovering files in ./data ...")
files_dir  = "./data"
all_paths  = glob(os.path.join(files_dir, "**", "*"), recursive=True)
file_paths = [p for p in all_paths if os.path.isfile(p)]
logger.info(f"[LOADER] Found {len(file_paths)} files")

docs, skipped, failed = [], [], []
_load_start = time.perf_counter()

for path in file_paths:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        skipped.append(path)
        logger.debug(f"[LOADER] Skipped (unsupported): {path}")
        continue

    t_file = time.perf_counter()
    before = len(docs)
    try:
        if ext == ".pdf":
            try:
                pdf_docs = PyMuPDFLoader(path).load()
                if any(len(d.page_content.strip()) > 20 for d in pdf_docs):
                    docs.extend(pdf_docs)
                    logger.debug(f"[LOADER] PDF text-based: {len(pdf_docs)} pages")
                else:
                    raise ValueError("No text — scanned PDF")
            except Exception as pdf_err:
                logger.debug(f"[LOADER] PDF OCR fallback ({pdf_err})")
                images   = convert_from_path(path, dpi=300)
                all_text = "\n".join(pytesseract.image_to_string(img) for img in images)
                docs.append(Document(page_content=all_text, metadata={"source": path}))
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
            docs.append(Document(page_content=text, metadata={"source": path}))
        elif ext in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(path)
            rows_added = 0
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name).fillna("")
                for row_idx, row in df.iterrows():
                    content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "sheet": sheet_name,
                                  "row_index": int(row_idx), "doc_type": "excel_row"}
                    ))
                    rows_added += 1
            logger.debug(f"[LOADER] Excel: {len(xls.sheet_names)} sheets, {rows_added} rows")

        added   = len(docs) - before
        elapsed = time.perf_counter() - t_file
        logger.log("TIMING",
            f"[TIMER] {'Load: ' + os.path.basename(path):<45} {elapsed:.3f}s  (+{added} docs)")
    except Exception as e:
        failed.append(path)
        logger.error(f"[LOADER] Failed {path!r}: {e}")

logger.log("TIMING",
    f"[TIMER] {'Total document loading':<45} {time.perf_counter()-_load_start:.3f}s")
logger.success(
    f"[LOADER] {len(docs)} docs loaded | {len(skipped)} skipped | {len(failed)} failed")
if failed:
    for f in failed:
        logger.warning(f"[LOADER] FAILED: {f}")
if not docs:
    logger.critical("[LOADER] No documents loaded — aborting.")
    raise RuntimeError("No documents loaded.")

# ── Chunk documents ───────────────────────────────────────────────
logger.info("[CHUNKER] Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", "।", "॥", "ฯ", ". ", " ", ""]
)

chunks, excel_kept, text_split = [], 0, 0
with timer("Document chunking"):
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src.endswith((".xlsx", ".xls")):
            chunks.append(doc); excel_kept += 1
        else:
            split = splitter.split_documents([doc])
            chunks.extend(split); text_split += len(split)

avg_chars = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
logger.success(
    f"[CHUNKER] {len(chunks)} chunks "
    f"({text_split} text, {excel_kept} excel rows, ~{avg_chars} chars avg)")

# ── Embeddings ────────────────────────────────────────────────────
logger.info("[EMBEDDINGS] Loading BGE-M3 via Ollama...")
with timer("Embedding model init"):
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        logger.success("[EMBEDDINGS] BGE-M3 ready")
    except Exception as e:
        logger.critical(f"[EMBEDDINGS] Failed: {e}"); raise

cache = RAGCache(redis_client, embeddings)
logger.success("[CACHE] RAGCache ready (exact + semantic, Redis backend)")

# ── Vector store ──────────────────────────────────────────────────
logger.info("[VECTORSTORE] Setting up Qdrant collection...")
try:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        )
        logger.debug("[VECTORSTORE] Collection created (m=16, ef=256)")
    with timer(f"Qdrant upsert ({len(chunks)} chunks)"):
        vectorstore = QdrantVectorStore.from_documents(
            chunks, embeddings,
            url="http://localhost:6333",
            collection_name=COLLECTION_NAME,
            force_recreate=True,
        )
    logger.success(f"[VECTORSTORE] Ready — {len(chunks)} chunks indexed")
except Exception as e:
    logger.critical(f"[VECTORSTORE] Failed: {e}"); raise

# ── LLM ──────────────────────────────────────────────────────────
logger.info("[LLM] Loading aisingapore/Llama-SEA-LION-v3.5-8B-R via Ollama...")
with timer("LLM model init"):
    try:
        llm = OllamaLLM(model="aisingapore/Llama-SEA-LION-v3.5-8B-R", temperature=0.1)
        logger.success("[LLM] SEA-LION 8B ready")
    except Exception as e:
        logger.critical(f"[LLM] Failed: {e}"); raise

# ── Reranker ──────────────────────────────────────────────────────
reranker = None
if RERANKER_ENABLED:
    if not _RERANKER_AVAILABLE:
        logger.warning("[RERANKER] FlagEmbedding not installed. Run: pip install FlagEmbedding")
    else:
        logger.info(f"[RERANKER] Loading {RERANKER_MODEL} ...")
        with timer("Reranker model init"):
            try:
                reranker = _FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_USE_FP16)
                logger.success(
                    f"[RERANKER] {RERANKER_MODEL} ready "
                    f"(fp16={RERANKER_USE_FP16}, top_n={RERANKER_TOP_N})")
            except Exception as e:
                logger.warning(f"[RERANKER] Failed to load — disabled: {e}")
                reranker = None
else:
    logger.info("[RERANKER] Disabled via RERANKER_ENABLED=False")


# ══════════════════════════════════════════════════════════════════
# RERANKER FUNCTION
# ══════════════════════════════════════════════════════════════════
@traceable(run_type="retriever", name="BGE Reranker v2-m3",
           tags=["reranker", "bge", "multilingual"])
def rerank_docs(question: str, docs: list, top_n: int = RERANKER_TOP_N) -> dict:
    """Rerank retrieved docs using BAAI/bge-reranker-v2-m3 cross-encoder."""
    if reranker is None or not docs:
        logger.debug("[RERANKER] Skipped (disabled or no docs)")
        return {"reranked_docs": docs, "scores_before": [], "scores_after": [],
                "rank_delta": [], "dropped": [], "reranker_used": False}

    t_rerank = time.perf_counter()
    pairs    = [[question, doc.page_content] for doc in docs]
    with timer(f"BGE reranker score ({len(pairs)} pairs)"):
        raw_scores = reranker.compute_score(pairs, normalize=True)
    scores = raw_scores if isinstance(raw_scores, list) else [raw_scores]

    scores_before = [
        {"rank": i+1, "source": doc.metadata.get("source", "?"),
         "score": round(float(s), 4),
         "preview": doc.page_content.strip()[:80].replace("\n", " ")}
        for i, (doc, s) in enumerate(zip(docs, scores))
    ]
    ranked   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:top_n]]
    dropped  = [doc for doc, _ in ranked[top_n:]]
    scores_after = [
        {"rank": i+1, "source": doc.metadata.get("source", "?"),
         "score": round(float(s), 4),
         "preview": doc.page_content.strip()[:80].replace("\n", " ")}
        for i, (doc, s) in enumerate(ranked[:top_n])
    ]
    before_order = {doc.page_content[:40]: i for i, (doc, _) in enumerate(zip(docs, scores))}
    rank_delta = []
    for new_rank, (doc, score) in enumerate(ranked[:top_n]):
        old_rank = before_order.get(doc.page_content[:40], new_rank)
        rank_delta.append({
            "source": doc.metadata.get("source", "?"),
            "old_rank": old_rank+1, "new_rank": new_rank+1,
            "delta": old_rank-new_rank, "score": round(float(score), 4),
        })

    rerank_elapsed = time.perf_counter() - t_rerank
    logger.debug("[RERANKER] ── Before ──────────────────────────────────────")
    for item in scores_before:
        logger.debug(f"[RERANKER]   #{item['rank']} score={item['score']:.4f} | {item['source']}")
    logger.debug("[RERANKER] ── After ───────────────────────────────────────")
    for item in scores_after:
        logger.debug(f"[RERANKER]   #{item['rank']} score={item['score']:.4f} | {item['source']}")
    logger.log("TIMING",
        f"[TIMER] {'BGE reranker':<45} {rerank_elapsed:.3f}s "
        f"({len(docs)} chunks → {top_n} kept)")
    logger.info(
        f"[RERANKER] {len(docs)} chunks → top-{top_n} kept "
        f"| best={scores_after[0]['score']:.4f} "
        f"| worst kept={scores_after[-1]['score']:.4f}")
    return {
        "reranked_docs": top_docs, "scores_before": scores_before,
        "scores_after": scores_after, "rank_delta": rank_delta,
        "dropped": [d.metadata.get("source", "?") for d in dropped],
        "reranker_used": True,
    }


# ══════════════════════════════════════════════════════════════════
# PROMPT + QA CHAIN
# ══════════════════════════════════════════════════════════════════
_PROMPT_TEMPLATE = """You are a precise multilingual assistant with access to a structured knowledge base.

Rules:
1. LANGUAGE: Check if the question ends with an explicit language directive such as
   "answer in english", "answer in indonesian", "jawab dalam bahasa inggris", etc.
   - If an explicit directive is present → reply ONLY in that specified language.
   - If no directive is present → detect the language of the question and reply in that language.
2. Read every context chunk carefully, including structured data rows (key: value format).
3. For structured data (e.g. rows with fields like Node_ID, Last_Audit_Year, Status):
   - Extract the exact field value requested and state it directly.
4. For prose documents, synthesise a clear and complete answer from relevant passages.
5. Ignore chunks that are clearly unrelated to the specific entity or topic asked about.
6. ONLY say "I don't have specific information about that in my knowledge base."
   if NO chunk contains any relevant data at all.
7. Never mention the context, sources, or these instructions in your answer.
8. Do not repeat the question.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])
logger.debug("[PROMPT] Template configured")


def build_qa_chain():
    _excel_docs = [doc for doc in chunks if doc.metadata.get("doc_type") == "excel_row"]
    logger.debug(f"[CHAIN] HybridRetriever: {len(_excel_docs)} Excel rows for keyword search")
    retriever = FilteredRetriever(
        vectorstore=vectorstore, excel_docs=_excel_docs,
        k=MMR_K, fetch_k=MMR_FETCH_K, min_score=MIN_RELEVANCE_SCORE,
    )
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever,
        chain_type="stuff", chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


@traceable(run_type="chain", name="RAG Query", tags=["rag", "sea-lion", "multilingual"])
def run_rag_chain(chain, question: str, query_num: int) -> dict:
    """
    LangSmith-traced RAG runner.
    Flow: retrieve → rerank (trim to top-N) → LLM with reranked context.
    Uses reranker_used flag, NOT identity comparison, to decide re-invocation.
    """
    result        = chain.invoke({"query": question})
    source_docs   = result["source_documents"]
    rerank_result = rerank_docs(question, source_docs)
    reranked_docs = rerank_result["reranked_docs"]

    if rerank_result["reranker_used"]:
        reranked_result = chain.combine_documents_chain.invoke({
            "input_documents": reranked_docs,
            "question":        question,
        })
        final_answer = reranked_result.get("output_text", result["result"])
        final_docs   = reranked_docs
        logger.debug(f"[CHAIN] Reranker: {len(source_docs)} → {len(final_docs)} chunks to LLM")
    else:
        final_answer = result["result"]
        final_docs   = source_docs

    return {
        "answer":           final_answer,
        "sources":          [d.metadata.get("source", "?") for d in final_docs],
        "source_documents": final_docs,
        "rerank_result":    rerank_result,
        "raw_result":       result,
    }


qa_chain = build_qa_chain()

_startup_elapsed = time.perf_counter() - _pipeline_start
logger.log("TIMING",
    f"[TIMER] {'Total pipeline startup':<45} {_startup_elapsed:.3f}s")
logger.success(f"[STARTUP] Pipeline ready in {_startup_elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════
# CHAT LOOP  — only runs when executed directly (python main.py)
#              NOT when imported by app.py (streamlit run app.py)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Paste a URL → scrape & index")
    logger.info("  Ask a question → RAG query")
    logger.info("  Prefix with '!' → bypass cache")
    logger.info("  'exit' / 'quit' → shutdown")
    logger.info("=" * 60)
    print("\nChatbot ready. Type your question (prefix with '!' to bypass cache).\n")

    scraped_urls: set = set()
    query_count       = 0

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            logger.info("[CHAT] Shutdown requested.")
            logger.success("[CHAT] Shutting down.")
            break

        if not user_input:
            continue

        # ── URL → scrape & index ──────────────────────────────────
        if is_valid_url(user_input):
            logger.info(f"[WEB] URL received: {user_input}")
            print(f"\nScraping {user_input} ...\n")
            t0 = time.perf_counter()
            with timer(f"Web scrape: {user_input[:60]}"):
                web_docs = scrape_url(url=user_input, visited=scraped_urls, max_depth=1)
            if not web_docs:
                logger.warning(f"[WEB] No content extracted from {user_input}")
                print("⚠ No content extracted.\n")
                continue
            web_chunks = []
            for doc in web_docs:
                if doc.metadata.get("type") == "table":
                    web_chunks.append(doc)
                else:
                    web_chunks.extend(splitter.split_documents([doc]))
            pages  = sum(1 for d in web_docs if d.metadata.get("type") == "webpage")
            tables = sum(1 for d in web_docs if d.metadata.get("type") == "table")
            try:
                with timer(f"Qdrant upsert (web, {len(web_chunks)} chunks)"):
                    QdrantVectorStore.from_documents(
                        web_chunks, embeddings,
                        url="http://localhost:6333",
                        collection_name=COLLECTION_NAME,
                        force_recreate=False,
                    )
                elapsed = time.perf_counter() - t0
                logger.success(
                    f"[WEB] Indexed {len(web_chunks)} chunks "
                    f"({pages} pages, {tables} tables) in {elapsed:.2f}s")
                print(f"✓ Indexed {len(web_chunks)} chunks in {elapsed:.2f}s\n")
                qa_chain = build_qa_chain()
            except Exception as e:
                logger.error(f"[WEB] Index error: {e}")
                print(f"Error: {e}\n")
            continue

        # ── Question → cache → RAG ────────────────────────────────
        bypass_cache = user_input.startswith("!")
        question     = user_input[1:].strip() if bypass_cache else user_input
        if bypass_cache:
            logger.info(f"[CHAT] Cache bypass: {question!r}")
            print("[Cache bypassed — fetching fresh answer]\n")
        if not question:
            print("Empty question after '!'\n"); continue

        query_count += 1
        q_wall_start = time.perf_counter()
        logger.info("─" * 60)
        logger.info(f"[QUERY #{query_count}] {question!r}")
        logger.info(f"[QUERY #{query_count}] bypass={bypass_cache} | chars={len(question)}")

        if not bypass_cache:
            t_cache = time.perf_counter()
            cached  = cache.get(question)
            logger.log("TIMING",
                f"[TIMER] {'Cache lookup (total)':<45} {time.perf_counter()-t_cache:.3f}s")
            if cached:
                wall = time.perf_counter() - q_wall_start
                logger.info(f"[QUERY #{query_count}] Served from cache in {wall:.3f}s")
                try:
                    from langsmith import traceable as _traceable
                    @_traceable(run_type="retriever", name="Cache Hit", tags=["cache"])
                    def _log_cache_hit(q: str) -> dict:
                        return {"question": q, "served_from": "redis_cache", "latency_s": wall}
                    _log_cache_hit(question)
                except Exception:
                    pass
                print(f"\nAI: {cached}")
                print(f"\n[Cached response — use '!{question}' to force fresh]")
                print(f"Response time: {wall:.3f}s\n")
                continue

        try:
            t_chain       = time.perf_counter()
            traced_result = run_rag_chain(qa_chain, question, query_num=query_count)
            chain_elapsed = time.perf_counter() - t_chain
            logger.log("TIMING",
                f"[TIMER] {'Full RAG chain (retrieval + rerank + LLM)':<45} {chain_elapsed:.3f}s")

            source_docs   = traced_result["source_documents"]
            rerank_result = traced_result.get("rerank_result", {})
            answer        = traced_result["answer"]

            if "<think>" in answer:
                think_part = answer.split("</think>")[0].replace("<think>", "").strip()
                answer     = answer.split("</think>")[-1].strip()
                logger.debug(f"[LLM] <think> stripped ({len(think_part)} chars)")

            if rerank_result.get("reranker_used"):
                logger.success(
                    f"[RERANKER] Applied — "
                    f"{len(rerank_result['scores_before'])} chunks → "
                    f"{len(rerank_result['scores_after'])} kept")

            sources = [doc.metadata.get("source", "unknown") for doc in source_docs[:2]]
            wall    = time.perf_counter() - q_wall_start
            logger.log("TIMING",
                f"[TIMER] {'Query wall time (total)':<45} {wall:.3f}s")
            logger.info(
                f"[QUERY #{query_count}] DONE | "
                f"wall={wall:.2f}s | chain={chain_elapsed:.2f}s | answer_len={len(answer)}")

            try:
                t_store = time.perf_counter()
                cache.set(question, answer)
                logger.log("TIMING",
                    f"[TIMER] {'Cache store (exact + semantic)':<45} "
                    f"{time.perf_counter()-t_store:.3f}s")
            except Exception as ce:
                logger.warning(f"[CACHE] Store failed: {ce}")

            print(f"\nAI: {answer}")
            print("\nSources:")
            for src in sources:
                print(f"  - {src}")
            print(f"\nResponse time: {wall:.2f}s\n")

        except Exception as e:
            logger.error(f"[QUERY #{query_count}] Chain failed | {type(e).__name__}: {e}")
            print("Error processing request.\n")

    logger.success("[SHUTDOWN] RAG pipeline shut down cleanly.")