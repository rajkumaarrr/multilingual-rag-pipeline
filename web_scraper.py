"""
web_scraper.py
──────────────
Standalone web scraping module for the RAG pipeline.
Handles URL validation, page text extraction, table extraction,
and recursive same-domain link following.

Usage:
    from web_scraper import scrape_url, is_valid_url
"""

import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from setup_logger import setup_logger

logger = setup_logger()

# ── Constants ──────────────────────────────────────────────────────────────────

# Browser-like header so most sites don't block the scraper
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# HTML tags that add no useful content — removed before text extraction
NOISE_TAGS = ["script", "style", "nav", "footer", "header",
               "aside", "form", "noscript", "iframe"]

# File extensions to skip when following links
SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif",
                   ".zip", ".exe", ".css", ".js", ".xml", ".rss"}

# Link protocols to skip
SKIP_PROTOCOLS = ("mailto:", "javascript:", "tel:", "#")

# Max child links to follow per page (safety cap)
MAX_LINKS_PER_PAGE = 2

# Request timeout in seconds
REQUEST_TIMEOUT = 15


# ── Public API ─────────────────────────────────────────────────────────────────

def is_valid_url(text: str) -> bool:
    """
    Return True if the given string is a valid HTTP/HTTPS URL.
    Used by the chat loop to decide whether input is a URL or a question.

    Args:
        text : The string to check

    Returns:
        True if valid URL, False otherwise
    """
    try:
        parsed = urlparse(text.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def scrape_url(url: str, visited: set, max_depth: int = 1, current_depth: int = 0) -> list:
    """
    Scrape a URL and optionally follow links found on the same domain.

    Extracts:
      - Main page text (noise tags removed)
      - All HTML tables (formatted as header:value rows)
      - Child pages linked from the page (same domain only, up to max_depth)

    Args:
        url           : The URL to scrape
        visited       : Set of already-visited URLs — pass the same set across
                        calls to avoid scraping the same page twice in a session
        max_depth     : How many link-levels deep to follow
                        0 = only scrape the given URL
                        1 = also follow links on that page (default)
                        2 = go two levels deep (can be slow)
        current_depth : Internal recursion counter — do not set this manually

    Returns:
        List of LangChain Documents ready for chunking and indexing.
        Each Document has metadata fields:
          - source     : the URL it came from
          - type       : "webpage" or "table"
          - table_index: (tables only) which table on the page (1-based)
    """
    if url in visited:
        logger.debug(f"[WEB] Skipping already-visited URL: {url}")
        return []

    visited.add(url)
    scraped_docs = []

    try:
        logger.info(f"[WEB] Fetching (depth={current_depth}): {url}")
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # handles non-UTF-8 sites

        soup = BeautifulSoup(response.text, "html.parser")

        # ── Strip noise tags ───────────────────────────────────────
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # ── Main page text ─────────────────────────────────────────
        raw_text = soup.get_text(separator="\n", strip=True)
        clean_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        clean_text = "\n".join(clean_lines)

        if clean_text:
            scraped_docs.append(Document(
                page_content=clean_text,
                metadata={"source": url, "type": "webpage"}
            ))
            logger.success(f"[WEB] Main content scraped: {len(clean_text)} chars — {url}")

        # ── Tables ─────────────────────────────────────────────────
        table_docs = _extract_tables(soup, url)
        scraped_docs.extend(table_docs)
        if table_docs:
            logger.info(f"[WEB] {len(table_docs)} table(s) extracted from {url}")

        # ── Follow links on the same domain ───────────────────────
        if current_depth < max_depth:
            child_docs = _follow_links(soup, url, visited, max_depth, current_depth)
            scraped_docs.extend(child_docs)

    except requests.exceptions.Timeout:
        logger.error(f"[WEB] Timeout fetching: {url}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"[WEB] HTTP {e.response.status_code} error: {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"[WEB] Connection failed: {url}")
    except Exception as e:
        logger.error(f"[WEB] Unexpected error scraping {url}: {type(e).__name__}: {e}")

    return scraped_docs


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_tables(soup: BeautifulSoup, source_url: str) -> list:
    """
    Extract all HTML tables from a parsed page.

    Each table becomes one Document. Rows are formatted as 'header: value'
    pairs separated by '---'. Tables are never split by the chunker because
    splitting a row across two chunks breaks the header↔value relationship.

    Args:
        soup       : Parsed BeautifulSoup object (noise already removed)
        source_url : URL the page was fetched from (used in metadata + content label)

    Returns:
        List of Documents, one per table found on the page
    """
    table_docs = []
    tables = soup.find_all("table")

    for t_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        if not rows:
            continue

        # Use <th> cells as headers if present, otherwise treat first row as header
        header_cells = rows[0].find_all("th")
        if header_cells:
            headers = [th.get_text(strip=True) for th in header_cells]
            data_rows = rows[1:]
        else:
            headers = [td.get_text(strip=True) for td in rows[0].find_all("td")]
            data_rows = rows[1:]

        table_text_rows = []
        for row in data_rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not any(cells):
                continue
            if headers and len(cells) == len(headers):
                row_text = "\n".join(f"{headers[i]}: {cells[i]}" for i in range(len(cells)))
            else:
                row_text = " | ".join(cells)
            table_text_rows.append(row_text)

        if table_text_rows:
            table_content = (
                f"[Table {t_idx + 1} from {source_url}]\n"
                + "\n---\n".join(table_text_rows)
            )
            table_docs.append(Document(
                page_content=table_content,
                metadata={
                    "source": source_url,
                    "type": "table",
                    "table_index": t_idx + 1
                }
            ))
            logger.info(f"[WEB] Table {t_idx + 1}: {len(table_text_rows)} rows — {source_url}")

    return table_docs


def _follow_links(soup: BeautifulSoup, base_url: str, visited: set,
                  max_depth: int, current_depth: int) -> list:
    """
    Find and recursively scrape links on the same domain.

    Filters out:
      - Links to other domains
      - Already-visited URLs
      - Non-HTML resources (images, PDFs, CSS, JS, etc.)
      - Special protocols (mailto, javascript, tel, anchors)

    Caps at MAX_LINKS_PER_PAGE links per page to prevent runaway crawling.

    Args:
        soup          : Parsed page to find links in
        base_url      : The URL this page was fetched from
        visited       : Shared visited set (passed through to child scrapes)
        max_depth     : Maximum crawl depth
        current_depth : Current depth level

    Returns:
        List of Documents from all followed child pages
    """
    base_domain = urlparse(base_url).netloc
    child_docs = []
    links_followed = 0

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        absolute_url = urljoin(base_url, href)

        # Same domain only
        if urlparse(absolute_url).netloc != base_domain:
            continue
        # Skip non-HTML file types
        if any(absolute_url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        # Skip special protocols
        if any(absolute_url.startswith(p) for p in SKIP_PROTOCOLS):
            continue
        if absolute_url in visited:
            continue

        docs = scrape_url(
            url=absolute_url,
            visited=visited,
            max_depth=max_depth,
            current_depth=current_depth + 1
        )
        child_docs.extend(docs)
        links_followed += 1

        if links_followed >= MAX_LINKS_PER_PAGE:
            logger.info(f"[WEB] {MAX_LINKS_PER_PAGE}-link cap reached for {base_url}")
            break

    return child_docs