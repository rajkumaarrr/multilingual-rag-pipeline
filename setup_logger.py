# setup_logger.py
#
# Log format matches the original main.py style:
#   Terminal  → INFO and above (clean, structured, human-readable)
#   Log file  → everything including DEBUG and TIMING
#
# HTTP libraries are silenced on terminal but written to rag_http.log

import sys
import logging
from loguru import logger

_HTTP_LIBS = {"httpx", "httpcore", "qdrant_client", "urllib3", "chromadb", "asyncio"}
_logger_initialized = False


def _is_http_lib(record: dict) -> bool:
    name = record["extra"].get("logger_name", "")
    return any(name.startswith(lib) for lib in _HTTP_LIBS)


def _terminal_filter(record: dict) -> bool:
    """Show INFO and above on terminal, never HTTP lib noise."""
    if _is_http_lib(record):
        return False
    return record["level"].no >= 20   # INFO=20, SUCCESS=25, WARNING=30, ERROR=40, CRITICAL=50


# ── Formats ───────────────────────────────────────────────────────
# Terminal: matches old main.py exactly — timestamp | level | message
_TERMINAL_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"

# File: full detail with milliseconds
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"

_HTTP_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[logger_name]} | {message}"


def setup_logger():
    global _logger_initialized
    if _logger_initialized:
        return logger
    _logger_initialized = True

    logger.remove()

    # ── Terminal: INFO and above, clean format ─────────────────────
    logger.add(
        sys.stdout,
        level="DEBUG",
        format=_TERMINAL_FORMAT,
        colorize=True,
        filter=_terminal_filter,
    )

    # ── Main log file: everything except HTTP noise ────────────────
    logger.add(
        "rag_detailed.log",
        level="TRACE",
        rotation="20 MB",
        retention="14 days",
        encoding="utf-8",
        format=_FILE_FORMAT,
        filter=lambda record: not _is_http_lib(record),
        enqueue=True,
    )

    # ── HTTP-only log file ─────────────────────────────────────────
    logger.add(
        "rag_http.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format=_HTTP_FILE_FORMAT,
        filter=_is_http_lib,
        enqueue=True,
    )

    # ── Custom levels ──────────────────────────────────────────────
    for name, no, color in [
        ("TIMING", 15, "<magenta>"),
        ("DETAIL", 12, "<yellow>"),
    ]:
        try:
            logger.level(name, no=no, color=color)
        except ValueError:
            pass

    # ── Bridge stdlib logging → loguru ────────────────────────────
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).bind(
                logger_name=record.name
            ).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)
    for lib in _HTTP_LIBS:
        logging.getLogger(lib).setLevel(logging.DEBUG)

    return logger