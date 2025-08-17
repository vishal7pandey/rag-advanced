from __future__ import annotations
import logging
import os
import re
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, MutableMapping, Mapping, Callable

import structlog
from structlog.contextvars import merge_contextvars, bind_contextvars, clear_contextvars

# Simple redactor: remove obvious secrets/PII patterns
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:openai_key]"),
    (re.compile(r"bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE), "[REDACTED:bearer]"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[REDACTED:email]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED:pan]"),
]

_LOG_CONFIGURED = False


def _redact_and_sanitize(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    def _redact_str(s: str) -> str:
        out = s
        for pat, repl in _REDACT_PATTERNS:
            out = pat.sub(repl, out)
        return out

    for k, v in list(event_dict.items()):
        try:
            if isinstance(v, str):
                event_dict[k] = _redact_str(v)
            elif isinstance(v, dict):
                event_dict[k] = {
                    ik: (_redact_str(iv) if isinstance(iv, str) else iv) for ik, iv in v.items()
                }
        except Exception:
            # Best-effort redaction; never break logging
            pass
    return event_dict


def configure_logging(
    env: str | None = None, logs_dir: Path | None = None
) -> structlog.stdlib.BoundLogger:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return structlog.get_logger("rag")

    env = env or os.getenv("APP_ENV", "dev").lower()
    logs_dir = logs_dir or (Path(__file__).resolve().parents[2] / "logs")
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Stdlib logging setup
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    handlers.append(console)

    # Daily rotation JSONL file
    try:
        file_handler = TimedRotatingFileHandler(
            str(logs_dir / "app.jsonl"), when="midnight", backupCount=14, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)
    except Exception:
        # If file handler fails, continue with console only
        pass

    root.handlers = handlers

    def _redact_processor(
        logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> Mapping[str, Any]:
        # Delegate to sanitizer while matching structlog's processor signature
        try:
            # Work on a plain dict to avoid surprising side-effects
            return _redact_and_sanitize(dict(event_dict))
        except Exception:
            return event_dict

    Processor = Callable[
        [Any, str, MutableMapping[str, Any]],
        Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...],
    ]

    processors: list[Processor] = [
        structlog.processors.add_log_level,  # type: ignore[arg-type]
        structlog.processors.TimeStamper(fmt="iso"),  # type: ignore[arg-type]
        merge_contextvars,  # type: ignore[arg-type]
        _redact_processor,
        structlog.processors.JSONRenderer(),  # type: ignore[arg-type]
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOG_CONFIGURED = True
    return structlog.get_logger("rag")


# Convenience helpers for context binding


def log_context_clear() -> None:
    clear_contextvars()


def log_context_bind(**kwargs: Any) -> None:
    bind_contextvars(**kwargs)
