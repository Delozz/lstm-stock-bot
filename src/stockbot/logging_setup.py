"""JSON-lines logging to logs/YYYYMMDD.log with secret redaction (NFR-8, FR-6)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

SECRET_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FINNHUB_API_KEY")


def redact(text: str) -> str:
    """Replace any secret value from the environment with ***.

    Third-party exceptions can echo request URLs, and Finnhub puts its token in the
    query string, so everything that leaves the process goes through here.
    """
    for name in SECRET_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "***")
    return text


class JsonFormatter(logging.Formatter):
    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": record.levelname.lower(),
            "command": self.command,
            "event": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


def setup_logging(command: str, log_dir: str | Path = "logs") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"{datetime.now(UTC):%Y%m%d}.log")
    handler.setFormatter(JsonFormatter(command))

    logger = logging.getLogger("stockbot")
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
