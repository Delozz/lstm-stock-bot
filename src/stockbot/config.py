"""Config loading and the paper-only guard (FR-1, FR-2)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PAPER_URL = "https://paper-api.alpaca.markets"


class ConfigError(Exception):
    exit_code = 2


class SafetyError(Exception):
    exit_code = 3


class DataError(Exception):
    exit_code = 4


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(path)
    load_dotenv(override=False)
    return cfg


def require_secret(name: str) -> str:
    """Return a secret from the environment. Errors name the variable, never the value."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"missing required environment variable {name}")
    return value


def assert_paper() -> str:
    """Refuse to continue unless the environment points at the Alpaca paper endpoint."""
    paper = os.environ.get("ALPACA_PAPER", "").strip().lower()
    url = os.environ.get("ALPACA_BASE_URL", "").strip().rstrip("/")
    if paper != "true" or url != PAPER_URL:
        raise SafetyError(
            "refusing to call Alpaca: ALPACA_PAPER must be 'true' and "
            f"ALPACA_BASE_URL must be {PAPER_URL}"
        )
    return url
