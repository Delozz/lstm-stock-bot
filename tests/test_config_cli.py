"""AC-1, AC-2, AC-3, AC-5, AC-6: config errors, paper guard, run records, secret hygiene."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stockbot import broker, cli
from stockbot.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # logs/ lands in tmp
    text = (ROOT / "config" / "default.toml").read_text()
    text = text.replace('db_path = "data/stockbot.db"', f'db_path = "{tmp_path / "t.db"}"')
    path = tmp_path / "cfg.toml"
    path.write_text(text)
    return path


def _set_keys(monkeypatch, base_url="https://paper-api.alpaca.markets"):
    monkeypatch.setenv("ALPACA_API_KEY", "SENTINEL_ALPACA_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SENTINEL_ALPACA_SECRET")
    monkeypatch.setenv("FINNHUB_API_KEY", "SENTINEL_FINNHUB_KEY")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_BASE_URL", base_url)


def _runs(cfg_path: Path):
    db_path = cfg_path.parent / "t.db"
    with sqlite3.connect(db_path) as c:
        return c.execute("SELECT command, status, exit_code, error_summary FROM runs").fetchall()


def test_missing_finnhub_key_exits_2(cfg_path, monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.delenv("FINNHUB_API_KEY")
    result = runner.invoke(app, ["ingest-news", "--config", str(cfg_path)])
    assert result.exit_code == 2
    assert "FINNHUB_API_KEY" in result.output
    assert "SENTINEL" not in result.output


@pytest.mark.parametrize("command", ["trade", "exit", "reconcile", "ingest-prices"])
def test_live_endpoint_is_refused_before_any_client(cfg_path, monkeypatch, command):
    _set_keys(monkeypatch, base_url="https://api.alpaca.markets")
    built = []
    monkeypatch.setattr(broker, "TradingClient", lambda *a, **k: built.append(a))
    monkeypatch.setattr(broker, "StockHistoricalDataClient", lambda *a, **k: built.append(a))

    result = runner.invoke(app, [command, "--config", str(cfg_path)])

    assert result.exit_code == 3
    assert built == []
    assert _runs(cfg_path)[-1][:3] == (command, "error", 3)


def test_paper_flag_must_be_true(cfg_path, monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    result = runner.invoke(app, ["trade", "--config", str(cfg_path)])
    assert result.exit_code == 3


def test_init_db_is_idempotent(cfg_path):
    first = runner.invoke(app, ["init-db", "--config", str(cfg_path)])
    with sqlite3.connect(cfg_path.parent / "t.db") as c:
        schema_1 = c.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    second = runner.invoke(app, ["init-db", "--config", str(cfg_path)])
    with sqlite3.connect(cfg_path.parent / "t.db") as c:
        schema_2 = c.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert first.output.strip() == "db ready"
    assert schema_1 == schema_2


def test_unexpected_error_exits_1_and_secrets_are_redacted(cfg_path, monkeypatch):
    _set_keys(monkeypatch)

    def boom():
        raise RuntimeError("GET https://finnhub.io/api/v1/news?token=SENTINEL_FINNHUB_KEY failed")

    monkeypatch.setattr(cli.broker, "make_trading_client", boom)
    result = runner.invoke(app, ["signal", "--config", str(cfg_path)])

    assert result.exit_code == 1
    assert "SENTINEL" not in result.output
    command, status, code, summary = _runs(cfg_path)[-1]
    assert (command, status, code) == ("signal", "error", 1)
    assert "SENTINEL" not in summary
    for log_file in (cfg_path.parent / "logs").glob("*.log"):
        assert "SENTINEL" not in log_file.read_text()
