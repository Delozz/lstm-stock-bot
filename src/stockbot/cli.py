"""Typer CLI. One function per command; contracts in PRD Section 10."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from stockbot import broker, db
from stockbot.calendar import NY, get_sessions, latest_completed, previous_session, session_on
from stockbot.config import ConfigError, DataError, SafetyError, load_config, require_secret
from stockbot.logging_setup import redact, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LSTM stock signal bot. Alpaca paper trading only.",
)
log = logging.getLogger("stockbot.cli")

ConfigOpt = Annotated[Path, typer.Option("--config", help="Path to the TOML config.")]
StartOpt = Annotated[str | None, typer.Option("--start", help="First date, YYYY-MM-DD.")]
DryRunOpt = Annotated[bool, typer.Option("--dry-run", help="Print orders, submit nothing.")]
DEFAULT_CONFIG = Path("config/default.toml")

Body = Callable[[dict[str, Any], sqlite3.Connection], list[str]]


def _abort(code: int, message: str) -> None:
    typer.echo(f"ERROR {code}: {redact(message)}", err=True)
    raise typer.Exit(code)


def _run(command: str, config: Path, body: Body) -> None:
    """Record the run, map exceptions to exit codes (FR-5), print the summary (UX-1, UX-2)."""
    setup_logging(command)
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        _abort(exc.exit_code, str(exc))
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    run_id = db.start_run(conn, command)

    code, message, lines = 0, "", []
    try:
        lines = body(cfg, conn)
    except (ConfigError, SafetyError, DataError) as exc:
        code, message = exc.exit_code, str(exc)
        log.error(message)
    except Exception as exc:
        code, message = 1, f"{type(exc).__name__}: {exc}"
        log.exception("unexpected error")
    finally:
        db.finish_run(conn, run_id, code, redact(message) or None)
        conn.close()

    if code:
        _abort(code, message)
    for line in lines:
        typer.echo(line)


def _parse_date(value: str | None, default: date) -> date:
    if value is None:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"bad date {value!r}, expected YYYY-MM-DD") from exc


def _sessions_around(client, day: date, days: int = 14):
    return get_sessions(client, day - timedelta(days=days), day + timedelta(days=days))


def _check_failures(failed: list[str], total: int, what: str) -> None:
    # EC-6: a few failed symbols are fine, a broad outage is not.
    if total and len(failed) >= 0.05 * total:
        raise DataError(f"{len(failed)} of {total} symbols failed during {what}")


# --- commands -----------------------------------------------------------------


@app.command("init-db")
def init_db_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Create all tables. Safe to run again."""
    _run("init-db", config, lambda cfg, conn: ["db ready"])


@app.command("load-universe")
def load_universe_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Load data/sp500.csv and the SPY benchmark into tickers."""
    from stockbot.universe import load_universe

    def body(cfg, conn):
        n = load_universe(conn, cfg["universe_csv"], cfg["benchmark"])
        return [f"loaded {n} tickers"]

    _run("load-universe", config, body)


def _ingest_prices(cfg, conn, start: str | None = None) -> list[str]:
    from stockbot.prices import ingest_prices
    from stockbot.universe import active_symbols

    trading = broker.make_trading_client()
    data = broker.make_data_client()
    now = datetime.now(NY)
    last = latest_completed(_sessions_around(trading, now.date()), now)
    default_start = now.date() - timedelta(days=365 * cfg["data"]["price_years"])
    n, failed = ingest_prices(
        conn,
        data,
        end=last.date,
        default_start=_parse_date(start, default_start),
        batch_size=cfg["data"]["alpaca_batch"],
    )
    _check_failures(failed, len(active_symbols(conn, include_benchmark=True)), "ingest-prices")
    return [f"{n} bars upserted, {len(failed)} symbols failed"]


@app.command("ingest-prices")
def ingest_prices_cmd(config: ConfigOpt = DEFAULT_CONFIG, start: StartOpt = None) -> None:
    """Fetch daily bars from Alpaca (IEX feed, adjusted)."""
    _run("ingest-prices", config, lambda cfg, conn: _ingest_prices(cfg, conn, start))


def _ingest_news(cfg, conn, start: str | None = None) -> list[str]:
    import finnhub

    from stockbot.news import RateLimiter, ingest_news
    from stockbot.universe import active_symbols

    api_key = require_secret("FINNHUB_API_KEY")
    trading = broker.make_trading_client()
    today = datetime.now(NY).date()
    default_start = _parse_date(start, today - timedelta(days=cfg["data"]["news_days"]))
    sessions = get_sessions(trading, default_start - timedelta(days=10), today + timedelta(days=10))
    n, failed = ingest_news(
        conn,
        finnhub.Client(api_key=api_key),
        trading_days=[s.date for s in sessions],
        today=today,
        default_start=default_start,
        chunk_days=cfg["data"]["news_chunk_days"],
        limiter=RateLimiter(cfg["data"]["finnhub_rpm"]),
    )
    _check_failures(failed, len(active_symbols(conn)), "ingest-news")
    return [f"{n} headlines upserted, {len(failed)} symbols failed"]


@app.command("ingest-news")
def ingest_news_cmd(config: ConfigOpt = DEFAULT_CONFIG, start: StartOpt = None) -> None:
    """Fetch Finnhub company news and score it with VADER."""
    _run("ingest-news", config, lambda cfg, conn: _ingest_news(cfg, conn, start))


def _features(cfg, conn) -> list[str]:
    from stockbot.features import build_features

    n, dropped = build_features(conn)
    return [f"{n} rows, {dropped} dropped"]


@app.command("features")
def features_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Build the 13 features and next-session labels."""
    _run("features", config, _features)


@app.command("train")
def train_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Leak-check, then train the final model on all labeled data."""
    from stockbot.leakcheck import run_leakchecks
    from stockbot.train import train_final

    def body(cfg, conn):
        run_leakchecks(conn, cfg["walkforward"]["leakcheck_sample"], cfg["seed"])
        model_id, val_loss = train_final(conn, cfg)
        return [f"model_id={model_id} val_loss={val_loss:.5f}"]

    _run("train", config, body)


@app.command("backtest")
def backtest_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    no_ablation: Annotated[bool, typer.Option("--no-ablation")] = False,
) -> None:
    """Walk-forward backtest vs logistic, random, and SPY baselines."""
    from stockbot.backtest import run_backtest
    from stockbot.leakcheck import run_leakchecks
    from stockbot.reports import write_backtest_report
    from stockbot.train import git_sha, pick_device, set_seed

    def body(cfg, conn):
        run_leakchecks(conn, cfg["walkforward"]["leakcheck_sample"], cfg["seed"])
        set_seed(cfg["seed"])
        device = pick_device(cfg.get("device", "auto"))
        log.info("seed %d, device %s", cfg["seed"], device)
        metrics, curves = run_backtest(conn, cfg, device, ablation=not no_ablation)
        run_id = f"lstm-{datetime.now(NY):%Y%m%d-%H%M%S}"
        return [
            str(write_backtest_report(metrics, curves, run_id=run_id, cfg=cfg, git_sha=git_sha()))
        ]

    _run("backtest", config, body)


def _signal(cfg, conn) -> list[str]:
    from stockbot.signals import already_processed, generate_signals

    trading = broker.make_trading_client()
    now = datetime.now(NY)
    last = latest_completed(_sessions_around(trading, now.date()), now)
    session = last.date.isoformat()
    if already_processed(conn, session):
        log.info("already processed")
        return [f"already processed {session}"]
    selected = generate_signals(conn, cfg, session)
    return [f"selected: {','.join(selected) if selected else 'none'}"]


@app.command("signal")
def signal_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Score the latest completed session and pick the top names."""
    _run("signal", config, _signal)


def _today_session(client):
    now = datetime.now(NY)
    sessions = _sessions_around(client, now.date())
    return now, sessions, session_on(sessions, now.date())


@app.command("trade")
def trade_cmd(config: ConfigOpt = DEFAULT_CONFIG, dry_run: DryRunOpt = False) -> None:
    """Submit opening-auction buys for yesterday's selected signals."""
    from stockbot.execution import run_trade

    def body(cfg, conn):
        client = broker.make_trading_client()
        now, sessions, today = _today_session(client)
        if today is None:
            log.info("market closed")
            return ["market closed"]
        prev = previous_session(sessions, today.date)
        return run_trade(
            conn,
            client,
            cfg,
            today=today.date.isoformat(),
            signal_date=prev.date.isoformat(),
            dry_run=dry_run,
        )

    _run("trade", config, body)


@app.command("exit")
def exit_cmd(config: ConfigOpt = DEFAULT_CONFIG, dry_run: DryRunOpt = False) -> None:
    """Submit closing-auction sells for positions the bot opened today."""
    from stockbot.execution import run_exit

    def body(cfg, conn):
        client = broker.make_trading_client()
        now, _, today = _today_session(client)
        if today is None:
            log.info("market closed")
            return ["market closed"]
        return run_exit(
            conn,
            client,
            today=today.date.isoformat(),
            now=now,
            session_close=today.close,
            dry_run=dry_run,
        )

    _run("exit", config, body)


@app.command("reconcile")
def reconcile_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Pull fills from Alpaca and snapshot equity."""
    from stockbot.execution import run_reconcile

    def body(cfg, conn):
        client = broker.make_trading_client()
        _, _, today = _today_session(client)
        if today is None:
            log.info("market closed")
            return ["market closed"]
        return run_reconcile(conn, client, today=today.date.isoformat())

    _run("reconcile", config, body)


@app.command("nightly")
def nightly_cmd(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """ingest-prices, ingest-news, features, signal. Stops at the first failure."""
    from stockbot.signals import already_processed

    def body(cfg, conn):
        trading = broker.make_trading_client()
        now = datetime.now(NY)
        last = latest_completed(_sessions_around(trading, now.date()), now)
        if already_processed(conn, last.date.isoformat()):
            log.info("already processed")
            return [f"already processed {last.date}"]
        lines = []
        for name, step in (
            ("ingest-prices", _ingest_prices),
            ("ingest-news", _ingest_news),
            ("features", _features),
            ("signal", _signal),
        ):
            lines += [f"{name}: {line}" for line in step(cfg, conn)]
        return lines

    _run("nightly", config, body)


if __name__ == "__main__":
    app()
