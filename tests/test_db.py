from stockbot.db import upsert
from stockbot.universe import active_symbols, load_universe


def test_upsert_updates_on_conflict(conn):
    row = {"symbol": "AAA", "name": "Old", "sector": None, "is_benchmark": 0}
    upsert(conn, "tickers", [row], ("symbol",))
    upsert(conn, "tickers", [{**row, "name": "New"}], ("symbol",))
    assert [tuple(r) for r in conn.execute("SELECT name FROM tickers")] == [("New",)]


def test_upsert_ignore_keeps_first(conn):
    upsert(
        conn,
        "tickers",
        [{"symbol": "AAA", "name": "A", "sector": None, "is_benchmark": 0}],
        ("symbol",),
    )
    news = {
        "symbol": "AAA",
        "published_at_utc": "2026-01-02T15:00:00+00:00",
        "session_date": "2026-01-02",
        "headline": "Same",
        "source": "first",
        "url": None,
        "vader_compound": 0.0,
    }
    upsert(conn, "news", [news], ("symbol", "published_at_utc", "headline"), update=False)
    upsert(
        conn,
        "news",
        [{**news, "source": "second"}],
        ("symbol", "published_at_utc", "headline"),
        update=False,
    )
    assert [tuple(r) for r in conn.execute("SELECT source FROM news")] == [("first",)]


def test_load_universe_adds_benchmark(conn, tmp_path):
    csv = tmp_path / "u.csv"
    csv.write_text("symbol,name,sector\n" + "".join(f"S{i},Name {i},Tech\n" for i in range(5)))
    assert load_universe(conn, csv, "SPY") == 6
    assert conn.execute("SELECT is_benchmark FROM tickers WHERE symbol = 'SPY'").fetchone()[0] == 1
    assert "SPY" not in active_symbols(conn)
    assert "SPY" in active_symbols(conn, include_benchmark=True)
