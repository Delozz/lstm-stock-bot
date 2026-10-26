"""AC-14 to AC-16: hand-checked features, labels, warm-up drop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.features import FEATURES, build_features, compute_features

NO_NEWS = pd.DataFrame(columns=["session_date", "vader_compound"])


def linear_bars(n: int = 30) -> pd.DataFrame:
    i = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "session_date": pd.bdate_range("2026-01-05", periods=n).strftime("%Y-%m-%d"),
            "open": 100 + i,
            "close": 101 + i,
            "high": 102 + i,
            "low": 99 + i,
            "volume": 1000 + 10 * i,
        }
    )


def test_hand_computed_price_features():
    f = compute_features(linear_bars(), NO_NEWS).iloc[25]
    t = 25
    assert f["ret_1d"] == pytest.approx(np.log((101 + t) / (100 + t)), abs=1e-9)
    assert f["ret_5d"] == pytest.approx(np.log((101 + t) / (96 + t)), abs=1e-9)
    assert f["ret_20d"] == pytest.approx(np.log((101 + t) / (81 + t)), abs=1e-9)
    assert f["gap"] == pytest.approx(0.0, abs=1e-12)  # open_t equals close_{t-1}
    assert f["oc_ret"] == pytest.approx(np.log((101 + t) / (100 + t)), abs=1e-9)
    assert f["hl_range"] == pytest.approx(3 / (101 + t), abs=1e-9)
    vols = 1000 + 10 * np.arange(t - 19, t + 1)
    assert f["vol_z_20d"] == pytest.approx((vols[-1] - vols.mean()) / vols.std(ddof=1), abs=1e-9)
    rets = np.log((101 + np.arange(t - 19, t + 1)) / (100 + np.arange(t - 19, t + 1)))
    assert f["vol_20d"] == pytest.approx(rets.std(ddof=1), abs=1e-9)


def test_sentiment_features():
    bars = linear_bars()
    news = pd.DataFrame(
        {"session_date": [bars.session_date[25]] * 3, "vader_compound": [0.5, -0.2, 0.3]}
    )
    f = compute_features(bars, news)
    with_news, without = f.iloc[25], f.iloc[24]
    assert with_news["sent_mean"] == pytest.approx(0.2)
    assert (with_news["sent_max"], with_news["sent_min"]) == (0.5, -0.2)
    assert with_news["news_count"] == pytest.approx(np.log(4))
    assert with_news["news_available"] == 1
    assert [
        without[c] for c in ("sent_mean", "sent_max", "sent_min", "news_count", "news_available")
    ] == [0, 0, 0, 0, 0]


def test_label_is_next_session_open_to_close():
    bars = linear_bars(25)
    bars.loc[21, ["open", "close"]] = [100.0, 101.0]
    f = compute_features(bars, NO_NEWS)
    assert f.loc[20, "next_oc_ret"] == pytest.approx(0.01)
    assert f.loc[20, "label"] == 1
    assert pd.isna(f.iloc[-1]["next_oc_ret"]) and pd.isna(f.iloc[-1]["label"])


def test_build_features_drops_warmup(conn, make_bars, add_bars):
    for k, sym in enumerate(("AAA", "BBB")):
        add_bars(conn, sym, make_bars(40, seed=k))
    rows, dropped = build_features(conn)
    assert (rows, dropped) == (40, 40)
    first = conn.execute("SELECT MIN(session_date) FROM features WHERE symbol = 'AAA'").fetchone()[
        0
    ]
    assert first == make_bars(40).session_date[20]
    nulls = conn.execute(
        f"SELECT COUNT(*) FROM features WHERE {' OR '.join(f'{c} IS NULL' for c in FEATURES)}"
    ).fetchone()[0]
    assert nulls == 0
