"""The 13 features and the next-session open-to-close label (FR-16 to FR-19, PRD 8.2).

Every feature for session t uses bars and headlines dated t or earlier. Only the label
columns (next_oc_ret, label) look at t+1.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from stockbot.db import upsert

log = logging.getLogger(__name__)

PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_z_20d",
    "hl_range",
    "gap",
    "oc_ret",
]
SENTIMENT_FEATURES = ["sent_mean", "sent_max", "sent_min", "news_count", "news_available"]
FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES
WINDOW = 20


def compute_features(bars: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Features for one symbol. bars: session_date, open, high, low, close, volume.
    news: session_date, vader_compound. Warm-up rows come back with NaN."""
    b = bars.sort_values("session_date").reset_index(drop=True)
    o, h, lo, c = b["open"], b["high"], b["low"], b["close"]
    v = b["volume"].astype(float)
    log_c = np.log(c)

    out = pd.DataFrame({"session_date": b["session_date"]})
    out["ret_1d"] = log_c.diff()
    out["ret_5d"] = log_c - log_c.shift(5)
    out["ret_20d"] = log_c - log_c.shift(WINDOW)
    out["vol_20d"] = out["ret_1d"].rolling(WINDOW).std()
    v_mean, v_std = v.rolling(WINDOW).mean(), v.rolling(WINDOW).std()
    out["vol_z_20d"] = ((v - v_mean) / v_std).where(v_std > 0, 0.0).where(v_std.notna())
    out["hl_range"] = (h - lo) / c
    out["gap"] = np.log(o / c.shift(1))
    out["oc_ret"] = np.log(c / o)

    daily = news.groupby("session_date")["vader_compound"].agg(["mean", "max", "min", "count"])
    out = out.merge(daily, left_on="session_date", right_index=True, how="left")
    count = out["count"].fillna(0)
    out["sent_mean"] = out["mean"].fillna(0.0)
    out["sent_max"] = out["max"].fillna(0.0)
    out["sent_min"] = out["min"].fillna(0.0)
    out["news_count"] = np.log1p(count)
    out["news_available"] = (count > 0).astype(int)

    out["next_oc_ret"] = c.shift(-1) / o.shift(-1) - 1
    out["label"] = (out["next_oc_ret"] > 0).astype("Int64").where(out["next_oc_ret"].notna())
    return out[["session_date", *FEATURES, "next_oc_ret", "label"]]


def build_features(conn: sqlite3.Connection) -> tuple[int, int]:
    """Recompute features for every symbol with bars. Returns (rows written, rows dropped)."""
    bars = pd.read_sql_query(
        "SELECT symbol, session_date, open, high, low, close, volume FROM bars_daily", conn
    )
    news = pd.read_sql_query("SELECT symbol, session_date, vader_compound FROM news", conn)
    news_by_symbol = dict(tuple(news.groupby("symbol")))
    empty_news = news.iloc[0:0]

    frames = []
    for symbol, sym_bars in bars.groupby("symbol"):
        f = compute_features(sym_bars, news_by_symbol.get(symbol, empty_news))
        f.insert(0, "symbol", symbol)
        frames.append(f)
    if not frames:
        return 0, 0
    df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)

    complete = df[FEATURES].notna().all(axis=1)
    dropped = int((~complete).sum())
    df = df[complete]
    warmup = WINDOW * len(frames)
    if dropped > warmup and dropped - warmup > 0.01 * len(df):
        log.warning("dropped %d rows beyond warm-up (NaN or inf features)", dropped - warmup)

    records = df.astype(object).where(df.notna(), None).to_dict("records")
    upsert(conn, "features", records, ("symbol", "session_date"))
    log.info("feature rows: %d, dropped: %d", len(df), dropped)
    return len(df), dropped
