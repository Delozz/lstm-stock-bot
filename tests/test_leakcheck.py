"""AC-17: both leak checks pass on clean data and fail on injected leaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.features import build_features, compute_features
from stockbot.leakcheck import (
    LeakError,
    assert_no_future_dependence,
    assert_no_target_proxy,
    run_leakchecks,
)

NO_NEWS = pd.DataFrame(columns=["session_date", "vader_compound"])


def leaky_compute(bars, news):
    out = compute_features(bars, news)
    close = bars.sort_values("session_date")["close"].reset_index(drop=True)
    out["ret_1d"] = np.log(close.shift(-1) / close)  # uses C_{t+1}
    return out


def test_clean_features_pass(make_bars):
    bars = make_bars(60)
    assert_no_future_dependence(bars, NO_NEWS, bars.session_date[40])


def test_future_dependence_is_caught(make_bars):
    bars = make_bars(60)
    with pytest.raises(LeakError, match="ret_1d"):
        assert_no_future_dependence(bars, NO_NEWS, bars.session_date[40], compute=leaky_compute)


def test_target_proxy_is_caught(make_bars):
    f = compute_features(make_bars(200), NO_NEWS).dropna()
    assert_no_target_proxy(f)
    f["ret_1d"] = f["next_oc_ret"]
    with pytest.raises(LeakError, match="ret_1d"):
        assert_no_target_proxy(f)


def test_run_leakchecks_on_database(conn, make_bars, add_bars):
    for k in range(3):
        add_bars(conn, f"S{k}", make_bars(80, seed=k))
    build_features(conn)
    run_leakchecks(conn, sample_size=3)
