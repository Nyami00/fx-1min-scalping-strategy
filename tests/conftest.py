"""pytest 共通設定とテスト用ビルダー。

- リポジトリルートを sys.path に載せ `src` パッケージを解決可能にする。
- make_bars / make_ind をファクトリ fixture として提供し、シーケンステストを
  指標計算そのものから切り離す。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from src import config
from src.indicators import Indicators


def _make_bars(o, h, l, c, volume=None, start_ts="2024-01-01 00:00", freq="1min"):
    """OHLCV 配列から契約準拠の DataFrame を作る（UTC index 'time', float64）。"""
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    vol = np.ones(n, dtype=np.float64) if volume is None else np.asarray(volume, dtype=np.float64)
    idx = pd.date_range(start=start_ts, periods=n, freq=freq, tz="UTC")
    idx.name = "time"
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": vol},
        index=idx,
    )


def _make_ind(sma400, mid, upper, lower, sigma=None):
    """手で設定した配列から Indicators を作る（sigma 省略時は (upper-mid)/BB_DEV）。"""
    sma400 = np.asarray(sma400, dtype=np.float64)
    mid = np.asarray(mid, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    if sigma is None:
        sigma = (upper - mid) / config.BB_DEV
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
    return Indicators(sma400=sma400, mid=mid, sigma=sigma, upper=upper, lower=lower)


@pytest.fixture
def make_bars():
    return _make_bars


@pytest.fixture
def make_ind():
    return _make_ind
