"""指標計算（SMA400 とボリンジャーバンド）。

SPEC.md §2 準拠。すべて float64・バー本数ベースのローリング。
sigma は母標準偏差（ddof=0）を既定とし、ddof は必ず引数で明示的に渡す
（pandas の std は既定 ddof=1 のため配線ミスに注意）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class Indicators:
    """バー列と同じ長さの指標配列（ウォームアップ区間は NaN）。"""

    sma400: np.ndarray
    mid: np.ndarray
    sigma: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


def compute_indicators(
    close: np.ndarray,
    bb_period: int = config.BB_PERIOD,
    bb_dev: float = config.BB_DEV,
    sma_period: int = config.SMA_PERIOD,
    ddof: int = config.BB_DDOF,
) -> Indicators:
    """終値配列から SMA400・BB中央線・母標準偏差・±バンドを算出する。

    - sma400: 直近 sma_period 本の終値単純平均
    - mid: 直近 bb_period 本の終値単純平均（BB中央線）
    - sigma: 直近 bb_period 本の終値標準偏差（ddof を明示的に適用）
    - upper/lower: mid ± bb_dev * sigma
    有効値が揃わないウォームアップ区間は NaN。
    """
    s = pd.Series(np.asarray(close, dtype=np.float64))
    sma400 = s.rolling(sma_period).mean().to_numpy()
    mid = s.rolling(bb_period).mean().to_numpy()
    # ddof を必ず渡す（pandas 既定は ddof=1、本手法は ddof=0）
    sigma = s.rolling(bb_period).std(ddof=ddof).to_numpy()
    upper = mid + bb_dev * sigma
    lower = mid - bb_dev * sigma
    return Indicators(sma400=sma400, mid=mid, sigma=sigma, upper=upper, lower=lower)
