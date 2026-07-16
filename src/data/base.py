"""バーデータの品質検証と QualityReport。

`validate_bars` は OHLC 不変条件・価格レンジ・重複・ギャップを数え上げる純粋な
レポータで、データ内容では例外を投げない（契約違反での raise は
`store.load_pair` の責務）。df は変更しない。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .. import config

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass
class QualityReport:
    """1ペア分のデータ品質サマリ。"""

    pair: str
    source: str
    rows: int
    first_ts: Optional[pd.Timestamp]
    last_ts: Optional[pd.Timestamp]
    n_dupes_dropped: int
    n_ohlc_violations: int
    n_out_of_range: int
    n_gaps_over_1min: int
    max_gap_minutes: float
    weekday_daily_bars_median: float

    def as_dict(self) -> dict:
        """JSON/表出力向けの素の dict（タイムスタンプは ISO8601 文字列）。"""

        def _ts(x) -> Optional[str]:
            return None if x is None else pd.Timestamp(x).isoformat()

        return {
            "pair": self.pair,
            "source": self.source,
            "rows": int(self.rows),
            "first_ts": _ts(self.first_ts),
            "last_ts": _ts(self.last_ts),
            "n_dupes_dropped": int(self.n_dupes_dropped),
            "n_ohlc_violations": int(self.n_ohlc_violations),
            "n_out_of_range": int(self.n_out_of_range),
            "n_gaps_over_1min": int(self.n_gaps_over_1min),
            "max_gap_minutes": float(self.max_gap_minutes),
            "weekday_daily_bars_median": float(self.weekday_daily_bars_median),
        }


def validate_bars(df: pd.DataFrame, pair: str, source: str = "dukascopy") -> QualityReport:
    """OHLC 不変条件・妥当レンジ・重複・ギャップを数え上げて QualityReport を返す。

    - OHLC 不変条件: ``low <= min(open, close)`` かつ ``max(open, close) <= high``
    - 妥当レンジ: ``config.PLAUSIBLE_RANGE[pair]`` の外にある価格を含む行を数える
    - 重複: 重複インデックス本数（build_parquet が keep="first" で落とす本数）
    - ギャップ: ユニーク・ソート済みインデックス上の 1 分超の間隔

    違反があっても例外は投げず件数として返す。df は変更しない。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index は DatetimeIndex である必要があります")

    idx = df.index
    n = len(df)

    # 重複本数（uniqueness 検査）
    n_dupes = int(idx.duplicated(keep="first").sum())

    if n:
        o = df["open"].to_numpy(dtype="float64")
        h = df["high"].to_numpy(dtype="float64")
        low = df["low"].to_numpy(dtype="float64")
        c = df["close"].to_numpy(dtype="float64")
        oc_min = np.minimum(o, c)
        oc_max = np.maximum(o, c)
        ok = (low <= oc_min) & (oc_max <= h)
        n_ohlc_violations = int((~ok).sum())

        lo, hi = config.PLAUSIBLE_RANGE[pair]
        prices = np.column_stack([o, h, low, c])
        oor_rows = ((prices < lo) | (prices > hi)).any(axis=1)
        n_out_of_range = int(oor_rows.sum())
    else:
        n_ohlc_violations = 0
        n_out_of_range = 0

    # ギャップ統計はユニーク・ソート済みインデックスで測る
    uniq = idx[~idx.duplicated(keep="first")].sort_values()
    if len(uniq) >= 2:
        deltas_min = uniq.to_series().diff().dropna().dt.total_seconds().to_numpy() / 60.0
        n_gaps_over_1min = int((deltas_min > 1.0 + 1e-9).sum())
        max_gap_minutes = float(deltas_min.max())
    else:
        n_gaps_over_1min = 0
        max_gap_minutes = 0.0

    # 平日 1 日あたりの本数の中央値（カバレッジ確認用、フル日 ~1440）
    if len(uniq):
        per_day = pd.Series(1, index=uniq).groupby(uniq.normalize()).size()
        weekday_mask = per_day.index.weekday < 5
        wk = per_day[weekday_mask]
        weekday_daily_bars_median = float(wk.median()) if len(wk) else 0.0
    else:
        weekday_daily_bars_median = 0.0

    return QualityReport(
        pair=pair,
        source=source,
        rows=n,
        first_ts=(uniq[0] if len(uniq) else None),
        last_ts=(uniq[-1] if len(uniq) else None),
        n_dupes_dropped=n_dupes,
        n_ohlc_violations=n_ohlc_violations,
        n_out_of_range=n_out_of_range,
        n_gaps_over_1min=n_gaps_over_1min,
        max_gap_minutes=max_gap_minutes,
        weekday_daily_bars_median=weekday_daily_bars_median,
    )
