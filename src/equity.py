"""資金管理シミュレーション（全ペア時系列マージ、R倍数ベース）。

SPEC.md §7 準拠。イベント（エントリー/エグジット）を時刻順に処理し、同時刻は
クローズ処理をエントリー処理より先に行う。リスク額はエントリー時点で確定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class EquityResult:
    """1つの資金管理方式・R列に対するシミュレーション結果。"""

    curve: pd.DataFrame            # 列: ts, equity（1イベント1行）
    final_equity: float
    max_dd_jpy: float
    max_dd_pct: float
    longest_underwater_days: float
    monthly: pd.DataFrame          # index=YYYY-MM(period), 列: start_eq, end_eq, ret
    max_concurrent: int
    max_open_risk_jpy: float
    ruined: bool
    ruin_ts: pd.Timestamp | None


def _empty_result(start: float) -> EquityResult:
    curve = pd.DataFrame({
        "ts": pd.Series(dtype="datetime64[ns]"),
        "equity": pd.Series(dtype="float64"),
    })
    monthly = pd.DataFrame({
        "start_eq": pd.Series(dtype="float64"),
        "end_eq": pd.Series(dtype="float64"),
        "ret": pd.Series(dtype="float64"),
    })
    return EquityResult(
        curve=curve, final_equity=float(start), max_dd_jpy=0.0, max_dd_pct=0.0,
        longest_underwater_days=0.0, monthly=monthly, max_concurrent=0,
        max_open_risk_jpy=0.0, ruined=False, ruin_ts=None,
    )


def simulate(
    trades: pd.DataFrame,
    mode: str,
    r_col: str,
    start: float = config.START_CAPITAL_JPY,
) -> EquityResult:
    """マージ済みトレード表へ資金管理方式を適用し資産曲線と各指標を返す。

    mode: "fixed"（¥2,000固定）/ "tiered"（ティア2%・ラチェット昇格）/
          "pct"（残高2%・連続複利）。r_col: 使用する R 列名。
    """
    n = len(trades)
    if n == 0:
        return _empty_result(start)

    ts_entry = pd.to_datetime(trades["ts_entry"]).tolist()
    ts_exit = pd.to_datetime(trades["ts_exit"]).tolist()
    r = trades[r_col].to_numpy(dtype=np.float64)
    pair = trades["pair"].astype(str).tolist()
    zero_hold = [ts_entry[i] == ts_exit[i] for i in range(n)]

    # イベント: (ts, rank, pair, trade_id)  rank: exit=0(クローズ先行), entry=1
    events: list[tuple] = []
    for i in range(n):
        events.append((ts_entry[i], 1, pair[i], i))
        events.append((ts_exit[i], 0, pair[i], i))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    equity = float(start)
    tier_base = float(start)
    taken = [False] * n
    risk_amt = [0.0] * n
    ruined = False
    ruin_ts: pd.Timestamp | None = None
    concurrent = 0
    open_risk = 0.0
    max_concurrent = 0
    max_open_risk = 0.0
    curve_ts: list = []
    curve_eq: list[float] = []

    for ts, rank, _pr, i in events:
        if rank == 1:  # エントリー
            if not ruined:
                if mode == "fixed":
                    ra = float(config.FIXED_RISK_JPY)
                elif mode == "tiered":
                    ra = config.RISK_PCT * tier_base
                elif mode == "pct":
                    ra = config.RISK_PCT * equity
                else:
                    raise ValueError(f"unknown mode: {mode}")
                taken[i] = True
                risk_amt[i] = ra
                if not zero_hold[i]:
                    concurrent += 1
                    open_risk += ra
                    if concurrent > max_concurrent:
                        max_concurrent = concurrent
                    if open_risk > max_open_risk:
                        max_open_risk = open_risk
            curve_ts.append(ts)
            curve_eq.append(equity)
        else:  # エグジット（クローズ）
            if taken[i]:
                equity += risk_amt[i] * float(r[i])
                if not zero_hold[i]:
                    concurrent -= 1
                    open_risk -= risk_amt[i]
                if mode == "tiered":
                    while equity >= 2.0 * tier_base:
                        tier_base *= 2.0
                if not ruined and equity <= 0.0:
                    ruined = True
                    ruin_ts = ts
            curve_ts.append(ts)
            curve_eq.append(equity)

    curve = pd.DataFrame({"ts": curve_ts, "equity": curve_eq})
    eq = curve["equity"].to_numpy(dtype=np.float64)
    # tz-naive の datetime64[ns] にそろえる（tz-aware だと to_numpy が object 配列を
    # 返し、差分計算や to_period で tz ドロップ警告を招くため）。
    _ts = curve["ts"]
    _ts_naive = _ts.dt.tz_convert(None) if _ts.dt.tz is not None else _ts
    ts_ns = _ts_naive.to_numpy(dtype="datetime64[ns]")

    peak = np.maximum.accumulate(eq)
    dd_jpy = peak - eq
    max_dd_jpy = float(dd_jpy.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peak > 0.0, dd_jpy / peak, 0.0)
    max_dd_pct = float(np.nanmax(dd_pct)) if dd_pct.size else 0.0

    # 最長ドローダウン期間: 直近の高値更新からの経過日数の最大
    day = np.timedelta64(1, "D")
    longest_uw = 0.0
    last_peak_ts = ts_ns[0]
    last_peak_val = eq[0]
    for k in range(eq.shape[0]):
        if eq[k] >= last_peak_val:
            last_peak_val = eq[k]
            last_peak_ts = ts_ns[k]
        else:
            uw = (ts_ns[k] - last_peak_ts) / day
            if uw > longest_uw:
                longest_uw = float(uw)

    # 月次リターン（カレンダー月境界、前月末→当月末、初月は初期資金基準）
    period = _ts_naive.dt.to_period("M")
    monthly_end = curve.groupby(period, sort=True)["equity"].last()
    starts = monthly_end.shift(1)
    starts.iloc[0] = float(start)
    monthly = pd.DataFrame({"start_eq": starts, "end_eq": monthly_end})
    monthly["ret"] = monthly["end_eq"] / monthly["start_eq"] - 1.0
    monthly.index.name = "month"

    return EquityResult(
        curve=curve,
        final_equity=float(eq[-1]),
        max_dd_jpy=max_dd_jpy,
        max_dd_pct=max_dd_pct,
        longest_underwater_days=longest_uw,
        monthly=monthly,
        max_concurrent=int(max_concurrent),
        max_open_risk_jpy=float(max_open_risk),
        ruined=ruined,
        ruin_ts=ruin_ts,
    )
