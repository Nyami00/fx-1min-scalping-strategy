"""執行（スキップ判定 → エグジット走査 → トレード生成）。

SPEC.md §4 準拠。候補を時刻順に処理し、建玉フィルタと SL/TP 前方走査を行う。
保守/楽観バリアントは同一走査を共有し t_exit は共通。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.signals import Candidate

# エグジット走査のチャンク幅（ベクトル化のブロックサイズ）
_SCAN_CHUNK = 4096

# 走査前のトレード表カラム（凍結スキーマの r_net 群を除いた部分）。
TRADE_COLUMNS: list[str] = [
    "pair", "direction", "t_cross", "t_break", "t_entry", "t_exit",
    "ts_entry", "ts_exit", "entry", "sl", "tp", "risk_dist",
    "exit_price_cons", "reason_cons", "exit_price_opt", "reason_opt",
    "ambiguous", "r_gross_cons", "r_gross_opt", "hold_bars", "hold_minutes",
]

SKIP_COLUMNS: list[str] = ["pair", "ts", "direction", "reason"]


@dataclass
class ExitResult:
    """1トレードのエグジット確定結果（保守/楽観共通の t_exit）。"""

    t_exit: int
    exit_price_cons: float
    reason_cons: str
    exit_price_opt: float
    reason_opt: str
    ambiguous: bool


@dataclass
class Trade:
    """トレード1件（凍結スキーマ §6 の r_net 列を除く全フィールド）。"""

    pair: str
    direction: int
    t_cross: int
    t_break: int
    t_entry: int
    t_exit: int
    ts_entry: pd.Timestamp
    ts_exit: pd.Timestamp
    entry: float
    sl: float
    tp: float
    risk_dist: float
    exit_price_cons: float
    reason_cons: str
    exit_price_opt: float
    reason_opt: str
    ambiguous: bool
    r_gross_cons: float
    r_gross_opt: float
    hold_bars: int
    hold_minutes: float


@dataclass
class Skip:
    """スキップ1件（サイクルは消費済み）。"""

    pair: str
    t: int
    ts: pd.Timestamp
    direction: int
    reason: str


def resolve_exit(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    t_entry: int,
    direction: int,
    sl: float,
    tp: float,
) -> ExitResult:
    """t_entry+1 から前方走査し、最初に SL/TP に触れた足で確定する。

    注: eod 約定価格に最終 close が必要なため、SPEC の走査対象（open/high/low）に
    加えて close も引数に取る。判定優先順位は SPEC.md §4（買い、売りは完全ミラー）。
    """
    n = close.shape[0]
    start = t_entry + 1
    t_exit = -1
    while start < n:
        stop = min(start + _SCAN_CHUNK, n)
        lo = low[start:stop]
        hi = high[start:stop]
        if direction == 1:
            touch = (lo <= sl) | (hi >= tp)
        else:
            touch = (hi >= sl) | (lo <= tp)
        rel = np.flatnonzero(touch)
        if rel.size:
            t_exit = start + int(rel[0])
            break
        start = stop

    if t_exit < 0:
        # データ終端まで未タッチ → eod（最終 close で両バリアント共通）
        px = float(close[n - 1])
        return ExitResult(n - 1, px, "eod", px, "eod", False)

    o = float(open_[t_exit])
    h = float(high[t_exit])
    l = float(low[t_exit])
    if direction == 1:
        if o <= sl:  # 寄付が SL を貫通 → 不利フィル（曖昧扱いしない）
            return ExitResult(t_exit, o, "sl_gap", o, "sl_gap", False)
        if l <= sl and h >= tp:  # 同一足で両到達 → 曖昧足
            return ExitResult(t_exit, sl, "sl", tp, "tp", True)
        if l <= sl:
            return ExitResult(t_exit, sl, "sl", sl, "sl", False)
        return ExitResult(t_exit, tp, "tp", tp, "tp", False)  # 有利ギャップでも tp ちょうど
    else:
        if o >= sl:
            return ExitResult(t_exit, o, "sl_gap", o, "sl_gap", False)
        if h >= sl and l <= tp:
            return ExitResult(t_exit, sl, "sl", tp, "tp", True)
        if h >= sl:
            return ExitResult(t_exit, sl, "sl", sl, "sl", False)
        return ExitResult(t_exit, tp, "tp", tp, "tp", False)


def run_pair(
    df: pd.DataFrame,
    candidates: list[Candidate],
    pair: str,
) -> tuple[list[Trade], list[Skip]]:
    """1ペアの候補列を時刻順に処理してトレードとスキップを返す。"""
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    index = df.index
    one_min = pd.Timedelta(minutes=1)

    trades: list[Trade] = []
    skips: list[Skip] = []
    open_until = -1  # 直近建玉の t_exit（-1 = 建玉なし）

    for c in candidates:
        t = c.t_entry
        # 建玉フィルタ
        if t <= open_until:
            skips.append(Skip(pair, t, index[t], c.direction, "position_open"))
            continue
        # 異常足フィルタ（買い entry<=sl / 売り entry>=sl）
        if (c.direction == 1 and c.entry <= c.sl) or (c.direction == -1 and c.entry >= c.sl):
            skips.append(Skip(pair, t, index[t], c.direction, "invalid_sl"))
            continue

        res = resolve_exit(open_, high, low, close, t, c.direction, c.sl, c.tp)
        open_until = res.t_exit

        risk_dist = abs(c.entry - c.sl)
        r_gross_cons = c.direction * (res.exit_price_cons - c.entry) / risk_dist
        r_gross_opt = c.direction * (res.exit_price_opt - c.entry) / risk_dist
        ts_entry = index[t]
        ts_exit = index[res.t_exit]

        trades.append(Trade(
            pair=pair,
            direction=c.direction,
            t_cross=c.t_cross,
            t_break=c.t_break,
            t_entry=t,
            t_exit=res.t_exit,
            ts_entry=ts_entry,
            ts_exit=ts_exit,
            entry=c.entry,
            sl=c.sl,
            tp=c.tp,
            risk_dist=risk_dist,
            exit_price_cons=res.exit_price_cons,
            reason_cons=res.reason_cons,
            exit_price_opt=res.exit_price_opt,
            reason_opt=res.reason_opt,
            ambiguous=res.ambiguous,
            r_gross_cons=r_gross_cons,
            r_gross_opt=r_gross_opt,
            hold_bars=res.t_exit - t,
            hold_minutes=(ts_exit - ts_entry) / one_min,
        ))
    return trades, skips


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    """トレード列を走査前スキーマ順の DataFrame にする（空でも列を保持）。"""
    if not trades:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS})
    return pd.DataFrame([asdict(t) for t in trades])[TRADE_COLUMNS]


def skips_to_df(skips: list[Skip]) -> pd.DataFrame:
    """スキップ列を SPEC §6 の4列（pair, ts, direction, reason）にする。"""
    if not skips:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in SKIP_COLUMNS})
    return pd.DataFrame([asdict(s) for s in skips])[SKIP_COLUMNS]
