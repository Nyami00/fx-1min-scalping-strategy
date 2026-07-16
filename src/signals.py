"""シグナル検出（サイクル → シーケンス → 候補）。

SPEC.md §3 準拠。全バー Python ループは使わず、クロス列で分割した窓ごとに
条件インデックス配列への searchsorted で t_break / t_entry を解決する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.indicators import Indicators


@dataclass
class Candidate:
    """1サイクルで成立した1件のエントリー候補。"""

    direction: int  # +1 買い / -1 売り
    t_cross: int
    t_break: int
    t_entry: int
    entry: float
    sl: float
    tp: float


def _cross_indices(diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """diff 列から時刻順のクロス index と向き(+1/-1)を返す。

    上抜け: diff[t-1] <= 0 かつ diff[t] > 0
    下抜け: diff[t-1] >= 0 かつ diff[t] < 0
    NaN を含む比較はすべて False（ウォームアップは不成立）。
    """
    prev = diff[:-1]
    cur = diff[1:]
    with np.errstate(invalid="ignore"):
        up = (prev <= 0.0) & (cur > 0.0)
        dn = (prev >= 0.0) & (cur < 0.0)
    up_idx = np.flatnonzero(up) + 1
    dn_idx = np.flatnonzero(dn) + 1
    idx = np.concatenate([up_idx, dn_idx])
    direction = np.concatenate([
        np.ones(up_idx.size, dtype=np.int64),
        -np.ones(dn_idx.size, dtype=np.int64),
    ])
    order = np.argsort(idx, kind="mergesort")  # 同一 index は起こらないが安定ソート
    return idx[order], direction[order]


def detect_candidates(
    close: np.ndarray,
    ind: Indicators,
    rr: float = config.RR,
) -> list[Candidate]:
    """終値と指標から候補列を時刻順に返す（1サイクル最初の完成シーケンス1回のみ）。"""
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    diff = ind.mid - ind.sma400

    cross_idx, cross_dir = _cross_indices(diff)
    if cross_idx.size == 0:
        return []

    # 条件インデックス配列（NaN 比較は False → 自然に除外される）
    with np.errstate(invalid="ignore"):
        up_break = np.flatnonzero(close > ind.upper)   # 買い: 終値 > +2σ
        up_entry = np.flatnonzero(close < ind.mid)     # 買い: 終値 < mid
        dn_break = np.flatnonzero(close < ind.lower)   # 売り: 終値 < -2σ
        dn_entry = np.flatnonzero(close > ind.mid)     # 売り: 終値 > mid

    out: list[Candidate] = []
    m = cross_idx.size
    for i in range(m):
        s = int(cross_idx[i])
        e = int(cross_idx[i + 1]) if i + 1 < m else n
        d = int(cross_dir[i])

        if d == 1:
            break_arr, entry_arr = up_break, up_entry
        else:
            break_arr, entry_arr = dn_break, dn_entry

        # t_break: 窓 [s, e) 内で最初のブレイク足（クロス足 s を含む）
        pos = np.searchsorted(break_arr, s, side="left")
        if pos >= break_arr.size:
            continue
        t_break = int(break_arr[pos])
        if t_break >= e:
            continue

        # t_entry: [t_break+1, e) 内で最初のプルバック足
        pos2 = np.searchsorted(entry_arr, t_break + 1, side="left")
        if pos2 >= entry_arr.size:
            continue
        t_entry = int(entry_arr[pos2])
        if t_entry >= e:
            continue

        entry = float(close[t_entry])
        if d == 1:
            sl = float(ind.lower[t_entry])
            tp = entry + rr * (entry - sl)
        else:
            sl = float(ind.upper[t_entry])
            tp = entry - rr * (sl - entry)

        out.append(Candidate(
            direction=d,
            t_cross=s,
            t_break=t_break,
            t_entry=t_entry,
            entry=entry,
            sl=sl,
            tp=tp,
        ))
    return out
