"""スポットチェック CLI（検出済みトレード/スキップの独立再計算による照合）。

results の CSV とペア parquet を読み、サンプルした各トレードを **src.signals を使わず**
素朴な numpy 窓計算だけで独立に再現して全ルール成立を機械照合する。1件でも不一致なら
exit 1。results / parquet が無い場合は明確なメッセージを出して exit 0。

前提: 既定設定（entry = シグナル足終値）で生成された results を対象とする。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

_BB, _SMA, _DEV, _RR = config.BB_PERIOD, config.SMA_PERIOD, config.BB_DEV, config.RR
_PRICE_TOL = 1e-6


# ---- 独立再計算（pandas rolling も src.signals も使わない） -------------------

def _bands(close: np.ndarray, t: int):
    """index t の sma400 / mid / sigma(ddof=0) / upper / lower を直接窓計算で返す。"""
    sma400 = float(np.sum(close[t - _SMA + 1 : t + 1]) / _SMA) if t + 1 >= _SMA else float("nan")
    if t + 1 >= _BB:
        w = close[t - _BB + 1 : t + 1]
        mid = float(np.sum(w) / _BB)
        sigma = float(np.sqrt(np.sum((w - mid) ** 2) / _BB))
    else:
        mid = sigma = float("nan")
    return sma400, mid, sigma, mid + _DEV * sigma, mid - _DEV * sigma


def _diff(close: np.ndarray, t: int) -> float:
    sma400, mid, _, _, _ = _bands(close, t)
    return mid - sma400


def _is_cross(close: np.ndarray, t: int) -> bool:
    p, c = _diff(close, t - 1), _diff(close, t)
    if np.isnan(p) or np.isnan(c):
        return False
    return (p <= 0 < c) or (p >= 0 > c)


def _indep_exit(o, h, l, c, t_entry, direction, sl, tp):
    """t_entry+1 から前方走査し、最初に触れた足で優先順位判定（resolve_exit の独立実装）。"""
    n = len(c)
    for t in range(t_entry + 1, n):
        if direction == 1:
            if not (l[t] <= sl or h[t] >= tp):
                continue
            if o[t] <= sl:
                return t, o[t], "sl_gap", o[t], "sl_gap", False
            if l[t] <= sl and h[t] >= tp:
                return t, sl, "sl", tp, "tp", True
            if l[t] <= sl:
                return t, sl, "sl", sl, "sl", False
            return t, tp, "tp", tp, "tp", False
        else:
            if not (h[t] >= sl or l[t] <= tp):
                continue
            if o[t] >= sl:
                return t, o[t], "sl_gap", o[t], "sl_gap", False
            if h[t] >= sl and l[t] <= tp:
                return t, sl, "sl", tp, "tp", True
            if h[t] >= sl:
                return t, sl, "sl", sl, "sl", False
            return t, tp, "tp", tp, "tp", False
    return n - 1, c[n - 1], "eod", c[n - 1], "eod", False


# ---- 照合 -------------------------------------------------------------------

def _check_trade(row, df) -> tuple[bool, str]:
    o = df["open"].to_numpy("float64"); h = df["high"].to_numpy("float64")
    l = df["low"].to_numpy("float64"); c = df["close"].to_numpy("float64")
    d = int(row["direction"])
    tc, tb, te, tx = int(row["t_cross"]), int(row["t_break"]), int(row["t_entry"]), int(row["t_exit"])

    # 1) t_cross のクロス不等式
    dp, dc = _diff(c, tc - 1), _diff(c, tc)
    if d == 1 and not (dp <= 0 < dc):
        return False, f"cross ineq (long) failed: diff[{tc-1}]={dp:.3g}, diff[{tc}]={dc:.3g}"
    if d == -1 and not (dp >= 0 > dc):
        return False, f"cross ineq (short) failed: diff[{tc-1}]={dp:.3g}, diff[{tc}]={dc:.3g}"

    # 2) (t_cross, t_entry] に他のクロスが無い
    for u in range(tc + 1, te + 1):
        if _is_cross(c, u):
            return False, f"unexpected cross at {u} inside window"

    # 3) t_break は窓内で最初のブレイク足（クロス足含む）
    for u in range(tc, tb):
        _, _, _, up, lo = _bands(c, u)
        if (d == 1 and c[u] > up) or (d == -1 and c[u] < lo):
            return False, f"earlier breakout at {u} before t_break={tb}"
    _, _, _, up_b, lo_b = _bands(c, tb)
    if not ((d == 1 and c[tb] > up_b) or (d == -1 and c[tb] < lo_b)):
        return False, f"t_break={tb} is not a breakout"

    # 4) t_entry は t_break 後で最初のプルバック足
    for u in range(tb + 1, te):
        _, mid_u, _, _, _ = _bands(c, u)
        if (d == 1 and c[u] < mid_u) or (d == -1 and c[u] > mid_u):
            return False, f"earlier pullback at {u} before t_entry={te}"
    _, mid_e, _, up_e, lo_e = _bands(c, te)
    if not ((d == 1 and c[te] < mid_e) or (d == -1 and c[te] > mid_e)):
        return False, f"t_entry={te} is not a pullback"

    # 5) entry/sl/tp 算術
    entry = c[te]
    sl = lo_e if d == 1 else up_e
    tp = entry + _RR * (entry - sl) if d == 1 else entry - _RR * (sl - entry)
    if abs(entry - row["entry"]) > _PRICE_TOL:
        return False, f"entry mismatch {row['entry']} vs {entry}"
    if abs(sl - row["sl"]) > _PRICE_TOL:
        return False, f"sl mismatch {row['sl']} vs {sl}"
    if abs(tp - row["tp"]) > _PRICE_TOL:
        return False, f"tp mismatch {row['tp']} vs {tp}"

    # 6) t_entry+1 からの再走査で t_exit・約定価格・reason を照合
    et, pc, rc, po, ro, amb = _indep_exit(o, h, l, c, te, d, row["sl"], row["tp"])
    if et != tx:
        return False, f"t_exit mismatch {tx} vs {et}"
    if rc != row["reason_cons"] or ro != row["reason_opt"]:
        return False, f"reason mismatch ({row['reason_cons']}/{row['reason_opt']}) vs ({rc}/{ro})"
    if abs(pc - row["exit_price_cons"]) > _PRICE_TOL or abs(po - row["exit_price_opt"]) > _PRICE_TOL:
        return False, "exit price mismatch"
    if bool(amb) != bool(row["ambiguous"]):
        return False, f"ambiguous mismatch {row['ambiguous']} vs {amb}"
    return True, "ok"


def _check_skip(row, df, trades_by_pair) -> tuple[bool, str]:
    d = int(row["direction"])
    reason = str(row["reason"])
    try:
        te = int(df.index.get_loc(row["ts"]))
    except KeyError:
        return False, f"skip ts {row['ts']} not in parquet index"

    if reason == "invalid_sl":
        c = df["close"].to_numpy("float64")
        _, _, _, up, lo = _bands(c, te)
        entry, sl = c[te], (lo if d == 1 else up)
        genuine = (d == 1 and entry <= sl) or (d == -1 and entry >= sl)
        return (genuine, "ok" if genuine else f"invalid_sl not genuine: entry={entry}, sl={sl}")

    if reason == "position_open":
        tp_pair = trades_by_pair.get(row["pair"])
        if tp_pair is not None:
            overlap = ((tp_pair["t_entry"] <= te) & (te <= tp_pair["t_exit"])).any()
            if overlap:
                return True, "ok"
        return False, f"position_open at {te} overlaps no actual trade"

    return False, f"unknown skip reason {reason!r}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Independently re-verify sampled trades/skips.")
    ap.add_argument("--n-trades", type=int, default=config.SPOT_CHECK_TRADES)
    ap.add_argument("--n-skips", type=int, default=config.SPOT_CHECK_SKIPS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    args = ap.parse_args(argv)

    trades_path = args.out / "trades_all.csv"
    skips_path = args.out / "skips_all.csv"
    if not trades_path.exists():
        print(f"no results at {trades_path}; run scripts/run_backtest.py first. nothing to check.")
        return 0

    trades = pd.read_csv(trades_path, parse_dates=["ts_entry", "ts_exit"])
    skips = pd.read_csv(skips_path, parse_dates=["ts"]) if skips_path.exists() else pd.DataFrame()

    rng = np.random.default_rng(args.seed)
    t_sample = trades.sample(min(args.n_trades, len(trades)), random_state=rng) if len(trades) else trades
    s_sample = skips.sample(min(args.n_skips, len(skips)), random_state=rng) if len(skips) else skips

    from src.data.store import load_pair  # 遅延 import（data 層依存を隔離）

    needed = set(t_sample.get("pair", pd.Series(dtype=str))) | set(s_sample.get("pair", pd.Series(dtype=str)))
    parquet: dict[str, pd.DataFrame] = {}
    for pair in needed:
        try:
            parquet[pair] = load_pair(pair)
        except FileNotFoundError:
            print(f"parquet missing for {pair}; cannot spot-check. nothing verified.")
            return 0

    trades_by_pair = {p: g.reset_index(drop=True) for p, g in trades.groupby("pair")}

    fails = 0
    print("=" * 72)
    for _, row in t_sample.iterrows():
        ok, msg = _check_trade(row, parquet[row["pair"]])
        fails += not ok
        print(f"TRADE {row['pair']:>6} entry@{int(row['t_entry']):>7} dir={int(row['direction']):+d} "
              f"{row['reason_cons']:>6} : {'PASS' if ok else 'FAIL - ' + msg}")
    for _, row in s_sample.iterrows():
        ok, msg = _check_skip(row, parquet[row["pair"]], trades_by_pair)
        fails += not ok
        print(f"SKIP  {row['pair']:>6} {str(row['ts'])[:16]} dir={int(row['direction']):+d} "
              f"{row['reason']:>13} : {'PASS' if ok else 'FAIL - ' + msg}")
    print("=" * 72)
    print(f"trades checked: {len(t_sample)}  skips checked: {len(s_sample)}  FAILS: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
