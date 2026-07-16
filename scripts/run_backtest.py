"""バックテスト実行 CLI（シグナル検出 → トレード生成 → results 出力）。

各ペアについて load_pair → compute_indicators → detect_candidates → run_pair を回し、
全ペアを時系列マージしてコストを適用、凍結スキーマの CSV と要約 JSON を書き出す。

data 層への依存は main() 内の遅延 import に隔離する（エンジンは data 非依存）。
非既定 ddof / --entry-next-open は感度分析として results/sensitivity_{name}/ に出力する。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from src import config
from src.costs import FROZEN_COLUMNS, apply_costs
from src.execution import run_pair, skips_to_df, trades_to_df
from src.indicators import compute_indicators
from src.signals import Candidate, detect_candidates

_SKIP_REASONS = ("position_open", "invalid_sl")
_EXIT_REASONS = ("tp", "sl", "sl_gap", "eod")


def _apply_next_open(cands: list[Candidate], open_, n: int, rr: float) -> list[Candidate]:
    """--entry-next-open 用: entry を次足始値に、tp を再計算（sl は t_entry のバンド値のまま）。

    エグジット走査は t_entry+1（= 実約定足）を含む（run_pair が t_entry+1 から走査する）。
    """
    out: list[Candidate] = []
    for c in cands:
        j = c.t_entry + 1
        if j >= n:                    # 次足が無い候補は約定不能
            continue
        entry = float(open_[j])
        sl = c.sl
        tp = entry + rr * (entry - sl) if c.direction == 1 else entry - rr * (sl - entry)
        out.append(Candidate(c.direction, c.t_cross, c.t_break, c.t_entry, entry, sl, tp))
    return out


def _stats(tdf: pd.DataFrame, sdf: pd.DataFrame, bars: int, ncand: int) -> dict:
    """要約用の集計（bars/candidates/trades/skips/ambiguous/reasons/速報勝率）。"""
    n = len(tdf)
    non_eod = tdf[tdf["reason_cons"] != "eod"] if n else tdf

    def win_rate(col: str):
        if len(non_eod) == 0:
            return None
        return float((non_eod[col] > 0).mean())

    reasons = tdf["reason_cons"].value_counts().to_dict() if n else {}
    skip_counts = sdf["reason"].value_counts().to_dict() if len(sdf) else {}
    return {
        "bars": int(bars),
        "candidates": int(ncand),
        "trades": int(n),
        "skips": {k: int(skip_counts.get(k, 0)) for k in _SKIP_REASONS},
        "ambiguous": int(tdf["ambiguous"].sum()) if n else 0,
        "reasons": {k: int(reasons.get(k, 0)) for k in _EXIT_REASONS},
        "win_rate_cons": win_rate("r_gross_cons"),
        "win_rate_standard": win_rate("r_net_cons_standard"),
    }


def _out_dir(base: Path, ddof: int, entry_next_open: bool) -> Path:
    parts: list[str] = []
    if ddof != config.BB_DDOF:
        parts.append(f"ddof{ddof}")
    if entry_next_open:
        parts.append("next_open")
    return base / f"sensitivity_{'_'.join(parts)}" if parts else base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the 1-min scalping backtest.")
    ap.add_argument("--pairs", nargs="+", default=list(config.PAIRS),
                    help="対象ペア（既定: config.PAIRS 全20ペア）")
    ap.add_argument("--ddof", type=int, default=config.BB_DDOF,
                    help="BB 標準偏差の ddof（既定 0。非既定は感度分析出力）")
    ap.add_argument("--entry-next-open", action="store_true",
                    help="エントリーをシグナル足の次足始値にする感度分析")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR,
                    help="出力ルート（既定: config.RESULTS_DIR）")
    args = ap.parse_args(argv)

    # data 層は遅延 import（エンジンモジュールは src.data に依存しない）
    from src.data.store import load_pair

    pair_meta: dict[str, dict] = {}
    trade_frames: list[pd.DataFrame] = []
    skip_frames: list[pd.DataFrame] = []

    for pair in args.pairs:
        try:
            df = load_pair(pair)
        except FileNotFoundError:
            warnings.warn(f"parquet not found for {pair!r}; skipping")
            continue

        close = df["close"].to_numpy(dtype="float64")
        ind = compute_indicators(close, ddof=args.ddof)
        cands = detect_candidates(close, ind)
        n_cand = len(cands)
        if args.entry_next_open:
            cands = _apply_next_open(cands, df["open"].to_numpy(dtype="float64"), len(df), config.RR)

        trades, skips = run_pair(df, cands, pair)
        pair_meta[pair] = {"bars": len(df), "candidates": n_cand}
        trade_frames.append(trades_to_df(trades))
        skip_frames.append(skips_to_df(skips))

    if not pair_meta:
        print("no pairs produced output (missing parquet?)", file=sys.stderr)
        return 1

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else trades_to_df([])
    skips_df = pd.concat(skip_frames, ignore_index=True) if skip_frames else skips_to_df([])
    trades_df = apply_costs(trades_df).reindex(columns=FROZEN_COLUMNS)

    per_pair = {
        pair: _stats(trades_df[trades_df["pair"] == pair], skips_df[skips_df["pair"] == pair],
                     meta["bars"], meta["candidates"])
        for pair, meta in pair_meta.items()
    }
    aggregate = _stats(trades_df, skips_df,
                       sum(m["bars"] for m in pair_meta.values()),
                       sum(m["candidates"] for m in pair_meta.values()))
    summary = {
        "pairs": list(pair_meta),
        "ddof": args.ddof,
        "entry_next_open": bool(args.entry_next_open),
        "per_pair": per_pair,
        "aggregate": aggregate,
    }

    out_dir = _out_dir(args.out, args.ddof, args.entry_next_open)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "trades_all.csv", index=False)
    skips_df.to_csv(out_dir / "skips_all.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"wrote {len(trades_df)} trades / {len(skips_df)} skips to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
