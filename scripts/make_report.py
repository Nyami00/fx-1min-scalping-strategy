"""レポート生成 CLI（統計・表・図を results/ に出力）。report.md は書かない。

results/trades_all.csv（+ skips_all.csv, summary.json）を読み、統計行列・連敗分布・
移動勝率・先頭50トレード・損益分岐表・資産曲線シム・プロップMCラダーを計算し、
results/stats_all.json / results/tables.md / results/charts/*.png を出力する。

src.equity・src.data・src.indicators は main() 内で遅延 import する（マージ前でも
本スクリプトの import 自体は失敗しないため）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config, metrics, montecarlo, report  # noqa: E402


# --- JSON / Markdown ヘルパ -------------------------------------------------

def _json_safe(obj):
    """inf/nan や numpy 型を JSON 可能な値へ再帰変換する。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _fmt(v) -> str:
    """Markdown セル1つの整形。"""
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def _df_to_md(df: pd.DataFrame) -> str:
    """DataFrame を Markdown 表へ（tabulate 非依存の自前実装）。"""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# --- ロード -----------------------------------------------------------------

def _load_trades(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "trades_all.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "ambiguous" in df.columns and df["ambiguous"].dtype != bool:
        df["ambiguous"] = df["ambiguous"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    return df


# --- セットアップ・ローソク足のサンプル選定 --------------------------------

def _pick_samples(trades: pd.DataFrame) -> list[int]:
    """long/short × 勝/負 と曖昧足を含む多様な5トレードの行インデックスを返す。"""
    picks: list[int] = []

    def add(mask):
        idx = trades.index[mask]
        for i in idx:
            if i not in picks:
                picks.append(int(i))
                return

    win_c = trades["r_net_cons_standard"] > 0
    add((trades["direction"] == 1) & win_c & (~trades["ambiguous"]))
    add((trades["direction"] == -1) & win_c & (~trades["ambiguous"]))
    add((trades["direction"] == 1) & (~win_c) & (~trades["ambiguous"]))
    add((trades["direction"] == -1) & (~win_c) & (~trades["ambiguous"]))
    add(trades["ambiguous"] == True)  # noqa: E712
    # 不足分は先頭から補完
    for i in trades.index:
        if len(picks) >= 5:
            break
        if int(i) not in picks:
            picks.append(int(i))
    return picks[:5]


def _make_setup_charts(trades: pd.DataFrame, charts_dir: Path) -> list[str]:
    """サンプルトレードのローソク足PNGを生成（データ未整備なら安全にスキップ）。"""
    try:
        from src.data import store  # 遅延 import
        from src.indicators import compute_indicators
    except Exception as e:  # pragma: no cover
        print(f"[setup] indicators/store を import できないためスキップ: {e}")
        return []

    out_files: list[str] = []
    for k, ridx in enumerate(_pick_samples(trades)):
        row = trades.loc[ridx]
        pair = str(row["pair"])
        t_entry = int(row["t_entry"])
        try:
            bars = store.load_pair(pair)
        except Exception as e:
            print(f"[setup] {pair} のバー読込に失敗、スキップ: {e}")
            continue
        n = len(bars)
        start = max(0, t_entry - 100)
        stop = min(n, t_entry + 20)
        window = bars.iloc[start:stop]
        ind = compute_indicators(bars["close"].to_numpy(dtype=np.float64))
        ind_arrays = {
            "sma400": ind.sma400[start:stop], "mid": ind.mid[start:stop],
            "upper": ind.upper[start:stop], "lower": ind.lower[start:stop],
        }
        trow = {
            "entry": float(row["entry"]), "sl": float(row["sl"]), "tp": float(row["tp"]),
            "direction": int(row["direction"]), "pair": pair,
            "entry_pos": t_entry - start,
        }
        out = charts_dir / f"setup_{k + 1}_{pair}.png"
        report.plot_setup_candles(window, ind_arrays, trow, out)
        out_files.append(out.name)
    return out_files


# --- メイン -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate analytics report artifacts (no report.md).")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR,
                   help="results ディレクトリ（入力 trades_all.csv と各出力の場所）")
    p.add_argument("--scenario", default="standard", choices=list(config.COST_SCENARIOS),
                   help="主要シナリオ（連敗/移動勝率/先頭50/MC で使用）")
    p.add_argument("--variant", default="cons", choices=["cons", "opt"],
                   help="主要バリアント（保守/楽観）")
    p.add_argument("--mc-sims", type=int, default=config.MC_SIMS, help="MC 試行数")
    p.add_argument("--mc-max-days", type=int, default=1000, help="MC の最大日数")
    args = p.parse_args(argv)

    results_dir: Path = args.out
    charts_dir = results_dir / "charts"
    primary_col = f"r_net_{args.variant}_{args.scenario}"
    mm_col = "r_net_cons_standard"

    trades = _load_trades(results_dir)
    if trades is None:
        print(f"trades_all.csv が {results_dir} にありません。先に "
              f"`python scripts/run_backtest.py` を実行してください。何もせず終了します。")
        return 0
    if len(trades) == 0:
        print("trades_all.csv が空です。何もせず終了します。")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 任意入力（存在すれば読む）
    skips_path = results_dir / "skips_all.csv"
    skips = pd.read_csv(skips_path) if skips_path.exists() else None
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    print(f"trades={len(trades)}  primary={primary_col}  out={results_dir}")

    # --- 統計 ---------------------------------------------------------------
    matrix = metrics.stats_matrix(trades)
    non_eod = trades[trades[metrics._variant_reason_col(primary_col)] != "eod"]
    ts_order = np.argsort(pd.to_datetime(non_eod["ts_entry"], utc=True).to_numpy(), kind="stable")
    outcomes = (non_eod[primary_col].to_numpy(dtype=np.float64)[ts_order] > 0.0)

    streaks = metrics.streak_distribution(outcomes)
    rolling = metrics.rolling_winrate(outcomes, window=50)
    first50 = metrics.first_n_table(trades, n=50, r_col=primary_col)
    breakeven_tbl = metrics.required_winrate_vs_spread(trades)
    breakeven_wr = metrics.breakeven_winrate()

    # --- 資産曲線シム（遅延 import）----------------------------------------
    from src import equity  # noqa: E402  遅延

    mm_results = {m: equity.simulate(trades, mode=m, r_col=mm_col)
                  for m in ("fixed", "tiered", "pct")}
    # コスト別: 固定リスクで6列すべて
    cost_final = {}
    for variant in ("cons", "opt"):
        for scen in config.COST_SCENARIOS:
            col = f"r_net_{variant}_{scen}"
            cost_final[col] = float(equity.simulate(trades, mode="fixed", r_col=col).final_equity)

    # --- MC ラダー ----------------------------------------------------------
    ladder = montecarlo.run_challenge_ladder(
        trades, primary_col, n_sims=args.mc_sims, max_days=args.mc_max_days)

    # --- stats_all.json -----------------------------------------------------
    def _eq_summary(res) -> dict:
        return {
            "final_equity": float(res.final_equity),
            "max_dd_jpy": float(res.max_dd_jpy),
            "max_dd_pct": float(res.max_dd_pct),
            "longest_underwater_days": float(res.longest_underwater_days),
            "max_concurrent": int(res.max_concurrent),
            "max_open_risk_jpy": float(res.max_open_risk_jpy),
            "ruined": bool(res.ruined),
            "ruin_ts": (res.ruin_ts.isoformat() if res.ruin_ts is not None else None),
            "monthly": [
                {"month": str(idx), "ret": (None if pd.isna(r) else float(r))}
                for idx, r in res.monthly["ret"].items()
            ],
        }

    stats_all = {
        "meta": {
            "primary_col": primary_col, "mm_col": mm_col,
            "variant": args.variant, "scenario": args.scenario,
            "n_trades": int(len(trades)),
            "n_skips": (int(len(skips)) if skips is not None else None),
            "breakeven_winrate": breakeven_wr,
            "summary_present": summary is not None,
        },
        "stats_matrix": [_json_safe(r) for r in matrix.to_dict(orient="records")],
        "streaks": _json_safe(streaks),
        "required_winrate_vs_spread": [_json_safe(r)
                                       for r in breakeven_tbl.to_dict(orient="records")],
        "first50": [
            {"date": pd.Timestamp(d).isoformat(), "pair": pr, "dir": int(di),
             "R": float(rr), "cumR": float(cr)}
            for d, pr, di, rr, cr in zip(
                first50["date"], first50["pair"], first50["dir"],
                first50["R"], first50["cumR"])
        ],
        "equity_modes": {m: _eq_summary(r) for m, r in mm_results.items()},
        "cost_scenarios_fixed_final_equity": _json_safe(cost_final),
        "mc_ladder": [_json_safe(r) for r in ladder.to_dict(orient="records")],
    }
    (results_dir / "stats_all.json").write_text(
        json.dumps(stats_all, ensure_ascii=False, indent=2))
    print("wrote stats_all.json")

    # --- tables.md ----------------------------------------------------------
    md: list[str] = ["# Analytics tables", ""]
    md += ["## Overall stats (variant x scenario)", ""]
    overall_cols = ["variant", "scenario", "n", "wins", "losses", "win_rate",
                    "win_rate_lo", "win_rate_hi", "expectancy_r", "profit_factor",
                    "avg_win_r", "avg_loss_r", "payoff_ratio", "hold_median_min",
                    "hold_p90_min", "ambiguous_share", "trades_per_day"]
    md += [_df_to_md(matrix[matrix["scope"] == "overall"][overall_cols]), ""]
    md += ["## Per-pair (cons, standard)", ""]
    pp = matrix[(matrix["scope"].str.startswith("pair:")) & (matrix["scenario"] == "standard")]
    md += [_df_to_md(pp[["scope", "n", "win_rate", "expectancy_r", "profit_factor"]]), ""]
    md += ["## Per-year (cons, standard)", ""]
    py = matrix[(matrix["scope"].str.startswith("year:")) & (matrix["scenario"] == "standard")]
    md += [_df_to_md(py[["scope", "n", "win_rate", "expectancy_r", "profit_factor"]]), ""]
    md += ["## Losing-streak distribution", "",
           f"- observed max losing streak: {streaks['observed_max_losing_streak']}",
           f"- shuffled p50/p95/p99: {streaks['shuffle_p50']:.1f} / "
           f"{streaks['shuffle_p95']:.1f} / {streaks['shuffle_p99']:.1f}", ""]
    md += ["## Required win-rate vs spread", "",
           f"- gross breakeven win-rate (RR {config.RR}): {breakeven_wr:.4f}",
           _df_to_md(breakeven_tbl), ""]
    md += ["## Equity modes summary", ""]
    eq_rows = pd.DataFrame([
        {"mode": m, "final_equity": r.final_equity, "max_dd_pct": r.max_dd_pct,
         "max_dd_jpy": r.max_dd_jpy, "longest_uw_days": r.longest_underwater_days,
         "max_concurrent": r.max_concurrent, "ruined": r.ruined}
        for m, r in mm_results.items()
    ])
    md += [_df_to_md(eq_rows), ""]
    md += ["## First 50 trades (chronological)", ""]
    f50 = first50.copy()
    f50["date"] = pd.to_datetime(f50["date"]).dt.strftime("%Y-%m-%d")
    md += [_df_to_md(f50), ""]
    md += ["## Prop-challenge MC ladder", "", _df_to_md(ladder), ""]
    (results_dir / "tables.md").write_text("\n".join(md))
    print("wrote tables.md")

    # --- 図 -----------------------------------------------------------------
    report.plot_equity_modes(mm_results, charts_dir / "equity_modes.png")
    report.plot_equity_modes(mm_results, charts_dir / "equity_modes_log.png", log_scale=True)
    report.plot_equity_cost_scenarios(trades, charts_dir / "equity_cost_scenarios.png")
    report.plot_drawdown(mm_results["tiered"], charts_dir / "drawdown.png")
    report.plot_monthly_heatmap(mm_results["pct"].monthly, charts_dir / "monthly_heatmap.png")
    report.plot_rolling_winrate(outcomes, charts_dir / "rolling_winrate.png", window=50,
                                breakeven=breakeven_wr)
    report.plot_r_distribution(non_eod[primary_col].to_numpy(dtype=np.float64),
                               charts_dir / "r_distribution.png")
    report.plot_holding_time(non_eod["hold_minutes"].to_numpy(dtype=np.float64),
                             charts_dir / "holding_time.png")
    pp_all = matrix[(matrix["scope"].str.startswith("pair:")) & (matrix["scenario"] == "standard")]
    report.plot_per_pair_expectancy(
        [s.replace("pair:", "") for s in pp_all["scope"]],
        pp_all["expectancy_r"].to_numpy(dtype=np.float64),
        charts_dir / "per_pair_expectancy.png")
    report.plot_mc_pass(ladder, charts_dir / "mc_pass.png")
    setups = _make_setup_charts(trades, charts_dir)
    print(f"wrote {10 + len(setups)} charts to {charts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
