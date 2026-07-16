"""統計メトリクス（勝率・期待値・PF・連敗分布・50トレード検証など）。

SPEC.md §8 の主張検証に必要な統計量を算出する。``reason == "eod"`` の
トレードは勝率・期待値の統計から除外し（SPEC §6）、活動量メタデータ
（reason 内訳・曖昧割合・トレード/日）は全エントリーを母数にする。

外部依存は numpy / pandas のみ（scipy 不使用）。Wilson 区間は手計算。
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd

from src import config


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項比率 wins/n の Wilson スコア95%信頼区間 (lo, hi) を返す（scipy不使用）。

    n == 0 のときは (nan, nan)。z は標準正規の分位点（既定 1.96 ≒ 95%）。
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (center - half, center + half)


def _variant_reason_col(r_col: str) -> str:
    """R列名から対応する reason 列名を返す（opt 系なら reason_opt）。"""
    return "reason_opt" if "_opt_" in r_col else "reason_cons"


def _as_bool(series: pd.Series) -> pd.Series:
    """True/False 文字列や 0/1 も含めて確実に bool 化する。"""
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def full_stats(trades: pd.DataFrame, r_col: str, exclude_eod: bool = True) -> dict:
    """1つの R 列に対する完全な統計量 dict を返す。

    パフォーマンス統計（N・勝敗・勝率・Wilson CI・期待値・PF・平均勝/敗・
    ペイオフ・保有時間分位）は eod 除外後の集合で算出。活動量メタデータ
    （reason 内訳・曖昧割合・トレード/日）は全エントリーを母数にする。
    win = r_net > 0、loss = r_net < 0。
    """
    reason_col = _variant_reason_col(r_col)
    n_total = len(trades)

    # 活動量メタデータ（全エントリー母数）
    reason_counts = {k: int(v) for k, v in trades[reason_col].value_counts().items()}
    ambiguous_share = float(_as_bool(trades["ambiguous"]).mean()) if n_total else 0.0
    if n_total:
        ts_all = pd.to_datetime(trades["ts_entry"], utc=True)
        span_days = int((ts_all.max().normalize() - ts_all.min().normalize()).days) + 1
        trades_per_day = float(n_total / span_days) if span_days > 0 else float(n_total)
    else:
        span_days = 0
        trades_per_day = 0.0

    # パフォーマンス統計（eod 除外後）
    stat = trades[trades[reason_col] != "eod"] if exclude_eod else trades
    n = len(stat)
    r = stat[r_col].to_numpy(dtype=np.float64)

    if n == 0:
        return {
            "n": 0, "n_total": int(n_total), "wins": 0, "losses": 0, "breakeven": 0,
            "win_rate": float("nan"), "win_rate_ci": [float("nan"), float("nan")],
            "expectancy_r": float("nan"), "profit_factor": float("nan"),
            "avg_win_r": float("nan"), "avg_loss_r": float("nan"),
            "payoff_ratio": float("nan"),
            "hold_median_min": float("nan"), "hold_mean_min": float("nan"),
            "hold_p90_min": float("nan"), "ambiguous_share": ambiguous_share,
            "reason_counts": reason_counts, "trades_per_day": trades_per_day,
            "span_days": int(span_days),
        }

    wins = int((r > 0.0).sum())
    losses = int((r < 0.0).sum())
    breakeven = int((r == 0.0).sum())
    win_rate = wins / n
    lo, hi = wilson_ci(wins, n)

    pos_sum = float(r[r > 0.0].sum())
    neg_sum = float(r[r < 0.0].sum())
    profit_factor = pos_sum / abs(neg_sum) if neg_sum < 0.0 else (
        float("inf") if pos_sum > 0.0 else float("nan"))
    avg_win_r = pos_sum / wins if wins else float("nan")
    avg_loss_r = neg_sum / losses if losses else float("nan")
    payoff_ratio = (avg_win_r / abs(avg_loss_r)) if (losses and avg_loss_r != 0.0) else float("nan")

    hold = stat["hold_minutes"].to_numpy(dtype=np.float64)
    return {
        "n": int(n), "n_total": int(n_total),
        "wins": wins, "losses": losses, "breakeven": breakeven,
        "win_rate": float(win_rate), "win_rate_ci": [float(lo), float(hi)],
        "expectancy_r": float(r.mean()),
        "profit_factor": float(profit_factor),
        "avg_win_r": float(avg_win_r), "avg_loss_r": float(avg_loss_r),
        "payoff_ratio": float(payoff_ratio),
        "hold_median_min": float(np.median(hold)),
        "hold_mean_min": float(hold.mean()),
        "hold_p90_min": float(np.percentile(hold, 90)),
        "ambiguous_share": ambiguous_share,
        "reason_counts": reason_counts,
        "trades_per_day": trades_per_day,
        "span_days": int(span_days),
    }


def _year_label(ts: pd.Series) -> pd.Series:
    """ts_entry から年ラベルを作る（2026 は部分年 '2026H1'）。"""
    years = pd.to_datetime(ts, utc=True).dt.year
    return years.map(lambda y: "2026H1" if y == 2026 else str(int(y)))


def _flatten(stats: dict, scope: str, variant: str, scenario: str, r_col: str) -> dict:
    """full_stats dict をスカラー行（reason_counts / CI を展開）に平坦化する。"""
    rc = stats.get("reason_counts", {})
    ci = stats.get("win_rate_ci", [float("nan"), float("nan")])
    return {
        "scope": scope, "variant": variant, "scenario": scenario, "r_col": r_col,
        "n": stats["n"], "n_total": stats["n_total"],
        "wins": stats["wins"], "losses": stats["losses"],
        "win_rate": stats["win_rate"],
        "win_rate_lo": ci[0], "win_rate_hi": ci[1],
        "expectancy_r": stats["expectancy_r"],
        "profit_factor": stats["profit_factor"],
        "avg_win_r": stats["avg_win_r"], "avg_loss_r": stats["avg_loss_r"],
        "payoff_ratio": stats["payoff_ratio"],
        "hold_median_min": stats["hold_median_min"],
        "hold_mean_min": stats["hold_mean_min"],
        "hold_p90_min": stats["hold_p90_min"],
        "ambiguous_share": stats["ambiguous_share"],
        "trades_per_day": stats["trades_per_day"],
        "n_tp": rc.get("tp", 0), "n_sl": rc.get("sl", 0),
        "n_sl_gap": rc.get("sl_gap", 0), "n_eod": rc.get("eod", 0),
    }


def stats_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    """{保守,楽観}×{zero,tight,standard} の全体統計に、主要2列のペア別・年別を加えた表。

    ペア別・年別は主要列 r_net_cons_standard と r_net_cons_zero について算出する。
    """
    rows: list[dict] = []

    # 全体: 6 バリアント×シナリオ
    for variant, scenario in product(("cons", "opt"), ("zero", "tight", "standard")):
        r_col = f"r_net_{variant}_{scenario}"
        rows.append(_flatten(full_stats(trades, r_col), "overall", variant, scenario, r_col))

    # 主要2列のペア別・年別
    primary = [("cons", "standard"), ("cons", "zero")]
    for variant, scenario in primary:
        r_col = f"r_net_{variant}_{scenario}"
        for pair, sub in trades.groupby("pair", sort=True):
            rows.append(_flatten(full_stats(sub, r_col), f"pair:{pair}", variant, scenario, r_col))
        labels = _year_label(trades["ts_entry"])
        for label, sub in trades.groupby(labels, sort=True):
            rows.append(_flatten(full_stats(sub, r_col), f"year:{label}", variant, scenario, r_col))

    return pd.DataFrame(rows)


def _max_false_run(outcomes: np.ndarray) -> int:
    """bool 配列（True=勝ち）における最長の False（連敗）の連続数を返す。"""
    x = ~np.asarray(outcomes, dtype=bool)
    if x.size == 0 or not x.any():
        return 0
    # 値が変化する境界で区切り、True(=負け)区間の最長 run を取る
    bounds = np.flatnonzero(np.concatenate(([True], x[1:] != x[:-1], [True])))
    run_lengths = np.diff(bounds)
    run_values = x[bounds[:-1]]
    loss_runs = run_lengths[run_values]
    return int(loss_runs.max()) if loss_runs.size else 0


def streak_distribution(
    outcomes: np.ndarray,
    n_shuffles: int = config.STREAK_SHUFFLES,
    seed: int = config.SEED,
) -> dict:
    """実測の最大連敗と、シャッフル分布の p50/p95/p99 を返す（決定論的）。

    outcomes は bool 配列（True=勝ち）。同じ勝敗の多重集合をシャッフルして
    最大連敗を n_shuffles 回サンプリングし、その分位点を求める。
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    observed = _max_false_run(outcomes)
    if outcomes.size == 0 or n_shuffles <= 0:
        return {
            "observed_max_losing_streak": observed,
            "shuffle_p50": 0.0, "shuffle_p95": 0.0, "shuffle_p99": 0.0,
            "n_shuffles": int(max(n_shuffles, 0)),
        }
    rng = np.random.default_rng(seed)
    sampled = np.empty(n_shuffles, dtype=np.int64)
    work = outcomes.copy()
    for i in range(n_shuffles):
        rng.shuffle(work)
        sampled[i] = _max_false_run(work)
    p50, p95, p99 = np.percentile(sampled, [50, 95, 99])
    return {
        "observed_max_losing_streak": int(observed),
        "shuffle_p50": float(p50), "shuffle_p95": float(p95), "shuffle_p99": float(p99),
        "shuffle_mean": float(sampled.mean()), "n_shuffles": int(n_shuffles),
    }


def rolling_winrate(outcomes: np.ndarray, window: int = 50) -> np.ndarray:
    """窓幅 window の移動勝率（入力と同長、先頭 window-1 個は NaN）。"""
    s = pd.Series(np.asarray(outcomes, dtype=bool).astype(np.float64))
    return s.rolling(window, min_periods=window).mean().to_numpy()


def first_n_table(trades: pd.DataFrame, n: int = 50, r_col: str = "r_net_cons_standard") -> pd.DataFrame:
    """時系列先頭 n トレードの date/pair/dir/R/cumR 表（動画の「50回検証」）。"""
    ts = pd.to_datetime(trades["ts_entry"], utc=True)
    order = np.argsort(ts.to_numpy(), kind="stable")
    sub = trades.iloc[order[:n]].reset_index(drop=True)
    r = sub[r_col].to_numpy(dtype=np.float64)
    return pd.DataFrame({
        "date": pd.to_datetime(sub["ts_entry"], utc=True).dt.normalize(),
        "pair": sub["pair"].to_numpy(),
        "dir": sub["direction"].to_numpy(),
        "R": r,
        "cumR": np.cumsum(r),
    })


def breakeven_winrate(rr: float = config.RR) -> float:
    """RR に対するコスト前の損益分岐勝率 = 1/(1+rr)（RR1.5 で 0.4）。"""
    return 1.0 / (1.0 + rr)


def required_winrate_vs_spread(
    trades: pd.DataFrame,
    scenarios=config.COST_SCENARIOS,
    rr: float = config.RR,
) -> pd.DataFrame:
    """シナリオ別に、コスト控除後の損益分岐勝率を返す。

    1トレードあたりのコスト（R単位）平均を c = mean(spread_price/risk_dist) とし、
    必要勝率 = (1 + c)/(1 + rr)。zero シナリオでは c=0 で 1/(1+rr) に一致する。
    """
    pairs = trades["pair"]
    pip = pairs.map(config.PIP_SIZE)
    risk_dist = trades["risk_dist"].to_numpy(dtype=np.float64)
    rows: list[dict] = []
    for scen in scenarios:
        spread_price = pairs.map(config.SPREADS_PIPS[scen]).to_numpy(dtype=np.float64) * \
            pip.to_numpy(dtype=np.float64)
        cost_r = spread_price / risk_dist
        c = float(np.mean(cost_r)) if cost_r.size else 0.0
        rows.append({
            "scenario": scen,
            "mean_cost_r": c,
            "required_wr": (1.0 + c) / (1.0 + rr),
        })
    return pd.DataFrame(rows)
