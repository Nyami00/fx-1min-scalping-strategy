"""図表生成（matplotlib, Agg）。各関数は PNG を保存し Path を返す。

英語ラベル・tab10 系の配色・seaborn 不使用。資産曲線/コスト別曲線/ドローダウン/
月次ヒートマップ/移動勝率/R分布/保有時間/ペア別期待値/MC合格率、および
セットアップのローソク足（自作。mplfinance 不使用）を提供する。

EquityResult はダック型で受け取り（.curve/.monthly 等）、src.equity は import しない。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

_DPI = 150
_TAB10 = list(plt.get_cmap("tab10").colors)
_SCEN_COLOR = {"zero": _TAB10[0], "tight": _TAB10[1], "standard": _TAB10[3]}
_UP_COLOR = "#2ca02c"
_DOWN_COLOR = "#d62728"
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save(fig, out: Path) -> Path:
    """共通の保存処理（親ディレクトリ作成・tight_layout・close）。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def _get(obj: Any, name: str):
    """dict でも属性オブジェクトでも値を取り出す。"""
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def plot_equity_modes(results: Mapping[str, Any], out: Path, log_scale: bool = False) -> Path:
    """3つの資金管理方式（fixed/tiered/pct）の資産曲線を重ね描きする。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mode, res) in enumerate(results.items()):
        curve = _get(res, "curve")
        if len(curve) == 0:
            continue
        ax.plot(pd.to_datetime(curve["ts"]), curve["equity"].to_numpy(),
                label=mode, color=_TAB10[i % 10], linewidth=1.3)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (JPY)")
    ax.set_title("Equity curves by money-management mode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out)


def plot_equity_cost_scenarios(trades: pd.DataFrame, out: Path) -> Path:
    """固定リスクの累積R（コスト3シナリオ×保守/楽観）を重ね描きする。"""
    ts = pd.to_datetime(trades["ts_entry"], utc=True)
    order = np.argsort(ts.to_numpy(), kind="stable")
    x = ts.to_numpy()[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    for scen in ("zero", "tight", "standard"):
        color = _SCEN_COLOR[scen]
        cons = np.cumsum(trades[f"r_net_cons_{scen}"].to_numpy(dtype=np.float64)[order])
        opt = np.cumsum(trades[f"r_net_opt_{scen}"].to_numpy(dtype=np.float64)[order])
        ax.plot(x, cons, color=color, linewidth=1.3, label=f"{scen} (cons)")
        ax.plot(x, opt, color=color, linewidth=1.1, linestyle="--", label=f"{scen} (opt)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative R (fixed risk)")
    ax.set_title("Cumulative R across cost scenarios (cons solid / opt dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, out)


def plot_drawdown(result: Any, out: Path) -> Path:
    """資産曲線からドローダウン(%)を描く。"""
    curve = _get(result, "curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(curve):
        eq = curve["equity"].to_numpy(dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (eq - peak) / peak * 100.0, 0.0)
        x = pd.to_datetime(curve["ts"])
        ax.fill_between(x, dd, 0.0, color=_TAB10[3], alpha=0.4)
        ax.plot(x, dd, color=_TAB10[3], linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def plot_monthly_heatmap(monthly: pd.DataFrame, out: Path) -> Path:
    """月次リターンを 年×月 のヒートマップ（RdYlGn, 0中心, %注記）で描く。"""
    idx = monthly.index
    if isinstance(idx, pd.PeriodIndex):
        years = idx.year.to_numpy()
        months = idx.month.to_numpy()
    else:
        dt = pd.to_datetime(idx)
        years = dt.year.to_numpy()
        months = dt.month.to_numpy()
    ret = monthly["ret"].to_numpy(dtype=np.float64) * 100.0

    uniq_years = sorted(set(int(y) for y in years))
    grid = np.full((len(uniq_years), 12), np.nan)
    yrow = {y: i for i, y in enumerate(uniq_years)}
    for y, m, r in zip(years, months, ret):
        grid[yrow[int(y)], int(m) - 1] = r

    finite = grid[np.isfinite(grid)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 0.7 * len(uniq_years) + 2))
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(_MONTHS)
    ax.set_yticks(range(len(uniq_years)))
    ax.set_yticklabels([str(y) for y in uniq_years])
    for i in range(len(uniq_years)):
        for j in range(12):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("Monthly returns (%)")
    fig.colorbar(im, ax=ax, label="Return (%)")
    return _save(fig, out)


def plot_rolling_winrate(outcomes: np.ndarray, out: Path, window: int = 50,
                         breakeven: float = 0.40) -> Path:
    """窓幅 window の移動勝率と損益分岐勝率(既定40%)の水平線を描く。"""
    s = pd.Series(np.asarray(outcomes, dtype=bool).astype(np.float64))
    rw = s.rolling(window, min_periods=window).mean().to_numpy() * 100.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, rw.size + 1), rw, color=_TAB10[0], linewidth=1.2,
            label=f"rolling win-rate (w={window})")
    ax.axhline(breakeven * 100.0, color=_DOWN_COLOR, linestyle="--",
               label=f"breakeven {breakeven * 100:.0f}%")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Rolling win rate")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out)


def plot_r_distribution(r: np.ndarray, out: Path, bins: int = 40) -> Path:
    """R（純益、R倍数）のヒストグラムを描く。"""
    r = np.asarray(r, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    if r.size:
        ax.hist(r, bins=bins, color=_TAB10[0], alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="gray", linewidth=1.0)
    ax.set_xlabel("R (net)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of net R")
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def plot_holding_time(hold_minutes: np.ndarray, out: Path, bins: int = 40) -> Path:
    """保有時間（分）のヒストグラムを描く。"""
    h = np.asarray(hold_minutes, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    if h.size:
        ax.hist(h, bins=bins, color=_TAB10[2], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Holding time (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Holding-time distribution")
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def plot_per_pair_expectancy(labels: Sequence[str], expectancy: Sequence[float], out: Path) -> Path:
    """ペア別の期待値R（正=緑 / 負=赤）を横棒で描く。"""
    labels = list(labels)
    vals = np.asarray(expectancy, dtype=np.float64)
    colors = [_UP_COLOR if v >= 0 else _DOWN_COLOR for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels) + 1)))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Expectancy (R)")
    ax.set_title("Per-pair expectancy")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, out)


def plot_mc_pass(ladder: pd.DataFrame, out: Path) -> Path:
    """MCラダーの P1/P2 合格率をリスク・サンプリング別のグループ棒で描く。"""
    labels = [f"{r * 100:.0f}% {s}" for r, s in zip(ladder["risk_pct"], ladder["sampling"])]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 2), 5))
    ax.bar(x - w / 2, ladder["p1_pass_rate"].to_numpy() * 100.0, w,
           label="P1 (+10%)", color=_TAB10[0])
    ax.bar(x + w / 2, ladder["p2_pass_rate"].to_numpy() * 100.0, w,
           label="P2 (+5%)", color=_TAB10[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Prop-challenge pass rate by risk / sampling")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out)


def plot_setup_candles(bars_df: pd.DataFrame, ind_arrays: Any, trade_row: Any, out: Path) -> Path:
    """1トレード周辺のローソク足を自作描画（BB(20,2)+SMA400、entry/SL/TP）。

    bars_df: open/high/low/close を含む窓（~120本）。ind_arrays: 同窓にスライス済みの
    sma400/mid/upper/lower（dict または属性）。trade_row: entry/sl/tp/direction と、
    窓内でのエントリー位置 entry_pos（あれば）を持つ dict/Series。
    """
    o = bars_df["open"].to_numpy(dtype=np.float64)
    h = bars_df["high"].to_numpy(dtype=np.float64)
    low = bars_df["low"].to_numpy(dtype=np.float64)
    c = bars_df["close"].to_numpy(dtype=np.float64)
    n = c.shape[0]
    x = np.arange(n)
    up = c >= o

    fig, ax = plt.subplots(figsize=(12, 6))
    # ヒゲ（高安）
    ax.vlines(x[up], low[up], h[up], color=_UP_COLOR, linewidth=0.8)
    ax.vlines(x[~up], low[~up], h[~up], color=_DOWN_COLOR, linewidth=0.8)
    # 実体（始値-終値の矩形）
    body_up = np.maximum(c[up] - o[up], 1e-12)
    ax.bar(x[up], body_up, bottom=o[up], width=0.6, color=_UP_COLOR, edgecolor=_UP_COLOR)
    body_dn = np.maximum(o[~up] - c[~up], 1e-12)
    ax.bar(x[~up], body_dn, bottom=c[~up], width=0.6, color=_DOWN_COLOR, edgecolor=_DOWN_COLOR)

    # 指標オーバーレイ
    def _ia(name):
        arr = ind_arrays[name] if isinstance(ind_arrays, Mapping) else getattr(ind_arrays, name)
        return np.asarray(arr, dtype=np.float64)

    ax.plot(x, _ia("sma400"), color=_TAB10[4], linewidth=1.2, label="SMA400")
    ax.plot(x, _ia("mid"), color=_TAB10[0], linewidth=1.0, label="BB mid (SMA20)")
    ax.plot(x, _ia("upper"), color=_TAB10[7], linewidth=0.9, linestyle="--", label="BB +2σ")
    ax.plot(x, _ia("lower"), color=_TAB10[7], linewidth=0.9, linestyle="--", label="BB -2σ")

    def _rg(name, default=None):
        if isinstance(trade_row, Mapping):
            return trade_row.get(name, default)
        return getattr(trade_row, name, default)

    entry = float(_rg("entry"))
    sl = float(_rg("sl"))
    tp = float(_rg("tp"))
    ax.axhline(entry, color="black", linewidth=1.0, label="entry")
    ax.axhline(sl, color=_DOWN_COLOR, linewidth=1.0, linestyle=":", label="SL")
    ax.axhline(tp, color=_UP_COLOR, linewidth=1.0, linestyle=":", label="TP")

    entry_pos = _rg("entry_pos")
    if entry_pos is not None and 0 <= int(entry_pos) < n:
        ax.scatter([int(entry_pos)], [entry], marker="^" if int(_rg("direction", 1)) == 1 else "v",
                   color="black", zorder=5, s=80)

    pair = _rg("pair", "")
    direction = int(_rg("direction", 1))
    dlabel = "LONG" if direction == 1 else "SHORT"
    ax.set_title(f"Setup: {pair} {dlabel}")
    ax.set_xlabel("Bar (window index)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out)
