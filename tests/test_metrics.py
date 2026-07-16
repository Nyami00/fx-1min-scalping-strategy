"""src.metrics のテスト（合成 analytics_fixture.csv を使用）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import metrics  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analytics_fixture.csv"


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


# --- Wilson CI --------------------------------------------------------------

def test_wilson_known_value():
    # 標準的な Wilson スコア区間（z=1.96）。SPEC 記載の (0.3557, 0.5477) は
    # 3dp まで一致する概算値で、正しい4dp値は (0.3561, 0.5476)。
    lo, hi = metrics.wilson_ci(45, 100)
    assert round(lo, 4) == 0.3561
    assert round(hi, 4) == 0.5476
    assert round(lo, 3) == 0.356 and round(hi, 3) == 0.548
    assert lo < 0.45 < hi


def test_wilson_edge_cases():
    lo, hi = metrics.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = metrics.wilson_ci(0, 10)
    assert lo >= -1e-9 and hi > lo  # 0成功では下限は解析的に0（浮動小数の微小負値を許容）
    lo, hi = metrics.wilson_ci(10, 10)
    assert hi <= 1.0 + 1e-12 and lo < hi


# --- full_stats -------------------------------------------------------------

def test_full_stats_cons_zero_handverified(trades):
    s = metrics.full_stats(trades, "r_net_cons_zero")
    # 60トレード中 eod=6 を除外 → 54。tp=24 のみ勝ち。
    assert s["n_total"] == 60
    assert s["n"] == 54
    assert s["wins"] == 24
    assert s["losses"] == 30
    assert s["win_rate"] == pytest.approx(24 / 54)
    # 期待値 = 4.5 / 54
    assert s["expectancy_r"] == pytest.approx(4.5 / 54, abs=1e-9)
    # PF = 36 / 31.5
    assert s["profit_factor"] == pytest.approx(36.0 / 31.5, rel=1e-6)
    assert s["avg_win_r"] == pytest.approx(1.5, rel=1e-6)
    assert s["avg_loss_r"] == pytest.approx(-1.05, rel=1e-6)
    assert s["payoff_ratio"] == pytest.approx(1.5 / 1.05, rel=1e-6)
    lo, hi = s["win_rate_ci"]
    assert lo < s["win_rate"] < hi
    # reason 内訳は全エントリー母数（eod が見える）
    assert s["reason_counts"]["eod"] == 6
    assert s["ambiguous_share"] == pytest.approx(6 / 60)


def test_full_stats_opt_zero_winrate(trades):
    s = metrics.full_stats(trades, "r_net_opt_zero")
    # 楽観では曖昧6件が TP 勝ち → 24 + 6 = 30 勝ち。
    assert s["wins"] == 30
    assert s["win_rate"] == pytest.approx(30 / 54)


def test_eod_exclusion_toggle(trades):
    inc = metrics.full_stats(trades, "r_net_cons_zero", exclude_eod=False)
    assert inc["n"] == 60
    exc = metrics.full_stats(trades, "r_net_cons_zero", exclude_eod=True)
    assert exc["n"] == 54


def test_full_stats_independent_recompute(trades):
    r_col = "r_net_cons_standard"
    sub = trades[trades["reason_cons"] != "eod"]
    r = sub[r_col].to_numpy()
    exp_wr = float((r > 0).mean())
    exp_exp = float(r.mean())
    exp_pf = float(r[r > 0].sum() / abs(r[r < 0].sum()))
    s = metrics.full_stats(trades, r_col)
    assert s["win_rate"] == pytest.approx(exp_wr)
    assert s["expectancy_r"] == pytest.approx(exp_exp)
    assert s["profit_factor"] == pytest.approx(exp_pf)
    assert s["hold_median_min"] == pytest.approx(float(np.median(sub["hold_minutes"])))
    assert s["hold_p90_min"] == pytest.approx(float(np.percentile(sub["hold_minutes"], 90)))


# --- stats_matrix -----------------------------------------------------------

def test_stats_matrix_structure(trades):
    m = metrics.stats_matrix(trades)
    overall = m[m["scope"] == "overall"]
    assert len(overall) == 6  # 2 variants x 3 scenarios
    pairs = m[m["scope"].str.startswith("pair:")]
    assert len(pairs) == 4 * 2  # 4 pairs x 2 primary columns
    years = m[m["scope"].str.startswith("year:")]
    assert len(years) == 3 * 2  # 2024/2025/2026H1 x 2 primary columns
    assert len(m) == 20
    assert {"win_rate", "expectancy_r", "profit_factor", "n_tp", "n_eod"} <= set(m.columns)
    # 2026 は部分年ラベル
    assert any("2026H1" in s for s in m["scope"].unique())
    # overall cons zero が full_stats と一致
    row = overall[(overall["variant"] == "cons") & (overall["scenario"] == "zero")].iloc[0]
    assert row["n"] == 54 and row["wins"] == 24


# --- streak_distribution ----------------------------------------------------

def test_streak_scripted_known_max():
    outcomes = np.array([True, False, False, False, True, False, True, False, False])
    d = metrics.streak_distribution(outcomes, n_shuffles=200, seed=7)
    assert d["observed_max_losing_streak"] == 3
    assert d["shuffle_p50"] <= d["shuffle_p95"] <= d["shuffle_p99"]


def test_streak_deterministic():
    outcomes = np.array([True, False, True, True, False, False, False, True, False, True])
    a = metrics.streak_distribution(outcomes, n_shuffles=300, seed=42)
    b = metrics.streak_distribution(outcomes, n_shuffles=300, seed=42)
    assert a == b


def test_streak_all_wins_and_empty():
    assert metrics.streak_distribution(np.array([True, True, True]), n_shuffles=10)[
        "observed_max_losing_streak"] == 0
    d = metrics.streak_distribution(np.array([], dtype=bool), n_shuffles=10)
    assert d["observed_max_losing_streak"] == 0


# --- rolling_winrate --------------------------------------------------------

def test_rolling_winrate_length_and_values():
    outcomes = np.array([True, True, False, True, False, False, True, True, True, False])
    rw = metrics.rolling_winrate(outcomes, window=3)
    assert rw.shape[0] == outcomes.shape[0]
    assert np.isnan(rw[0]) and np.isnan(rw[1])
    assert not np.isnan(rw[2])
    assert rw[2] == pytest.approx(2 / 3)


# --- first_n_table ----------------------------------------------------------

def test_first_n_table_chronological(trades):
    t = metrics.first_n_table(trades, n=50, r_col="r_net_cons_standard")
    assert list(t.columns) == ["date", "pair", "dir", "R", "cumR"]
    assert len(t) == 50
    dates = pd.to_datetime(t["date"])
    assert (dates.values[1:] >= dates.values[:-1]).all()
    assert t["cumR"].to_numpy() == pytest.approx(np.cumsum(t["R"].to_numpy()))
    # n が総数を超える場合は全件
    assert len(metrics.first_n_table(trades, n=1000, r_col="r_net_cons_zero")) == 60


# --- breakeven / required WR -------------------------------------------------

def test_breakeven_winrate():
    assert metrics.breakeven_winrate(1.5) == pytest.approx(0.4)
    assert metrics.breakeven_winrate(1.0) == pytest.approx(0.5)


def test_required_winrate_vs_spread(trades):
    df = metrics.required_winrate_vs_spread(trades)
    row_zero = df[df["scenario"] == "zero"].iloc[0]
    assert row_zero["mean_cost_r"] == pytest.approx(0.0)
    assert row_zero["required_wr"] == pytest.approx(0.4)
    row_std = df[df["scenario"] == "standard"].iloc[0]
    assert row_std["mean_cost_r"] > 0.0
    assert row_std["required_wr"] > 0.4
