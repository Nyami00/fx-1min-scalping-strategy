"""src.montecarlo のテスト（プロップチャレンジMC）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config, montecarlo  # noqa: E402
from src.montecarlo import LADDER_COLUMNS, MCResult, prop_challenge_mc, run_challenge_ladder  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analytics_fixture.csv"


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


def test_determinism_fixed_seed():
    rs = np.array([-1.0, 1.5, -1.0, 1.5, -1.0])
    tpd = np.array([3, 3, 3])
    a = prop_challenge_mc(rs, tpd, 0.01, 0.10, n_sims=200, seed=42, max_days=60)
    b = prop_challenge_mc(rs, tpd, 0.01, 0.10, n_sims=200, seed=42, max_days=60)
    assert a == b
    # 異なるシードでは（一般に）別の結果になり得る（同一性は要求しない）
    assert isinstance(a, MCResult)


def test_all_losses_pass_zero_with_daily_breaches():
    rs = np.full(5, -1.0)
    tpd = np.array([6, 6, 6])  # 1日6トレード → 5トレードで日次-5%到達
    res = prop_challenge_mc(rs, tpd, 0.01, 0.10, n_sims=300, seed=1, max_days=100)
    assert res.pass_rate == 0.0
    assert res.breach_daily_rate == pytest.approx(1.0)
    assert res.breach_maxdd_rate == 0.0


def test_all_wins_pass_one():
    rs = np.full(5, 1.5)
    tpd = np.array([6, 6, 6])
    res = prop_challenge_mc(rs, tpd, 0.01, 0.10, n_sims=300, seed=2, max_days=100)
    assert res.pass_rate == pytest.approx(1.0)
    assert res.breach_daily_rate == 0.0
    assert res.breach_maxdd_rate == 0.0
    # 日2件目(=8トレード目)で +10.5% 到達 → 2日目に合格
    assert res.median_days_to_pass == pytest.approx(2.0)


def test_block_mode_runs():
    rs = np.array([-1.0, -1.0, 2.0, -1.0, 1.5])
    tpd = np.array([2, 0, 5, 1, 3, 0, 4])
    res = prop_challenge_mc(rs, tpd, 0.01, 0.10, n_sims=150, seed=3, block=3, max_days=80)
    assert isinstance(res, MCResult)
    assert 0.0 <= res.pass_rate <= 1.0
    assert res.pass_rate + res.fail_rate == pytest.approx(1.0)


def test_risk_scaling_daily_breach_more_at_2pct():
    # 1日4トレード・±1R の分布。1%では日中最大-4%で日次上限-5%に到達不能→違反0。
    # 2%では-2.5Rで到達可能となり日次違反が発生する（高リスクほど日次違反が多い）。
    rs = np.array([-1.0, 1.0])
    tpd = np.array([4, 4, 4])
    r1 = prop_challenge_mc(rs, tpd, 0.01, 0.05, n_sims=500, seed=5, max_days=60)
    r2 = prop_challenge_mc(rs, tpd, 0.02, 0.05, n_sims=500, seed=5, max_days=60)
    assert r1.breach_daily_rate == 0.0
    assert r2.breach_daily_rate > r1.breach_daily_rate


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        prop_challenge_mc(np.array([]), np.array([1]), 0.01, 0.10, n_sims=10)
    with pytest.raises(ValueError):
        prop_challenge_mc(np.array([1.0]), np.array([]), 0.01, 0.10, n_sims=10)


def test_ladder_shape_and_columns(trades):
    df = run_challenge_ladder(
        trades, "r_net_cons_standard", n_sims=100, seed=42, block=config.MC_BLOCK, max_days=40)
    assert list(df.columns) == LADDER_COLUMNS
    assert len(df) == len(config.PROP_RISK_VARIANTS) * 2
    assert set(df["sampling"]) == {"iid", "block"}
    assert set(df["risk_pct"]) == set(config.PROP_RISK_VARIANTS)
    assert ((df["pass_endtoend"] >= 0.0) & (df["pass_endtoend"] <= 1.0)).all()


def test_ladder_deterministic(trades):
    a = run_challenge_ladder(trades, "r_net_cons_zero", n_sims=80, seed=11, max_days=30)
    b = run_challenge_ladder(trades, "r_net_cons_zero", n_sims=80, seed=11, max_days=30)
    pd.testing.assert_frame_equal(a, b)
