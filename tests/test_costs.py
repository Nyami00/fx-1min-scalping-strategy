"""コスト適用のユニットテスト（SPEC.md §5）。

往復スプレッド1回分を価格ベースで R 換算控除する。JPY クオート（pip 0.01）と
非 JPY（pip 0.0001）の双方で 12 桁精度を確認し、zero シナリオが gross に一致する
ことを検証する。
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.costs import apply_costs, FROZEN_COLUMNS


def _trades() -> pd.DataFrame:
    return pd.DataFrame({
        "pair": ["USDJPY", "EURUSD"],
        "risk_dist": [0.20, 0.0020],          # ともに 20 pips 相当
        "r_gross_cons": [1.5, -1.0],
        "r_gross_opt": [1.5, 1.5],
    })


def test_spread_to_r_conversion_12_decimals():
    """r_net = r_gross - spread_price/risk_dist を全シナリオ・両ペアで 12 桁一致させる。"""
    out = apply_costs(_trades())
    for i, pair in enumerate(out["pair"]):
        rd = out["risk_dist"].iloc[i]
        for scen in config.COST_SCENARIOS:
            cost = config.SPREADS_PIPS[scen][pair] * config.PIP_SIZE[pair] / rd
            exp_cons = out["r_gross_cons"].iloc[i] - cost
            exp_opt = out["r_gross_opt"].iloc[i] - cost
            assert out[f"r_net_cons_{scen}"].iloc[i] == pytest.approx(exp_cons, abs=1e-12, rel=0)
            assert out[f"r_net_opt_{scen}"].iloc[i] == pytest.approx(exp_opt, abs=1e-12, rel=0)


def test_zero_scenario_equals_gross_exactly():
    """zero シナリオはスプレッド 0 なので gross と完全一致する。"""
    out = apply_costs(_trades())
    assert (out["r_net_cons_zero"] == out["r_gross_cons"]).all()
    assert (out["r_net_opt_zero"] == out["r_gross_opt"]).all()


def test_known_values_usdjpy_and_eurusd():
    """具体値: USDJPY standard=0.7pip, EURUSD standard=0.6pip の控除量を確認。"""
    out = apply_costs(_trades())
    # USDJPY: 0.7 * 0.01 / 0.20 = 0.035 → 1.5 - 0.035 = 1.465
    assert out["r_net_cons_standard"].iloc[0] == pytest.approx(1.465, abs=1e-12)
    # EURUSD: 0.6 * 0.0001 / 0.0020 = 0.03 → cons: -1.0-0.03=-1.03 / opt: 1.5-0.03=1.47
    assert out["r_net_cons_standard"].iloc[1] == pytest.approx(-1.03, abs=1e-12)
    assert out["r_net_opt_standard"].iloc[1] == pytest.approx(1.47, abs=1e-12)


def test_appended_columns_in_frozen_order():
    """付加される6列が凍結スキーマ順（全 cons → 全 opt）で並ぶ。"""
    out = apply_costs(_trades())
    assert list(out.columns[-6:]) == FROZEN_COLUMNS[-6:] == [
        "r_net_cons_zero", "r_net_cons_tight", "r_net_cons_standard",
        "r_net_opt_zero", "r_net_opt_tight", "r_net_opt_standard",
    ]
    # 追加後もコスト列に NaN は無い
    assert not out[FROZEN_COLUMNS[-6:]].isna().to_numpy().any()
