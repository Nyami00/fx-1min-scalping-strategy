"""コスト適用（往復スプレッド1回分を R 換算で控除）。

SPEC.md §5 準拠。BID ローソクのみで計算し、シナリオ別に価格ベースのスプレッドを
risk_dist で割って R から差し引く。
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.execution import TRADE_COLUMNS

# 出力トレード表の凍結スキーマ（SPEC.md §6）。
FROZEN_COLUMNS: list[str] = TRADE_COLUMNS + [
    "r_net_cons_zero", "r_net_cons_tight", "r_net_cons_standard",
    "r_net_opt_zero", "r_net_opt_tight", "r_net_opt_standard",
]


def apply_costs(trades: pd.DataFrame) -> pd.DataFrame:
    """r_net_{cons,opt}_{scenario} の6列を追加して返す（ペア別ベクトル化）。

    r_net = r_gross - (SPREADS_PIPS[scenario][pair] * PIP_SIZE[pair]) / risk_dist
    列順は凍結スキーマに合わせ、全 cons を先に、全 opt を後に並べる。
    """
    out = trades.copy()
    pairs = out["pair"]
    pip = pairs.map(config.PIP_SIZE)
    risk_dist = out["risk_dist"]

    cost_r: dict[str, pd.Series] = {}
    for scen in config.COST_SCENARIOS:
        spread_price = pairs.map(config.SPREADS_PIPS[scen]) * pip
        cost_r[scen] = spread_price / risk_dist

    for scen in config.COST_SCENARIOS:
        out[f"r_net_cons_{scen}"] = out["r_gross_cons"] - cost_r[scen]
    for scen in config.COST_SCENARIOS:
        out[f"r_net_opt_{scen}"] = out["r_gross_opt"] - cost_r[scen]
    return out
