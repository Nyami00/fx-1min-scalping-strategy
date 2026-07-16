"""プロップファーム・チャレンジのモンテカルロ（FTMO型近似、SPEC.md §8）。

実測 r_net 分布と実測トレード数/日をブートストラップ（iid またはブロック）して、
各フェーズの合格確率・所要日数分布・違反内訳を推定する。リスクはフェーズ初期残高
に対する固定割合（複利なし、FTMO慣行）: pnl = risk_pct * start_balance * r。

決定論性は numpy Generator（固定シード）で担保する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class MCResult:
    """1フェーズ・1条件のモンテカルロ結果。"""

    pass_rate: float
    fail_rate: float
    median_days_to_pass: float
    p90_days_to_pass: float
    breach_daily_rate: float
    breach_maxdd_rate: float


def prop_challenge_mc(
    r_samples: np.ndarray,
    trades_per_day: np.ndarray,
    risk_pct: float,
    profit_target: float,
    daily_loss: float = config.PROP_DAILY_LOSS,
    max_dd: float = config.PROP_MAX_DD,
    n_sims: int = config.MC_SIMS,
    seed: int = config.SEED,
    block: int | None = None,
    max_days: int = 1000,
) -> MCResult:
    """1フェーズのチャレンジを n_sims 回シミュレートして合格確率などを返す。

    各シム: 日ごとに trades_per_day（ゼロ日含む）からその日のトレード数を引き
    （iid、または block=k で連続kブロックを引いて自己相関を保つ）、各トレードの
    r を r_samples から iid で引く。判定は初期残高 start(=1.0) を基準に、
    - 日次損失違反: equity <= day_start - daily_loss*start（日中・各トレード後も判定）
    - 最大DD違反: equity < start*(1 - max_dd)
    - 合格: equity >= start*(1 + profit_target)
    """
    rs = np.asarray(r_samples, dtype=np.float64)
    tpd = np.asarray(trades_per_day)
    n_len = tpd.shape[0]
    n_r = rs.shape[0]
    if n_len == 0 or n_r == 0:
        raise ValueError("r_samples と trades_per_day は非空である必要があります")

    start = 1.0
    risk_amt = risk_pct * start
    daily_floor_amt = daily_loss * start
    dd_floor = start * (1.0 - max_dd)
    target = start * (1.0 + profit_target)

    rng = np.random.default_rng(seed)
    passes = 0
    breach_daily = 0
    breach_maxdd = 0
    pass_days: list[int] = []

    for _ in range(n_sims):
        equity = start
        outcome = None  # "pass" | "daily" | "maxdd" | None(timeout)
        day_pass = -1
        block_pos = block if block else 0
        block_start = 0

        for d in range(max_days):
            if block is None:
                cnt = int(tpd[rng.integers(0, n_len)])
            else:
                if block_pos >= block:
                    block_start = int(rng.integers(0, n_len))
                    block_pos = 0
                cnt = int(tpd[(block_start + block_pos) % n_len])
                block_pos += 1

            if cnt <= 0:
                continue
            day_start_eq = equity
            draws = rs[rng.integers(0, n_r, size=cnt)]
            for rr in draws:
                equity += risk_amt * rr
                if equity <= day_start_eq - daily_floor_amt:
                    outcome = "daily"
                    break
                if equity < dd_floor:
                    outcome = "maxdd"
                    break
                if equity >= target:
                    outcome = "pass"
                    day_pass = d + 1
                    break
            if outcome is not None:
                break

        if outcome == "pass":
            passes += 1
            pass_days.append(day_pass)
        elif outcome == "daily":
            breach_daily += 1
        elif outcome == "maxdd":
            breach_maxdd += 1

    pass_rate = passes / n_sims if n_sims else 0.0
    if pass_days:
        arr = np.asarray(pass_days, dtype=np.float64)
        median_days = float(np.median(arr))
        p90_days = float(np.percentile(arr, 90))
    else:
        median_days = float("nan")
        p90_days = float("nan")

    return MCResult(
        pass_rate=float(pass_rate),
        fail_rate=float(1.0 - pass_rate),
        median_days_to_pass=median_days,
        p90_days_to_pass=p90_days,
        breach_daily_rate=float(breach_daily / n_sims) if n_sims else 0.0,
        breach_maxdd_rate=float(breach_maxdd / n_sims) if n_sims else 0.0,
    )


def _trades_per_day_from(trades: pd.DataFrame) -> np.ndarray:
    """ts_entry の UTC カレンダー日で、初日〜最終日（両端含む・ゼロ日込み）の日次トレード数。"""
    if len(trades) == 0:
        return np.zeros(0, dtype=np.int64)
    days = pd.to_datetime(trades["ts_entry"], utc=True).dt.floor("D")
    counts = days.value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D", tz="UTC")
    return counts.reindex(full, fill_value=0).to_numpy(dtype=np.int64)


LADDER_COLUMNS: list[str] = [
    "risk_pct", "sampling",
    "p1_pass_rate", "p1_median_days", "p1_p90_days", "p1_breach_daily", "p1_breach_maxdd",
    "p2_pass_rate", "p2_median_days", "p2_p90_days", "p2_breach_daily", "p2_breach_maxdd",
    "pass_endtoend", "expected_attempts", "expected_fee_10k_jpy", "expected_fee_100k_jpy",
]


def run_challenge_ladder(
    trades: pd.DataFrame,
    r_col: str,
    risk_variants=config.PROP_RISK_VARIANTS,
    n_sims: int = config.MC_SIMS,
    seed: int = config.SEED,
    block: int = config.MC_BLOCK,
    max_days: int = 1000,
) -> pd.DataFrame:
    """実トレードから r 分布とトレード数/日を作り、P1/P2 の合格ラダーを算出する。

    各リスク変種 × {iid, block} について P1(target=PROP_PROFIT_TARGET_P1) と
    P2(=PROP_PROFIT_TARGET_P2) を回し、独立近似で通し合格率 = p1*p2、期待受験回数
    = 1/合格率、期待受験料 = 期待回数 × PROP_FEE_JPY を求める。
    """
    r_samples = trades[r_col].to_numpy(dtype=np.float64)
    tpd = _trades_per_day_from(trades)

    rows: list[dict] = []
    k = 0
    for risk in risk_variants:
        for sampling, blk in (("iid", None), ("block", block)):
            s1 = seed + 1000 * k + 1
            s2 = seed + 1000 * k + 2
            p1 = prop_challenge_mc(
                r_samples, tpd, risk, config.PROP_PROFIT_TARGET_P1,
                n_sims=n_sims, seed=s1, block=blk, max_days=max_days)
            p2 = prop_challenge_mc(
                r_samples, tpd, risk, config.PROP_PROFIT_TARGET_P2,
                n_sims=n_sims, seed=s2, block=blk, max_days=max_days)
            e2e = p1.pass_rate * p2.pass_rate
            attempts = (1.0 / e2e) if e2e > 0.0 else float("inf")
            rows.append({
                "risk_pct": risk, "sampling": sampling,
                "p1_pass_rate": p1.pass_rate,
                "p1_median_days": p1.median_days_to_pass,
                "p1_p90_days": p1.p90_days_to_pass,
                "p1_breach_daily": p1.breach_daily_rate,
                "p1_breach_maxdd": p1.breach_maxdd_rate,
                "p2_pass_rate": p2.pass_rate,
                "p2_median_days": p2.median_days_to_pass,
                "p2_p90_days": p2.p90_days_to_pass,
                "p2_breach_daily": p2.breach_daily_rate,
                "p2_breach_maxdd": p2.breach_maxdd_rate,
                "pass_endtoend": e2e,
                "expected_attempts": attempts,
                "expected_fee_10k_jpy": attempts * config.PROP_FEE_JPY["10k"],
                "expected_fee_100k_jpy": attempts * config.PROP_FEE_JPY["100k"],
            })
            k += 1
    return pd.DataFrame(rows, columns=LADDER_COLUMNS)
