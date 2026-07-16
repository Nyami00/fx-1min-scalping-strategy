"""資金管理シミュレーションのユニットテスト（SPEC.md §7）。

fixed / tiered / pct の損益算術、ティアのラチェット昇格（降格なし・2段跳び）、
同時刻クローズ先行、破産フラグ、同時保有数と合計オープンリスク、月次リターンを検証。
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from src.equity import simulate


def _mk(rows) -> pd.DataFrame:
    """(pair, ts_entry, ts_exit, r) タプル列からトレード表を作る。"""
    pair, te, tx, r = zip(*rows)
    return pd.DataFrame({
        "pair": list(pair),
        "ts_entry": pd.to_datetime(list(te), utc=True),
        "ts_exit": pd.to_datetime(list(tx), utc=True),
        "r_net_cons_zero": list(r),
    })


def test_fixed_pnl_arithmetic():
    """fixed: PnL = ¥2,000 × r の単純加算（複利なし）。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-02", 1.5),
        ("EURUSD", "2024-01-03", "2024-01-04", -1.0),
        ("EURUSD", "2024-01-05", "2024-01-06", 0.5),
    ])
    res = simulate(tr, "fixed", "r_net_cons_zero")
    assert res.final_equity == 102000.0        # 100000 + 2000*(1.5-1.0+0.5)
    assert res.ruined is False


def test_tiered_ratchet_and_no_demotion():
    """tiered: 200k で昇格し risk=¥4,000、以後ドローダウンしても降格しない。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-02", 50.0),   # +100000 → 200000 昇格
        ("EURUSD", "2024-01-03", "2024-01-04", -1.0),    # risk 4000 → 196000
        ("EURUSD", "2024-01-05", "2024-01-06", 1.0),     # risk 4000（降格せず）→ 200000
    ])
    res = simulate(tr, "tiered", "r_net_cons_zero")
    eq = res.curve["equity"].tolist()
    assert eq[1] == 200000.0                    # trade1 exit
    assert eq[3] == 196000.0                    # trade2: -4000 → 昇格後 risk=4000 の証拠
    assert eq[5] == 200000.0                    # trade3: +4000 → 降格していない証拠
    assert res.final_equity == 200000.0


def test_tiered_double_promotion():
    """tiered: 1回の決済で 400k を跨ぐと2段昇格し risk=¥8,000 になる。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-02", 175.0),   # +350000 → 450000（100k→200k→400k）
        ("EURUSD", "2024-01-03", "2024-01-04", -1.0),     # risk 8000 → 442000
    ])
    res = simulate(tr, "tiered", "r_net_cons_zero")
    eq = res.curve["equity"].tolist()
    assert eq[1] == 450000.0
    assert eq[3] == 442000.0                    # -8000 → 2段昇格（tier_base=400000）の証拠
    assert res.final_equity == 442000.0


def test_pct_compounding_close_before_open_same_timestamp():
    """pct: 同時刻はクローズ先行。trade2 の risk は trade1 決済後の残高基準。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-05", 1.0),     # +2000 → 102000
        ("USDJPY", "2024-01-05", "2024-01-10", -1.0),    # risk=2%*102000=2040 → 99960
    ])
    res = simulate(tr, "pct", "r_net_cons_zero")
    assert res.final_equity == pytest.approx(99960.0, abs=1e-9)
    assert res.max_concurrent == 1              # クローズ先行のため重ならない


def test_ruin_flag_stops_new_entries():
    """equity<=0 で破産フラグ、以後の新規エントリーは停止（PnL 反映されない）。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-02", -60.0),   # -120000 → -20000（破産）
        ("EURUSD", "2024-01-03", "2024-01-04", 5.0),     # 破産後 → スキップ
    ])
    res = simulate(tr, "fixed", "r_net_cons_zero")
    assert res.ruined is True
    assert res.ruin_ts == pd.Timestamp("2024-01-02", tz="UTC")
    assert res.final_equity == -20000.0         # 2件目は取られない


def test_concurrency_and_open_risk():
    """重なるトレードで最大同時保有数と合計オープンリスクを記録（fixed=各¥2,000）。"""
    tr = _mk([
        ("EURUSD", "2024-01-01", "2024-01-10", 1.0),     # 外側
        ("USDJPY", "2024-01-03", "2024-01-05", 1.0),     # 内側（重なる）
    ])
    res = simulate(tr, "fixed", "r_net_cons_zero")
    assert res.max_concurrent == 2
    assert res.max_open_risk_jpy == 4000.0


def test_two_month_monthly_return():
    """カレンダー月境界の月次リターン（初月は初期資金基準、翌月は前月末基準）。"""
    tr = _mk([
        ("EURUSD", "2024-01-10", "2024-01-20", 2.0),     # 1月: +4000 → 104000
        ("EURUSD", "2024-02-10", "2024-02-20", -1.0),    # 2月: -2000 → 102000
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # tz ドロップ等の警告が出ないこと
        res = simulate(tr, "fixed", "r_net_cons_zero")

    m = res.monthly
    assert len(m) == 2
    assert str(m.index[0]) == "2024-01" and str(m.index[1]) == "2024-02"
    assert m.iloc[0]["start_eq"] == 100000.0 and m.iloc[0]["end_eq"] == 104000.0
    assert m.iloc[0]["ret"] == pytest.approx(0.04)
    assert m.iloc[1]["start_eq"] == 104000.0 and m.iloc[1]["end_eq"] == 102000.0
    assert m.iloc[1]["ret"] == pytest.approx(102000.0 / 104000.0 - 1.0)


def test_empty_trades_returns_start_capital():
    """トレードが無い場合は初期資金のまま・破産なし。"""
    empty = pd.DataFrame({
        "pair": pd.Series(dtype="object"),
        "ts_entry": pd.Series(dtype="datetime64[ns, UTC]"),
        "ts_exit": pd.Series(dtype="datetime64[ns, UTC]"),
        "r_net_cons_zero": pd.Series(dtype="float64"),
    })
    res = simulate(empty, "pct", "r_net_cons_zero")
    assert res.final_equity == 100000.0 and res.ruined is False and res.max_concurrent == 0
