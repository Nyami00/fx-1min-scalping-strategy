"""執行のユニットテスト（SPEC.md §4）。

エグジット走査の優先順位（sl_gap / 曖昧 / sl / tp / eod）、建玉フィルタ、
異常足フィルタ、ロング/ショート鏡像対称、および終値→トレードの一気通貫を検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_indicators
from src.signals import Candidate, detect_candidates
from src.execution import resolve_exit, run_pair


# ---- resolve_exit: 優先順位 -------------------------------------------------

def test_ambiguous_bar_long():
    """同一足で low<=SL かつ high>=TP → 保守=SL / 楽観=TP、t_exit 共有（SPEC §4-2）。"""
    open_ = np.array([100, 100.5, 100], float)
    high = np.array([100, 102.0, 100], float)
    low = np.array([100, 98.5, 100], float)
    close = np.array([100, 100.0, 100], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=1, sl=99.0, tp=101.5)
    assert res.t_exit == 1 and res.ambiguous is True
    assert res.exit_price_cons == 99.0 and res.reason_cons == "sl"
    assert res.exit_price_opt == 101.5 and res.reason_opt == "tp"


def test_ambiguous_bar_short():
    """売りミラー: high>=SL かつ low<=TP → 保守=SL / 楽観=TP（SPEC §4-2）。"""
    open_ = np.array([100, 99.5, 100], float)
    high = np.array([100, 101.5, 100], float)
    low = np.array([100, 98.0, 100], float)
    close = np.array([100, 100.0, 100], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=-1, sl=101.0, tp=98.5)
    assert res.t_exit == 1 and res.ambiguous is True
    assert res.exit_price_cons == 101.0 and res.reason_cons == "sl"
    assert res.exit_price_opt == 98.5 and res.reason_opt == "tp"


def test_sl_gap_fills_at_open_not_ambiguous():
    """寄付が SL を貫通 → 始値約定 sl_gap。TP に触れても曖昧扱いしない（SPEC §4-1）。"""
    open_ = np.array([100, 98.5, 100], float)   # 98.5 <= sl
    high = np.array([100, 102.0, 100], float)   # tp にも到達しているが…
    low = np.array([100, 98.0, 100], float)
    close = np.array([100, 100.0, 100], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=1, sl=99.0, tp=101.5)
    assert res.t_exit == 1 and res.ambiguous is False
    assert res.reason_cons == "sl_gap" and res.reason_opt == "sl_gap"
    assert res.exit_price_cons == 98.5 and res.exit_price_opt == 98.5


def test_favorable_gap_fills_at_exactly_tp():
    """TP を飛び越える有利ギャップでも約定価格は tp ちょうど（SPEC §4-4, §9）。"""
    open_ = np.array([100, 102.0, 100], float)   # tp=101.5 を上抜けて寄り付く
    high = np.array([100, 103.0, 100], float)
    low = np.array([100, 101.8, 100], float)     # 足全体が tp より上（SL 未到達）
    close = np.array([100, 102.5, 100], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=1, sl=99.0, tp=101.5)
    assert res.t_exit == 1 and res.ambiguous is False
    assert res.reason_cons == "tp" and res.reason_opt == "tp"
    assert res.exit_price_cons == 101.5 and res.exit_price_opt == 101.5


def test_entry_bar_extremes_are_ignored():
    """走査は t_entry+1 から。エントリー足自身の高安は無視する（SPEC §4）。"""
    open_ = np.array([100, 100.0, 100], float)
    high = np.array([102, 101.6, 100], float)    # index0 は tp 超えだが無視
    low = np.array([98, 99.5, 100], float)       # index0 は sl 割れだが無視
    close = np.array([100, 100.0, 100], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=1, sl=99.0, tp=101.5)
    assert res.t_exit == 1 and res.reason_cons == "tp" and res.exit_price_cons == 101.5


def test_eod_at_last_close():
    """データ終端まで未タッチなら最終終値で eod 約定（SPEC §4-5）。"""
    open_ = np.array([100, 100, 100, 100], float)
    high = np.array([100.5, 100.5, 100.5, 100.5], float)
    low = np.array([99.5, 99.5, 99.5, 99.5], float)
    close = np.array([100, 100, 100, 100.2], float)
    res = resolve_exit(open_, high, low, close, t_entry=0, direction=1, sl=99.0, tp=101.5)
    assert res.t_exit == 3 and res.ambiguous is False
    assert res.reason_cons == "eod" and res.reason_opt == "eod"
    assert res.exit_price_cons == 100.2 and res.exit_price_opt == 100.2


# ---- run_pair: 建玉/異常足フィルタ -----------------------------------------

def _flat_bars(make_bars, n):
    o = np.full(n, 100.0)
    return make_bars(o, o + 0.2, o - 0.2, o.copy())


def _long(t_entry, t_cross=0, t_break=1, entry=100.0, sl=99.0, tp=101.5):
    return Candidate(1, t_cross, t_break, t_entry, entry, sl, tp)


def test_position_open_skip_then_later_cycle_enters(make_bars):
    """建玉中の候補は position_open でスキップ、t_exit 後の候補は約定（SPEC §4）。"""
    df = _flat_bars(make_bars, 20)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    h[10] = 101.5; c[10] = 101.0     # c1 は bar10 で TP
    h[13] = 101.5; c[13] = 101.0     # c3 は bar13 で TP
    df = make_bars(o, h, l, c)

    c1 = _long(t_entry=5)                       # 5→ exit 10
    c2 = _long(t_entry=8, t_cross=6, t_break=7) # 8 <= open_until(10) → position_open
    c3 = _long(t_entry=12, t_cross=11, t_break=11)  # 12 > 10 → 約定
    trades, skips = run_pair(df, [c1, c2, c3], "USDJPY")

    assert len(trades) == 2 and len(skips) == 1
    assert (trades[0].t_entry, trades[0].t_exit) == (5, 10)
    assert (trades[1].t_entry, trades[1].t_exit) == (12, 13)
    assert skips[0].reason == "position_open" and skips[0].t == 8 and skips[0].direction == 1


def test_invalid_sl_skip_both_directions(make_bars):
    """買い entry<=sl / 売り entry>=sl は invalid_sl でスキップ（SPEC §4）。"""
    df = _flat_bars(make_bars, 10)
    long_bad = Candidate(1, 0, 1, 5, entry=99.9, sl=100.0, tp=110.0)   # entry<=sl
    short_bad = Candidate(-1, 0, 1, 6, entry=100.1, sl=100.0, tp=90.0)  # entry>=sl
    trades, skips = run_pair(df, [long_bad, short_bad], "USDJPY")
    assert trades == []
    assert [s.reason for s in skips] == ["invalid_sl", "invalid_sl"]
    assert [s.direction for s in skips] == [1, -1]


# ---- 一気通貫（合成価格）と鏡像対称 ----------------------------------------

def _e2e_ohlc():
    """クロス→ブレイク→プルバック→TP を含む合成 OHLC（sma400 有効化のため 470 本）。"""
    n = 470
    pat = np.array([100.00, 100.10, 100.00, 99.90])   # 母σ>0 を保つ周期4パターン
    c = np.array([pat[i % 4] for i in range(n)], dtype=float)
    c[404:] += 0.05                     # 水準シフト → index404 で上抜けクロス
    c[430] = 100.05 + 0.40              # ブレイク足（終値>+2σ）
    c[431] = 100.05 + 0.05
    c[432] = 100.05 - 0.20              # プルバック足（終値<mid）→ エントリー
    for t in range(433, n):
        c[t] = c[432] + (t - 432) * 0.06   # TP へ上昇
    o = c.copy()
    h = c + 0.02
    l = c - 0.02
    return o, h, l, c


def _pipe(make_bars, o, h, l, c, pair):
    ind = compute_indicators(c)
    cands = detect_candidates(c, ind)
    df = make_bars(o, h, l, c, start_ts="2024-06-03 00:00")
    return cands, run_pair(df, cands, pair)


def test_end_to_end_exact_trade_row(make_bars):
    """合成価格の一気通貫で、想定どおりの1トレード行を厳密に得る（SPEC §2-4）。"""
    o, h, l, c = _e2e_ohlc()
    cands, (trades, skips) = _pipe(make_bars, o, h, l, c, "USDJPY")
    assert len(cands) == 1 and len(trades) == 1 and skips == []
    tr = trades[0]

    # 構成から一意に定まる各インデックス
    assert tr.direction == 1
    assert (tr.t_cross, tr.t_break, tr.t_entry, tr.t_exit) == (404, 430, 432, 433)
    assert tr.reason_cons == "tp" and tr.reason_opt == "tp" and tr.ambiguous is False

    # 指標を pandas に依らず素朴窓計算で独立再現し、entry/sl/tp を照合
    te = tr.t_entry
    win = c[te - 19 : te + 1]
    exp_mid = win.mean()
    exp_sigma = win.std(ddof=0)
    exp_lower = exp_mid - 2.0 * exp_sigma
    exp_entry = c[te]
    exp_tp = exp_entry + 1.5 * (exp_entry - exp_lower)
    assert tr.entry == pytest.approx(exp_entry, rel=0, abs=1e-12)
    assert tr.sl == pytest.approx(exp_lower, rel=1e-12)
    assert tr.tp == pytest.approx(exp_tp, rel=1e-12)
    assert tr.risk_dist == pytest.approx(exp_entry - exp_lower, rel=1e-12)

    # クリーンな TP なので r_gross は 1.5、保有 1 本 / 1 分
    assert tr.r_gross_cons == pytest.approx(1.5, rel=1e-9)
    assert tr.hold_bars == 1 and tr.hold_minutes == pytest.approx(1.0)

    # シーケンス不変条件（spot_check と同じ独立判定）
    diff = compute_indicators(c).mid - compute_indicators(c).sma400
    assert diff[tr.t_cross - 1] <= 0 < diff[tr.t_cross]
    ind = compute_indicators(c)
    assert c[tr.t_break] > ind.upper[tr.t_break]
    assert c[tr.t_entry] < ind.mid[tr.t_entry]


def test_long_short_mirror_symmetry(make_bars):
    """価格を p'=K-p に鏡映すると、対称なショートが生成され |r| が一致する（SPEC §3）。"""
    o, h, l, c = _e2e_ohlc()
    _, (tr, _) = _pipe(make_bars, o, h, l, c, "USDJPY")

    K = 200.0
    # 鏡映: O'=K-O, C'=K-C, H'=K-L, L'=K-H（高安が入れ替わる）
    om, hm, lm, cm = K - o, K - l, K - h, K - c
    _, (trm, _) = _pipe(make_bars, om, hm, lm, cm, "USDJPY")

    assert len(tr) == 1 and len(trm) == 1
    a, b = tr[0], trm[0]
    assert a.direction == 1 and b.direction == -1
    assert (a.t_entry, a.t_exit) == (b.t_entry, b.t_exit)
    assert a.reason_cons == b.reason_cons
    assert b.entry == pytest.approx(K - a.entry, rel=1e-12)
    assert b.sl == pytest.approx(K - a.sl, rel=1e-12)
    assert abs(a.r_gross_cons) == pytest.approx(abs(b.r_gross_cons), rel=1e-9)
    assert abs(a.r_gross_opt) == pytest.approx(abs(b.r_gross_opt), rel=1e-9)
