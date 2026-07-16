"""シグナル検出のユニットテスト（SPEC.md §3）。

サイクル窓とシーケンス（ブレイク→プルバック）の判定を、手で組んだ Indicators を
注入して検証する。指標そのものの正しさは test_indicators に分離してある。
"""
from __future__ import annotations

import numpy as np

from src import config
from src.signals import Candidate, detect_candidates, _cross_indices


def test_warmup_produces_no_candidates_in_nan_region(make_bars):
    """NaN ウォームアップ域ではクロスもブレイクも成立しない（SPEC §2, §3）。"""
    # 系列が sma_period 未満なら sma400 は全 NaN → 候補ゼロ
    short = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.02, size=100))
    from src.indicators import compute_indicators
    assert detect_candidates(short, compute_indicators(short)) == []

    # 十分長い系列でも、クロスは有効域（index>=400）以降にしか現れない
    long = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.03, size=520))
    ind = compute_indicators(long)
    cross_idx, _ = _cross_indices(ind.mid - ind.sma400)
    assert np.all(cross_idx >= config.SMA_PERIOD)
    for c in detect_candidates(long, ind):
        assert c.t_cross >= config.SMA_PERIOD
        assert c.t_break >= config.SMA_PERIOD
        assert c.t_entry >= config.SMA_PERIOD


def test_cross_bar_itself_counts_as_break(make_ind):
    """クロス足自身が終値>+2σ なら、その足を t_break とする（SPEC §3-1）。"""
    sma = [100, 100, 100, 100, 100, 100]
    mid = [99, 101, 101, 101, 101, 101]          # index1 で上抜けクロス
    upper = [102, 100.5, 102, 102, 102, 102]      # index1 の upper を低くしておく
    lower = [98, 98, 98, 98, 98, 98]
    close = np.array([100, 101, 100, 100, 100, 100], dtype=float)  # close[1]>upper[1]
    cands = detect_candidates(close, make_ind(sma, mid, upper, lower))
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == 1 and c.t_cross == 1 and c.t_break == 1 and c.t_entry == 2


def test_pullback_before_break_is_ignored(make_ind):
    """t_break 以前の 終値<mid はエントリーにしない（SPEC §3-2）。"""
    sma = [100] * 7
    mid = [99, 101, 101, 101, 101, 101, 101]      # index1 で上抜け
    upper = [102] * 7
    lower = [98] * 7
    # index1,2 は close<mid だがブレイク前 / ブレイクは index3 / エントリーは index4
    close = np.array([100, 100.5, 100, 103, 100, 100, 100], dtype=float)
    cands = detect_candidates(close, make_ind(sma, mid, upper, lower))
    assert len(cands) == 1
    c = cands[0]
    assert c.t_break == 3 and c.t_entry == 4          # 4 であって 1/2 ではない


def test_opposite_cross_cancels_pending_long(make_ind):
    """反対クロスが未完成の買いサイクルを取消し、売りサイクルが進む（SPEC §3）。"""
    sma = [100] * 8
    #      0   1    2    3    4   5   6   7
    mid = [99, 101, 101, 101, 99, 99, 99, 99]         # up@1, down@4
    upper = [102] * 8
    lower = [98] * 8
    # 買い窓[1,4): ブレイク@2 だが [3,4) にプルバックなし → 取消
    # 売り窓[4,8): ブレイク@5(<-2σ), エントリー@6(>mid)
    close = np.array([100, 100.5, 103, 101.5, 99, 97, 100, 99], dtype=float)
    cands = detect_candidates(close, make_ind(sma, mid, upper, lower))
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == -1 and c.t_cross == 4 and c.t_break == 5 and c.t_entry == 6


def test_zero_touch_restarts_cycle(make_ind):
    """diff が +,0,+ で同方向クロスが2回起き、2回目でサイクルが再始動（SPEC §3）。"""
    sma = [100] * 8
    diff = [-1, 1, 1, 1, 0, 1, 1, 1]                  # up@1, ちょうど0タッチ@4, up@5
    mid = [s + d for s, d in zip(sma, diff)]
    upper = [102] * 8
    lower = [98] * 8
    # 窓1[1,5): ブレイク@2 だがエントリーなし / 窓2[5,8): ブレイク@6, エントリー@7
    close = np.array([100, 100.5, 103, 101.5, 100.5, 100.8, 103, 100], dtype=float)

    cross_idx, cross_dir = _cross_indices(np.asarray(mid, float) - np.asarray(sma, float))
    assert list(cross_idx) == [1, 5] and list(cross_dir) == [1, 1]   # 同方向2クロス

    cands = detect_candidates(close, make_ind(sma, mid, upper, lower))
    assert len(cands) == 1
    c = cands[0]
    assert c.t_cross == 5 and c.t_break == 6 and c.t_entry == 7        # 再始動側のみ


def test_only_first_sequence_per_window(make_ind):
    """1サイクル窓では最初の完成シーケンスのみ候補化（SPEC §3-3）。"""
    sma = [100] * 8
    mid = [99, 101, 101, 101, 101, 101, 101, 101]     # up@1 のみ（以後クロスなし）
    upper = [102] * 8
    lower = [98] * 8
    # 完成シーケンスが2組: break@2→entry@3, break@5→entry@6。採用は1組目のみ。
    close = np.array([100, 100.5, 103, 100, 101.5, 103, 100, 101.5], dtype=float)
    cands = detect_candidates(close, make_ind(sma, mid, upper, lower))
    assert len(cands) == 1
    c = cands[0]
    assert c.t_break == 2 and c.t_entry == 3


def test_rr_arithmetic_long(make_ind):
    """買いの entry/sl/tp 算術（RR1:1.5, SPEC §3）。"""
    sma = [100] * 6
    mid = [99, 101, 101, 101, 101, 101]               # up@1
    upper = [102] * 6
    lower = [98, 98, 99, 99, 99, 99]                  # lower[3]=99
    close = np.array([100, 100, 103, 100, 100, 100], dtype=float)  # break@2, entry@3
    c = detect_candidates(close, make_ind(sma, mid, upper, lower), rr=1.5)[0]
    assert c.direction == 1 and c.t_entry == 3
    assert c.entry == 100.0
    assert c.sl == 99.0
    assert c.tp == 101.5                               # 100 + 1.5*(100-99)


def test_rr_arithmetic_short(make_ind):
    """売りの entry/sl/tp 算術（RR1:1.5, SPEC §3）。"""
    sma = [100] * 6
    mid = [101, 99, 99, 99, 99, 99]                   # down@1
    upper = [102, 102, 101, 101, 101, 101]            # upper[3]=101
    lower = [98] * 6
    close = np.array([100, 100, 97, 100, 100, 100], dtype=float)   # break@2, entry@3
    c = detect_candidates(close, make_ind(sma, mid, upper, lower), rr=1.5)[0]
    assert c.direction == -1 and c.t_entry == 3
    assert c.entry == 100.0
    assert c.sl == 101.0
    assert c.tp == 98.5                                # 100 - 1.5*(101-100)
