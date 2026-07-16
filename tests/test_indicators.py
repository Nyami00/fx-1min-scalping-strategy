"""指標計算のユニットテスト（SPEC.md §2）。

pandas ローリングによる SMA400・BB を素朴な numpy 窓計算と突き合わせ、
母標準偏差（ddof=0）が既定であること（pandas 既定 ddof=1 の罠）を検証する。
"""
from __future__ import annotations

import numpy as np

from src import config
from src.indicators import compute_indicators


def _series() -> np.ndarray:
    """再現可能な 25 本の終値系列（定数でない値を持たせる）。"""
    rng = np.random.default_rng(config.SEED)
    return 100.0 + np.cumsum(rng.normal(0.0, 0.05, size=25))


def test_indicators_match_hand_computed_25bar():
    """mid / sigma / upper / lower / sma400 を素朴窓計算と一致させる。"""
    close = _series()
    bb, dev, sma_p = 20, 2.0, 25
    ind = compute_indicators(close, bb_period=bb, bb_dev=dev, sma_period=sma_p, ddof=0)

    # BB 有効域（index 19 以降）を手計算と突き合わせる
    for t in range(bb - 1, close.size):
        win = close[t - bb + 1 : t + 1]
        exp_mid = win.mean()
        exp_sig = win.std(ddof=0)  # 母標準偏差
        np.testing.assert_allclose(ind.mid[t], exp_mid, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(ind.sigma[t], exp_sig, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(ind.upper[t], exp_mid + dev * exp_sig, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(ind.lower[t], exp_mid - dev * exp_sig, rtol=1e-10, atol=0.0)

    # sma400（ここでは sma_period=25）は最終足のみ有効
    assert np.isnan(ind.sma400[sma_p - 2])
    np.testing.assert_allclose(ind.sma400[sma_p - 1], close.mean(), rtol=1e-12, atol=0.0)


def test_all_arrays_float64():
    """全指標配列が float64 であること（float32 禁止・SPEC §9）。"""
    ind = compute_indicators(_series(), bb_period=20, sma_period=25)
    for arr in (ind.sma400, ind.mid, ind.sigma, ind.upper, ind.lower):
        assert arr.dtype == np.float64


def test_default_ddof_is_population_and_differs_from_ddof1():
    """既定 sigma は母標準偏差（ddof=0）。ddof=1 とは検出可能に異なる。"""
    close = _series()
    ind_default = compute_indicators(close, bb_period=20, sma_period=25)  # 既定 ddof
    ind0 = compute_indicators(close, bb_period=20, sma_period=25, ddof=0)
    ind1 = compute_indicators(close, bb_period=20, sma_period=25, ddof=1)

    # 既定は ddof=0（母標準偏差）と一致する
    np.testing.assert_allclose(ind_default.sigma, ind0.sigma, rtol=0, atol=0, equal_nan=True)

    # ddof=0 と ddof=1 は同一ではない（配線ミス検出）
    valid = ~np.isnan(ind0.sigma)
    assert valid.any()
    assert not np.allclose(ind0.sigma[valid], ind1.sigma[valid])

    # 比は sqrt(N/(N-1))（母→標本）に一致する
    ratio = ind1.sigma[valid] / ind0.sigma[valid]
    np.testing.assert_allclose(ratio, np.sqrt(20.0 / 19.0), rtol=1e-9, atol=0.0)

    # 既定は母標準偏差なので、素朴な母標準偏差と一致し標本標準偏差とは一致しない
    t = close.size - 1
    win = close[t - 19 : t + 1]
    np.testing.assert_allclose(ind_default.sigma[t], win.std(ddof=0), rtol=1e-10)
    assert not np.isclose(ind_default.sigma[t], win.std(ddof=1))


def test_warmup_nan_boundaries():
    """SMA400 は index 399 以降、mid は index 19 以降で有効（それ以前は NaN）。"""
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.02, size=405))
    ind = compute_indicators(close)  # 既定 sma_period=400, bb_period=20
    assert np.all(np.isnan(ind.sma400[:399]))
    assert not np.isnan(ind.sma400[399])
    assert np.all(np.isnan(ind.mid[:19]))
    assert not np.isnan(ind.mid[19])
