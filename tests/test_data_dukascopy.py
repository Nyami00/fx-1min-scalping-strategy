"""データ層（dukascopy / base）のオフラインテスト。ネットワーク不要。"""
from __future__ import annotations

import lzma
import struct
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data import base, store  # noqa: E402
from src.data.dukascopy import (  # noqa: E402
    bi5_local_path,
    bi5_url,
    build_parquet,
    decode_bi5_candles,
    verify_bi5_layout,
)

_REC = struct.Struct(">IIIIIf")
OHLCV = ["open", "high", "low", "close", "volume"]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "EURUSD_20240103_BID_candles_min_1.bi5"


def _pack(records) -> bytes:
    """records: [(sec, p1, p2, p3, p4, vol), ...] を FORMAT_ALONE 圧縮 bi5 にする。"""
    blob = b"".join(_REC.pack(*r) for r in records)
    return lzma.compress(blob, format=lzma.FORMAT_ALONE)


def _index(day: date, secs) -> pd.DatetimeIndex:
    base = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz="UTC")
    return pd.DatetimeIndex(base + pd.to_timedelta(np.asarray(secs, dtype="int64"), unit="s"), name="time")


# --------------------------------------------------------------------------- #
# URL / パス（月は 0 起点）
# --------------------------------------------------------------------------- #
def test_bi5_url_month_zero_indexed_january():
    url = bi5_url("EURUSD", date(2024, 1, 15))
    assert url.endswith("/EURUSD/2024/00/15/BID_candles_min_1.bi5")
    assert url == "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/15/BID_candles_min_1.bi5"


def test_bi5_url_december_is_month_11():
    url = bi5_url("USDJPY", date(2024, 12, 31))
    assert url.endswith("/USDJPY/2024/11/31/BID_candles_min_1.bi5")


def test_bi5_local_path_mirrors_url():
    d = date(2024, 12, 5)
    path = bi5_local_path("USDJPY", d)
    assert path == config.RAW_DIR / "USDJPY" / "2024" / "11" / "05" / "BID_candles_min_1.bi5"
    tail = bi5_url("USDJPY", d).split("/datafeed/", 1)[1]
    assert str(path).endswith(tail)


# --------------------------------------------------------------------------- #
# デコード合成ラウンドトリップ（JPY・非JPY × 両フィールド順）
# --------------------------------------------------------------------------- #
# 同一の整数コードを両ペアで使う（除数分岐のみが違う）。roundtrip は厳密一致を要求。
_CODES = [
    #  sec,  p1,     p2,     p3,     p4,     vol
    (0, 109500, 109600, 109400, 109700, 10.0),
    (60, 109550, 109650, 109450, 109750, 20.0),
    (120, 109600, 109700, 109500, 109800, 30.0),
]


@pytest.mark.parametrize("pair,divisor", [("EURUSD", 1e5), ("USDJPY", 1e3)])
@pytest.mark.parametrize("field_order", ["OCLH", "OHLC"])
def test_decode_roundtrip_exact(pair, divisor, field_order):
    day = date(2024, 1, 3)
    raw = _pack(_CODES)
    got = decode_bi5_candles(raw, pair, day, field_order=field_order)

    p1 = np.array([r[1] for r in _CODES], dtype="float64") / divisor
    p2 = np.array([r[2] for r in _CODES], dtype="float64") / divisor
    p3 = np.array([r[3] for r in _CODES], dtype="float64") / divisor
    p4 = np.array([r[4] for r in _CODES], dtype="float64") / divisor
    vol = np.array([r[5] for r in _CODES], dtype="float64")
    idx = _index(day, [r[0] for r in _CODES])

    if field_order == "OCLH":  # p1=open, p2=close, p3=low, p4=high
        expected = pd.DataFrame(
            {"open": p1, "high": p4, "low": p3, "close": p2, "volume": vol}, index=idx
        )
    else:  # p1=open, p2=high, p3=low, p4=close
        expected = pd.DataFrame(
            {"open": p1, "high": p2, "low": p3, "close": p4, "volume": vol}, index=idx
        )
    expected = expected[OHLCV]
    assert_frame_equal(got, expected)
    assert got.index.tz is not None and str(got.index.tz) == "UTC"
    assert got.index.name == "time"
    assert all(got[c].dtype == np.float64 for c in OHLCV)


def test_decode_default_field_order_is_oclh():
    raw = _pack(_CODES)
    a = decode_bi5_candles(raw, "EURUSD", date(2024, 1, 3))
    b = decode_bi5_candles(raw, "EURUSD", date(2024, 1, 3), field_order="OCLH")
    assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# ゼロバイト → 空 DF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [b"", b"\x00\x00\x00\x00"])
def test_decode_zero_bytes_empty(raw):
    df = decode_bi5_candles(raw, "EURUSD", date(2024, 1, 6))
    assert df.empty
    assert list(df.columns) == OHLCV
    assert df.index.name == "time"
    assert str(df.index.tz) == "UTC"
    assert all(df[c].dtype == np.float64 for c in OHLCV)


# --------------------------------------------------------------------------- #
# レイアウト経験的検証
# --------------------------------------------------------------------------- #
def test_verify_layout_winner_oclh():
    # OCLH で妥当（low=1.094, o=1.095, c=1.096, high=1.097）／OHLC では違反
    codes = [(s, 109500, 109600, 109400, 109700, 10.0) for s in (0, 60, 120, 180)]
    res = verify_bi5_layout(_pack(codes), "EURUSD", date(2024, 1, 3))
    assert res["winner"] == "OCLH"
    assert res["OCLH"]["ohlc_violations"] == 0
    assert res["OCLH"]["out_of_range"] == 0
    assert res["OHLC"]["ohlc_violations"] > 0
    assert res["offsets_ok"] is True


def test_verify_layout_winner_ohlc():
    # OHLC で妥当（o=1.095, high=1.097, low=1.094, c=1.096）／OCLH では違反
    codes = [(s, 109500, 109700, 109400, 109600, 10.0) for s in (0, 60, 120, 180)]
    res = verify_bi5_layout(_pack(codes), "EURUSD", date(2024, 1, 3))
    assert res["winner"] == "OHLC"
    assert res["OHLC"]["ohlc_violations"] == 0
    assert res["OCLH"]["ohlc_violations"] > 0


def test_verify_layout_indeterminate_raises():
    # 全価格が等しいと両仮説とも違反ゼロ → 判別不能で例外
    codes = [(s, 109500, 109500, 109500, 109500, 10.0) for s in (0, 60, 120)]
    with pytest.raises(ValueError):
        verify_bi5_layout(_pack(codes), "EURUSD", date(2024, 1, 3))


def test_verify_layout_bad_offsets_raises():
    # 秒offset が 60 の倍数でない → offsets_ok=False で両仮説とも不合格 → 例外
    codes = [(s, 109500, 109600, 109400, 109700, 10.0) for s in (0, 61, 130)]
    with pytest.raises(ValueError):
        verify_bi5_layout(_pack(codes), "EURUSD", date(2024, 1, 3))


def test_verify_layout_empty_raises():
    with pytest.raises(ValueError):
        verify_bi5_layout(b"", "EURUSD", date(2024, 1, 6))


# --------------------------------------------------------------------------- #
# validate_bars: 違反の数え上げ
# --------------------------------------------------------------------------- #
def test_validate_bars_counts_planted_issues():
    base_ts = pd.Timestamp("2024-01-03", tz="UTC")
    idx = pd.DatetimeIndex(
        [
            base_ts,
            base_ts + pd.Timedelta(minutes=1),
            base_ts + pd.Timedelta(minutes=2),
            base_ts + pd.Timedelta(minutes=2),  # 重複
        ],
        name="time",
    )
    df = pd.DataFrame(
        {
            # row0: OHLC 違反（high < max(open,close)）だがレンジ内
            # row1: レンジ外（全て 5.0）だが OHLC 整合
            # row2/row3: 正常（row3 は row2 の重複タイムスタンプ）
            "open": [1.10, 5.0, 1.09, 1.09],
            "high": [1.09, 5.0, 1.10, 1.10],
            "low": [1.08, 5.0, 1.08, 1.08],
            "close": [1.105, 5.0, 1.095, 1.095],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )
    snapshot = df.copy(deep=True)
    rep = base.validate_bars(df, "EURUSD")

    assert rep.n_ohlc_violations == 1
    assert rep.n_out_of_range == 1
    assert rep.n_dupes_dropped == 1
    assert rep.rows == 4
    assert rep.pair == "EURUSD"
    # df を変更していないこと
    assert_frame_equal(df, snapshot)


def test_validate_bars_clean_gap_stats():
    base_ts = pd.Timestamp("2024-01-03", tz="UTC")
    # 0,1,2 分 … その後 5 分に飛ぶ（3 分ギャップ）
    idx = pd.DatetimeIndex(
        [base_ts + pd.Timedelta(minutes=m) for m in (0, 1, 2, 5)], name="time"
    )
    df = pd.DataFrame(
        {c: [1.10, 1.10, 1.10, 1.10] for c in ["open", "high", "low", "close"]}
        | {"volume": [1.0, 1.0, 1.0, 1.0]},
        index=idx,
    )
    rep = base.validate_bars(df, "EURUSD")
    assert rep.n_gaps_over_1min == 1
    assert rep.max_gap_minutes == 3.0
    assert rep.n_ohlc_violations == 0
    d = rep.as_dict()
    assert d["first_ts"].startswith("2024-01-03T00:00:00")


# --------------------------------------------------------------------------- #
# build_parquet: 閉場時間のフラット・フィラーバー除去
# --------------------------------------------------------------------------- #
def test_build_parquet_drops_flat_filler_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path / "parquet")

    day = date(2024, 1, 3)
    base_ts = pd.Timestamp("2024-01-03", tz="UTC")
    # OCLH コード: (sec, p1=open, p2=close, p3=low, p4=high, vol)
    codes = [
        (0, 109500, 109520, 109480, 109550, 5.0),     # 実バー → 保持
        (60, 109500, 109500, 109500, 109500, 0.0),    # フラット vol0 → フィラー除去
        (120, 109500, 109500, 109500, 109500, 0.0),   # フラット vol0 → フィラー除去
        (180, 109500, 109500, 109500, 109500, 0.0),   # フラット vol0 → フィラー除去
        (240, 109500, 109500, 109490, 109510, 0.0),   # vol0 だが high!=low → 保持
        (300, 109500, 109500, 109500, 109500, 3.0),   # フラットだが vol>0 → 保持
        (360, 109510, 109530, 109490, 109560, 7.0),   # 実バー → 保持
    ]
    path = bi5_local_path("EURUSD", day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pack(codes))

    report = build_parquet("EURUSD", day, day, field_order="OCLH")

    # フラット・フィラー 3 本のみ除去、残り 4 本
    assert report.n_filler_dropped == 3
    assert report.as_dict()["n_filler_dropped"] == 3
    assert report.rows == 4
    assert report.n_dupes_dropped == 0

    df = store.load_pair("EURUSD")
    assert len(df) == 4
    # 残存フレームにフィラー（vol0 かつ high==low）は無い
    assert not ((df["volume"] == 0) & (df["high"] == df["low"])).any()
    # 保持されるべき端ケースが存在する
    assert (base_ts + pd.Timedelta(seconds=240)) in df.index  # vol0 high!=low
    assert (base_ts + pd.Timedelta(seconds=300)) in df.index  # flat vol>0
    # 実バーは残る
    assert (base_ts + pd.Timedelta(seconds=0)) in df.index
    assert (base_ts + pd.Timedelta(seconds=360)) in df.index
    # 除去されたフィラー分足は不在
    for sec in (60, 120, 180):
        assert (base_ts + pd.Timedelta(seconds=sec)) not in df.index


# --------------------------------------------------------------------------- #
# 実データ fixture（存在時のみ）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FIXTURE.exists(), reason="実データ fixture 無し（ネットワーク未許可）")
def test_real_fixture_layout_and_prices():
    raw = FIXTURE.read_bytes()
    day = date(2024, 1, 3)
    res = verify_bi5_layout(raw, "EURUSD", day)
    assert res["winner"] in ("OCLH", "OHLC")

    df = decode_bi5_candles(raw, "EURUSD", day, field_order=res["winner"])
    assert 1200 <= len(df) <= 1500  # 平日は ~1440 本
    prices = df[["open", "high", "low", "close"]].to_numpy()
    assert (prices > 1.0).all() and (prices < 1.2).all()  # 2024年初 EURUSD ≈ 1.09
    o, h, low, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    assert ((low <= np.minimum(o, c)) & (np.maximum(o, c) <= h)).all()
