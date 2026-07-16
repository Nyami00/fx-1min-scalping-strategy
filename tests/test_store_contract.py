"""store の契約テスト（save/load ラウンドトリップ・ソース混在拒否・契約違反）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data import store  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume"]


def _make_df() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-03", tz="UTC")
    idx = pd.DatetimeIndex(
        base + pd.to_timedelta(np.array([0, 60, 120], dtype="int64"), unit="s"), name="time"
    )
    return pd.DataFrame(
        {
            "open": [1.09, 1.091, 1.092],
            "high": [1.095, 1.096, 1.097],
            "low": [1.085, 1.086, 1.087],
            "close": [1.092, 1.093, 1.094],
            "volume": [10.0, 20.0, 30.0],
        },
        index=idx,
    )[OHLCV]


@pytest.fixture()
def parquet_dir(tmp_path, monkeypatch):
    d = tmp_path / "parquet"
    monkeypatch.setattr(config, "PARQUET_DIR", d)
    return d


def test_save_load_roundtrip_preserves_schema_and_metadata(parquet_dir):
    df = _make_df()
    path = store.save_pair("EURUSD", df, source="dukascopy")
    assert path.exists()
    assert path == parquet_dir / "EURUSD.parquet"

    loaded = store.load_pair("EURUSD")
    pd.testing.assert_frame_equal(loaded, df)
    assert loaded.index.name == "time"
    assert str(loaded.index.tz) == "UTC"
    assert all(loaded[c].dtype == np.float64 for c in OHLCV)

    meta = store.get_metadata("EURUSD")
    assert meta["source"] == "dukascopy"
    assert meta["rows"] == 3
    assert meta["first_ts"].startswith("2024-01-03T00:00:00")
    assert meta["last_ts"].startswith("2024-01-03T00:02:00")
    assert "generated_at" in meta


def test_source_mismatch_refused(parquet_dir):
    df = _make_df()
    store.save_pair("EURUSD", df, source="dukascopy")
    with pytest.raises(ValueError):
        store.save_pair("EURUSD", df, source="histdata")
    # 同一ソースでの上書きは許可
    store.save_pair("EURUSD", df, source="dukascopy")
    assert store.get_metadata("EURUSD")["source"] == "dukascopy"


def test_load_pair_missing_file_raises(parquet_dir):
    with pytest.raises(FileNotFoundError):
        store.load_pair("EURUSD")


def _write_raw(parquet_dir: Path, pair: str, df: pd.DataFrame) -> None:
    """save_pair を経由せず素の parquet を書く（契約違反の作り込み用）。"""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_dir / f"{pair}.parquet")


def test_load_pair_rejects_tz_naive_index(parquet_dir):
    df = _make_df()
    df.index = df.index.tz_localize(None)
    _write_raw(parquet_dir, "AAA1", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA1")


def test_load_pair_rejects_non_monotonic_index(parquet_dir):
    df = _make_df().iloc[[2, 0, 1]]  # 並びが崩れている
    _write_raw(parquet_dir, "AAA2", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA2")


def test_load_pair_rejects_duplicate_index(parquet_dir):
    df = pd.concat([_make_df(), _make_df().iloc[[0]]]).sort_index()
    _write_raw(parquet_dir, "AAA3", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA3")


def test_load_pair_rejects_wrong_dtype(parquet_dir):
    df = _make_df()
    df["open"] = df["open"].astype("float32")
    _write_raw(parquet_dir, "AAA4", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA4")


def test_load_pair_rejects_wrong_columns(parquet_dir):
    df = _make_df().drop(columns=["volume"])
    _write_raw(parquet_dir, "AAA5", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA5")


def test_load_pair_rejects_wrong_index_name(parquet_dir):
    df = _make_df()
    df.index = df.index.rename("timestamp")
    _write_raw(parquet_dir, "AAA6", df)
    with pytest.raises(ValueError):
        store.load_pair("AAA6")
