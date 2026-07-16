"""parquet キャッシュ（ソース非依存の OHLCV 契約）。

`load_pair` は SPEC §1 の契約（UTC tz-aware・狭義単調増加・重複なし・float64）を
検査し、違反時は例外を投げる。ソース混在は parquet メタデータの ``source`` で禁止する。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from .base import OHLCV_COLUMNS

_META_KEYS = ("source", "generated_at", "rows", "first_ts", "last_ts")


def _parquet_path(pair: str) -> Path:
    return config.PARQUET_DIR / f"{pair}.parquet"


def save_pair(pair: str, df: pd.DataFrame, source: str) -> Path:
    """df を zstd 圧縮 parquet として保存し、メタデータを埋め込む。

    既存 parquet の ``source`` と異なる場合は上書きを拒否する（混在禁止）。
    """
    out = _parquet_path(pair)
    if out.exists():
        existing = get_metadata(pair).get("source")
        if existing is not None and existing != source:
            raise ValueError(
                f"ソース不一致のため上書き拒否: 既存={existing!r} != 新規={source!r} "
                f"({out})。混在禁止（SPEC §9）"
            )
    out.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=True)
    n = len(df)
    embed = {
        b"source": source.encode(),
        b"generated_at": datetime.now(timezone.utc).isoformat().encode(),
        b"rows": str(n).encode(),
        b"first_ts": (df.index[0].isoformat() if n else "").encode(),
        b"last_ts": (df.index[-1].isoformat() if n else "").encode(),
    }
    # pandas のインデックス復元メタ（b"pandas"）を保持したままマージする
    merged = {**(table.schema.metadata or {}), **embed}
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, out, compression="zstd")
    return out


def get_metadata(pair: str) -> dict:
    """埋め込みメタデータ（source, generated_at, rows, first_ts, last_ts）を返す。"""
    path = _parquet_path(pair)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に `python scripts/download_data.py` を実行してください。"
        )
    meta = pq.read_schema(path).metadata or {}
    out: dict = {}
    for key in _META_KEYS:
        bkey = key.encode()
        if bkey in meta:
            out[key] = meta[bkey].decode()
    if "rows" in out:
        out["rows"] = int(out["rows"])
    return out


def load_pair(pair: str) -> pd.DataFrame:
    """parquet を読み、SPEC §1 の契約を検査して返す。"""
    path = _parquet_path(pair)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に `python scripts/download_data.py` を実行してください。"
        )
    df = pd.read_parquet(path)
    _validate_contract(df, pair)
    return df


def _validate_contract(df: pd.DataFrame, pair: str) -> None:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{pair}: index が DatetimeIndex ではありません")
    if idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError(f"{pair}: index が UTC tz-aware ではありません（tz={idx.tz}）")
    if idx.name != "time":
        raise ValueError(f"{pair}: index 名が 'time' ではありません（{idx.name!r}）")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{pair}: index が単調増加ではありません")
    if idx.has_duplicates:
        raise ValueError(f"{pair}: index に重複があります")
    if list(df.columns) != OHLCV_COLUMNS:
        raise ValueError(f"{pair}: 列が {OHLCV_COLUMNS} ではありません（{list(df.columns)}）")
    for col in OHLCV_COLUMNS:
        if df[col].dtype != "float64":
            raise ValueError(f"{pair}: 列 {col} が float64 ではありません（{df[col].dtype}）")
