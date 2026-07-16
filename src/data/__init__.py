"""データ層: Dukascopy bi5 の取得/デコード/検証と parquet キャッシュ。"""
from __future__ import annotations

from .base import OHLCV_COLUMNS, QualityReport, validate_bars
from .dukascopy import (
    DownloadStats,
    ProxyBlockedError,
    bi5_local_path,
    bi5_url,
    build_parquet,
    decode_bi5_candles,
    download_range,
    fetch_day,
    iter_days,
    verify_bi5_layout,
)
from .store import get_metadata, load_pair, save_pair

__all__ = [
    "OHLCV_COLUMNS",
    "QualityReport",
    "validate_bars",
    "DownloadStats",
    "ProxyBlockedError",
    "bi5_url",
    "bi5_local_path",
    "iter_days",
    "decode_bi5_candles",
    "verify_bi5_layout",
    "download_range",
    "fetch_day",
    "build_parquet",
    "save_pair",
    "load_pair",
    "get_metadata",
]
