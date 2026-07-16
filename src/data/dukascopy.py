"""Dukascopy BID 1分足（bi5）のダウンロード・デコード・レイアウト検証・parquet 構築。

URL の月は 0 起点（2024年1月 → ``/2024/00/``）。この 0 起点変換は `bi5_url` と
`bi5_local_path` の 2 箇所だけに隔離する。bi5 のフィールド順（OCLH/OHLC）と除数
（JPYクオート=1e3 / 他=1e5）は実データの経験的検証（`verify_bi5_layout`）で確定する。
"""
from __future__ import annotations

import lzma
import struct
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .. import config
from . import store
from .base import OHLCV_COLUMNS, QualityReport, validate_bars

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
BI5_NAME = "BID_candles_min_1.bi5"

# 24 バイト・ビッグエンディアン: [秒offset, p1, p2, p3, p4, volume(float32)]
_RECORD = struct.Struct(">IIIIIf")
_RECORD_DTYPE = np.dtype(
    [("sec", ">u4"), ("p1", ">u4"), ("p2", ">u4"), ("p3", ">u4"), ("p4", ">u4"), ("vol", ">f4")]
)
_RECORD_SIZE = _RECORD.size  # 24

_SECONDS_PER_DAY = 86400


# --------------------------------------------------------------------------- #
# URL / パス（0 起点の月変換はここだけ）
# --------------------------------------------------------------------------- #
def bi5_url(pair: str, d: date) -> str:
    """bi5 の URL。月は 0 起点（1月→00, 12月→11）。"""
    return f"{BASE_URL}/{pair}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/{BI5_NAME}"


def bi5_local_path(pair: str, d: date) -> Path:
    """URL パスを `config.RAW_DIR` 配下にミラーしたローカルパス。月は 0 起点。"""
    return config.RAW_DIR / pair / f"{d.year:04d}" / f"{d.month - 1:02d}" / f"{d.day:02d}" / BI5_NAME


def iter_days(start: date, end: date) -> Iterator[date]:
    """start〜end（両端含む）のカレンダー日を列挙する。"""
    d = start
    one = timedelta(days=1)
    while d <= end:
        yield d
        d += one


# --------------------------------------------------------------------------- #
# デコード
# --------------------------------------------------------------------------- #
def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=idx)


def _decompress(raw: bytes) -> bytes:
    """既定フォーマットで解凍し、LZMAError なら FORMAT_ALONE で再試行する。"""
    try:
        return lzma.decompress(raw)
    except lzma.LZMAError:
        # なお失敗すれば LZMAError を呼び出し側へ送出する
        return lzma.decompress(raw, format=lzma.FORMAT_ALONE)


def _parse_records(data: bytes):
    """解凍済みバイト列を構造化配列に読み、(sec, p1..p4, vol) の float64 配列を返す。"""
    n = len(data) // _RECORD_SIZE
    arr = np.frombuffer(data, dtype=_RECORD_DTYPE, count=n)
    sec = arr["sec"].astype(np.int64)
    return sec, arr


def decode_bi5_candles(
    raw: bytes, pair: str, day: date, field_order: str = "OCLH"
) -> pd.DataFrame:
    """bi5 生バイト列を OHLCV DataFrame へデコードする。

    - 空/全ゼロバイト → 正しいスキーマの空 DF
    - 価格 = 整数 / 1e3（JPYクオート）または / 1e5（他）
    - index = day 00:00 UTC + 秒offset、名前 "time"、全列 float64
    """
    if field_order not in ("OCLH", "OHLC"):
        raise ValueError(f"未知の field_order: {field_order!r}")
    if not raw or not raw.strip(b"\x00"):
        return _empty_frame()

    data = _decompress(raw)
    n = len(data) // _RECORD_SIZE
    if n == 0:
        return _empty_frame()

    sec, arr = _parse_records(data)
    divisor = 1e3 if pair.endswith("JPY") else 1e5
    p1 = arr["p1"].astype(np.float64) / divisor
    p2 = arr["p2"].astype(np.float64) / divisor
    p3 = arr["p3"].astype(np.float64) / divisor
    p4 = arr["p4"].astype(np.float64) / divisor
    vol = arr["vol"].astype(np.float64)

    if field_order == "OCLH":  # H1: p1=open, p2=close, p3=low, p4=high
        open_, close_, low_, high_ = p1, p2, p3, p4
    else:  # OHLC / H2: p1=open, p2=high, p3=low, p4=close
        open_, high_, low_, close_ = p1, p2, p3, p4

    base = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz="UTC")
    idx = pd.DatetimeIndex(base + pd.to_timedelta(sec, unit="s"), name="time")
    df = pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close_, "volume": vol},
        index=idx,
    )
    return df[OHLCV_COLUMNS]


# --------------------------------------------------------------------------- #
# レイアウト経験的検証
# --------------------------------------------------------------------------- #
def verify_bi5_layout(raw: bytes, pair: str, day: date) -> dict:
    """両フィールド順でデコードし、OHLC不変条件・妥当レンジ・秒offsetを検査する。

    勝者 = OHLC違反0・レンジ内・秒offset妥当を満たす唯一の仮説。両方/どちらも
    満たさない場合、または敗者に OHLC 違反が無い（判別が不確実）場合は例外を投げる。
    """
    if not raw or not raw.strip(b"\x00"):
        raise ValueError("空データではレイアウト判別不可")

    data = _decompress(raw)
    n = len(data) // _RECORD_SIZE
    if n == 0:
        raise ValueError("レコードが無くレイアウト判別不可")

    sec, _ = _parse_records(data)
    offsets_increasing = bool(np.all(np.diff(sec) > 0)) if n >= 2 else True
    offsets_multiple_of_60 = bool(np.all(sec % 60 == 0))
    offsets_in_day = bool(np.all((sec >= 0) & (sec < _SECONDS_PER_DAY)))
    offsets_ok = offsets_increasing and offsets_multiple_of_60 and offsets_in_day

    lo, hi = config.PLAUSIBLE_RANGE[pair]
    result: dict = {
        "pair": pair,
        "day": day.isoformat(),
        "rows": int(n),
        "offsets_increasing": offsets_increasing,
        "offsets_multiple_of_60": offsets_multiple_of_60,
        "offsets_in_day": offsets_in_day,
        "offsets_ok": offsets_ok,
    }

    for order in ("OCLH", "OHLC"):
        df = decode_bi5_candles(raw, pair, day, field_order=order)
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        low = df["low"].to_numpy()
        c = df["close"].to_numpy()
        ok = (low <= np.minimum(o, c)) & (np.maximum(o, c) <= h)
        ohlc_violations = int((~ok).sum())
        prices = np.column_stack([o, h, low, c])
        out_of_range = int(((prices < lo) | (prices > hi)).sum())
        result[order] = {
            "ohlc_violations": ohlc_violations,
            "out_of_range": out_of_range,
            "passed": bool(ohlc_violations == 0 and out_of_range == 0 and offsets_ok),
        }

    winners = [o for o in ("OCLH", "OHLC") if result[o]["passed"]]
    if len(winners) != 1:
        raise ValueError(
            "レイアウト判別不能: "
            f"OCLH={result['OCLH']}, OHLC={result['OHLC']}, offsets_ok={offsets_ok}"
        )
    winner = winners[0]
    loser = "OHLC" if winner == "OCLH" else "OCLH"
    if result[loser]["ohlc_violations"] <= 0:
        raise ValueError(
            f"敗者仮説 {loser} が OHLC 不変条件違反を示さず判別が不確実: {result}"
        )
    result["winner"] = winner
    return result


# --------------------------------------------------------------------------- #
# ダウンロード
# --------------------------------------------------------------------------- #
@dataclass
class DownloadStats:
    """ダウンロード結果の集計。"""

    downloaded: int = 0
    skipped_existing: int = 0
    empty_days: int = 0
    failed: list = field(default_factory=list)


class ProxyBlockedError(RuntimeError):
    """Dukascopy ドメインがプロキシ egress ポリシーで未許可（403 CONNECT）。全DLを即中止。"""


_thread_local = threading.local()


def _build_session() -> requests.Session:
    """Retry 付き Session を生成する。verify は上書きしない（環境変数の CA を尊重）。"""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = _build_session()
        _thread_local.session = s
    return s


def _write_marker(path: Path) -> None:
    """「データなし」を表すゼロバイトのマーカーを書く（レジューム用）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def _fetch(session: requests.Session, pair: str, d: date, abort: threading.Event) -> str:
    """1 日分を取得。戻り値は "downloaded" / "skipped" / "empty"。"""
    if abort.is_set():
        raise ProxyBlockedError("プロキシ拒否により中止済み")
    path = bi5_local_path(pair, d)
    if path.exists():  # レジューム: サイズ 0 でも存在すれば skip
        return "skipped"
    url = bi5_url(pair, d)
    try:
        resp = session.get(url, timeout=30)
    except requests.exceptions.ProxyError as exc:
        abort.set()
        raise ProxyBlockedError(str(exc)) from exc

    status = resp.status_code
    if status in (403, 407):  # プロキシによる CONNECT 拒否
        abort.set()
        raise ProxyBlockedError(f"HTTP {status}（proxy CONNECT denied）: {url}")
    if status == 404:
        _write_marker(path)
        return "empty"
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {url}")

    content = resp.content
    if len(content) == 0:  # 200 だが 0 バイト → データなし
        _write_marker(path)
        return "empty"
    _atomic_write(path, content)
    return "downloaded"


def _fetch_threadlocal(pair: str, d: date, abort: threading.Event) -> str:
    return _fetch(_session(), pair, d, abort)


def download_range(pairs, start: date, end: date, workers: int = 8) -> DownloadStats:
    """pairs × [start, end]（両端含む）を並列取得する。レジューム可能・冪等。

    404/0バイトはゼロバイトマーカーで記録。ProxyError/403 CONNECT は全取得を即中止。
    """
    stats = DownloadStats()
    abort = threading.Event()
    tasks = [(p, d) for p in pairs for d in iter_days(start, end)]
    if not tasks:
        return stats

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_threadlocal, p, d, abort): (p, d) for p, d in tasks}
        try:
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="dukascopy DL", unit="file"
            ):
                pair, d = futures[fut]
                try:
                    kind = fut.result()
                except ProxyBlockedError:
                    raise
                except Exception as exc:  # noqa: BLE001 — 個別失敗は集計して継続
                    stats.failed.append((pair, d.isoformat(), str(exc)))
                    continue
                if kind == "downloaded":
                    stats.downloaded += 1
                elif kind == "skipped":
                    stats.skipped_existing += 1
                elif kind == "empty":
                    stats.empty_days += 1
        except ProxyBlockedError as exc:
            for f in futures:
                f.cancel()
            raise ProxyBlockedError(
                "Dukascopy ドメインがプロキシ egress ポリシーで未許可です（403 CONNECT）。"
                "ネットワーク許可後に再実行してください。ダウンロードを中止しました。"
            ) from exc
    return stats


def fetch_day(pair: str, d: date) -> bytes:
    """1 日分のローカル bi5 を返す（無ければ単発取得）。CLI の --verify-layout 用。"""
    path = bi5_local_path(pair, d)
    if not path.exists():
        _download_single(pair, d)
    return path.read_bytes() if path.exists() else b""


def _download_single(pair: str, d: date) -> None:
    """破損再取得・単発取得に使う同期ダウンロード（1 日 1 ファイル）。"""
    session = _build_session()
    url = bi5_url(pair, d)
    try:
        resp = session.get(url, timeout=30)
    except requests.exceptions.ProxyError as exc:
        raise ProxyBlockedError(str(exc)) from exc
    if resp.status_code in (403, 407):
        raise ProxyBlockedError(f"HTTP {resp.status_code}（proxy CONNECT denied）: {url}")
    if resp.status_code == 404:
        _write_marker(bi5_local_path(pair, d))
        return
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {url}")
    content = resp.content
    if len(content) == 0:
        _write_marker(bi5_local_path(pair, d))
        return
    _atomic_write(bi5_local_path(pair, d), content)


# --------------------------------------------------------------------------- #
# parquet 構築
# --------------------------------------------------------------------------- #
def _recover_corrupt(pair: str, d: date, field_order: str, failed: list) -> Optional[pd.DataFrame]:
    """LZMAError のファイルを削除して 1 回だけ再取得し、再デコードを試みる。"""
    path = bi5_local_path(pair, d)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _download_single(pair, d)
    except Exception as exc:  # noqa: BLE001
        failed.append((pair, d.isoformat(), f"再取得失敗: {exc}"))
        return None
    if not path.exists():
        failed.append((pair, d.isoformat(), "再取得後もファイル無し"))
        return None
    raw = path.read_bytes()
    if len(raw) == 0:
        return None  # マーカー＝データなし
    try:
        return decode_bi5_candles(raw, pair, d, field_order=field_order)
    except lzma.LZMAError as exc:
        failed.append((pair, d.isoformat(), f"再取得後も LZMAError: {exc}"))
        return None


def build_parquet(
    pair: str, start: date, end: date, field_order: str, source: str = "dukascopy"
) -> QualityReport:
    """範囲内のローカル bi5 を全て読み、デコード・連結・ソート・重複除去・検証・保存する。"""
    frames: list[pd.DataFrame] = []
    failed: list = []
    for d in iter_days(start, end):
        path = bi5_local_path(pair, d)
        if not path.exists():
            continue
        raw = path.read_bytes()
        if len(raw) == 0:  # データなしマーカー
            continue
        try:
            df = decode_bi5_candles(raw, pair, d, field_order=field_order)
        except lzma.LZMAError:
            df = _recover_corrupt(pair, d, field_order, failed)
            if df is None:
                continue
        if not df.empty:
            frames.append(df)

    full = pd.concat(frames) if frames else _empty_frame()
    full = full.sort_index(kind="stable")
    dup_mask = full.index.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        full = full[~dup_mask]

    report = validate_bars(full, pair, source=source)
    report.n_dupes_dropped = n_dupes  # 実際に落とした本数で上書き
    if failed:
        warnings.warn(
            f"{pair}: 再取得しても復旧できなかった破損ファイル {len(failed)} 件: {failed}",
            RuntimeWarning,
            stacklevel=2,
        )

    store.save_pair(pair, full, source=source)
    return report
