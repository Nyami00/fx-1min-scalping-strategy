#!/usr/bin/env python
"""Dukascopy 1分足のダウンロード & parquet 構築 CLI。

フロー: download_range → 各ペアの最初の非空日でレイアウト検証（勝者はペア間で一致
することをアサート）→ ペアごとに build_parquet → QualityReport 表を表示。
冪等・レジューム可能。失敗時は非ゼロ終了。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.base import QualityReport  # noqa: E402
from src.data.dukascopy import (  # noqa: E402
    ProxyBlockedError,
    bi5_local_path,
    build_parquet,
    download_range,
    fetch_day,
    iter_days,
    verify_bi5_layout,
)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dukascopy 1分足の取得と parquet 構築")
    p.add_argument("--pairs", nargs="+", default=list(config.PAIRS), help="対象ペア")
    p.add_argument("--start", type=date.fromisoformat, default=config.DUKA_START, help="開始日 (ISO)")
    p.add_argument("--end", type=date.fromisoformat, default=config.DUKA_END, help="終了日 (ISO)")
    p.add_argument("--workers", type=int, default=8, help="並列ワーカー数")
    p.add_argument("--skip-download", action="store_true", help="DL を行わず parquet 構築のみ")
    p.add_argument(
        "--verify-layout",
        nargs=2,
        metavar=("PAIR", "YYYY-MM-DD"),
        help="指定ペア・日で verify_bi5_layout を表示して終了",
    )
    return p.parse_args(argv)


def _first_nonempty_local(pair: str, start: date, end: date):
    """範囲内で最初の「存在する非空ローカル bi5」を (day, raw) で返す。無ければ None。"""
    for d in iter_days(start, end):
        path = bi5_local_path(pair, d)
        if path.exists():
            raw = path.read_bytes()
            if len(raw) > 0:
                return d, raw
    return None


def _print_reports(reports: list[QualityReport]) -> None:
    if not reports:
        print("(レポート無し)")
        return
    table = pd.DataFrame([r.as_dict() for r in reports])
    print(table.to_string(index=False))


def main(argv=None) -> int:
    args = _parse_args(argv)

    # --verify-layout: 単発検証して終了
    if args.verify_layout:
        pair, dstr = args.verify_layout
        d = date.fromisoformat(dstr)
        raw = fetch_day(pair, d)
        result = verify_bi5_layout(raw, pair, d)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # 1) ダウンロード
    if not args.skip_download:
        stats = download_range(args.pairs, args.start, args.end, workers=args.workers)
        print(
            f"[download] downloaded={stats.downloaded} skipped={stats.skipped_existing} "
            f"empty={stats.empty_days} failed={len(stats.failed)}"
        )
        if stats.failed:
            for pair, day, msg in stats.failed[:20]:
                print(f"  failed: {pair} {day} {msg}", file=sys.stderr)

    # 2) レイアウト検証（最初の非空日、勝者はペア間一致をアサート）
    winners: dict[str, str] = {}
    for pair in args.pairs:
        found = _first_nonempty_local(pair, args.start, args.end)
        if found is None:
            print(f"[layout] {pair}: 非空データが見つからず検証をスキップ", file=sys.stderr)
            continue
        d, raw = found
        res = verify_bi5_layout(raw, pair, d)
        winners[pair] = res["winner"]
        print(f"[layout] {pair} {d.isoformat()} winner={res['winner']}")

    unique = set(winners.values())
    if len(unique) > 1:
        print(f"[layout] レイアウト勝者がペア間で不一致: {winners}", file=sys.stderr)
        return 1
    field_order = unique.pop() if unique else "OCLH"
    print(f"[layout] 採用フィールド順: {field_order}")

    # 3) parquet 構築
    reports: list[QualityReport] = []
    for pair in args.pairs:
        report = build_parquet(pair, args.start, args.end, field_order)
        reports.append(report)

    # 4) QualityReport 表
    _print_reports(reports)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProxyBlockedError as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        raise SystemExit(1)
