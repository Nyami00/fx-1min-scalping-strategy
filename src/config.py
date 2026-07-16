"""検証全体で共有する定数（仕様値）。ロジックは置かない。

値の根拠は SPEC.md を参照。実装側はこのファイルの既存値を変更しないこと
（不足があれば追記のみ可）。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "dukascopy"
PARQUET_DIR = DATA_DIR / "parquet"
RESULTS_DIR = REPO_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

PAIRS: list[str] = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "EURGBP", "EURAUD", "EURCHF", "EURCAD", "GBPAUD", "GBPCHF", "AUDNZD",
]

PIP_SIZE: dict[str, float] = {p: (0.01 if p.endswith("JPY") else 0.0001) for p in PAIRS}

# 1往復あたり1回控除するスプレッド幅（pips）
_TIGHT: dict[str, float] = {
    "EURUSD": 0.2, "USDJPY": 0.2, "GBPUSD": 0.3, "AUDUSD": 0.2, "NZDUSD": 0.3,
    "USDCAD": 0.3, "USDCHF": 0.3, "EURJPY": 0.4, "GBPJPY": 0.6, "AUDJPY": 0.4,
    "NZDJPY": 0.6, "CADJPY": 0.5, "CHFJPY": 0.6, "EURGBP": 0.3, "EURAUD": 0.5,
    "EURCHF": 0.4, "EURCAD": 0.5, "GBPAUD": 0.7, "GBPCHF": 0.7, "AUDNZD": 0.6,
}
_STANDARD: dict[str, float] = {
    "EURUSD": 0.6, "USDJPY": 0.7, "GBPUSD": 0.9, "AUDUSD": 0.8, "NZDUSD": 1.0,
    "USDCAD": 0.9, "USDCHF": 0.9, "EURJPY": 1.0, "GBPJPY": 1.4, "AUDJPY": 1.1,
    "NZDJPY": 1.6, "CADJPY": 1.4, "CHFJPY": 1.6, "EURGBP": 0.9, "EURAUD": 1.5,
    "EURCHF": 1.2, "EURCAD": 1.6, "GBPAUD": 2.0, "GBPCHF": 2.0, "AUDNZD": 1.8,
}
SPREADS_PIPS: dict[str, dict[str, float]] = {
    "zero": {p: 0.0 for p in PAIRS},
    "tight": _TIGHT,
    "standard": _STANDARD,
}
COST_SCENARIOS: tuple[str, ...] = ("zero", "tight", "standard")

# データ期間（UTC）
DUKA_START = date(2024, 1, 1)
DUKA_END = date(2026, 6, 30)

# 手法パラメータ（SPEC.md §2, §3）
BB_PERIOD = 20
BB_DEV = 2.0
SMA_PERIOD = 400
RR = 1.5
BB_DDOF = 0

# 資金管理（SPEC.md §7）
START_CAPITAL_JPY = 100_000
FIXED_RISK_JPY = 2_000
RISK_PCT = 0.02

# bi5 デコード健全性チェック用の許容価格レンジ（2019〜2026を余裕を持ってカバー。
# 目的は桁誤り・フィールド順誤りの検出であり、タイトである必要はない）
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "EURUSD": (0.85, 1.35), "USDJPY": (90.0, 180.0), "GBPUSD": (0.95, 1.50),
    "AUDUSD": (0.50, 0.90), "NZDUSD": (0.48, 0.85), "USDCAD": (1.15, 1.55),
    "USDCHF": (0.70, 1.15), "EURJPY": (105.0, 200.0), "GBPJPY": (115.0, 230.0),
    "AUDJPY": (55.0, 120.0), "NZDJPY": (55.0, 110.0), "CADJPY": (68.0, 130.0),
    "CHFJPY": (100.0, 220.0), "EURGBP": (0.80, 0.98), "EURAUD": (1.40, 2.05),
    "EURCHF": (0.85, 1.20), "EURCAD": (1.35, 1.75), "GBPAUD": (1.55, 2.30),
    "GBPCHF": (0.95, 1.40), "AUDNZD": (0.95, 1.30),
}

# 乱数・検証
SEED = 42
SPOT_CHECK_TRADES = 25
SPOT_CHECK_SKIPS = 10
STREAK_SHUFFLES = 10_000

# プロップファーム・チャレンジMC（FTMO型近似、SPEC.md §8）
MC_SIMS = 10_000
MC_BLOCK = 20
PROP_PROFIT_TARGET_P1 = 0.10
PROP_PROFIT_TARGET_P2 = 0.05
PROP_DAILY_LOSS = 0.05
PROP_MAX_DD = 0.10
PROP_RISK_VARIANTS: tuple[float, ...] = (0.01, 0.02)
PROP_FEE_JPY: dict[str, int] = {"10k": 16_000, "100k": 90_000}  # 受験料の概算
