# FX 1分足スキャルピング手法 検証プロジェクト

YouTube動画 https://www.youtube.com/watch?v=zKxMzCp-CUo で解説される
「SMA400×ボリンジャーバンド1分足スキャルピングで10万円→2000万円」という手法を
**ルール完全再現のバックテストで定量検証**するリポジトリ。

- 手法の確定仕様: [SPEC.md](SPEC.md)（実装の唯一の正）
- 実装計画: [docs/PLAN.md](docs/PLAN.md)
- 検証結果: `results/report.md`（生成後）

## 実行手順

```bash
pip install -r requirements.txt
pytest -q                                 # ルール実装のユニットテスト
python scripts/download_data.py          # Dukascopyから1分足を取得（要ネットワーク許可）
python scripts/run_backtest.py           # シグナル検出→トレード生成→results/trades_all.csv
python scripts/spot_check.py             # 検出トレードの独立再計算による照合
python scripts/make_report.py            # 統計・図表・レポート生成
```

データは `data/` にキャッシュされ、リポジトリにはコミットしない（`.gitignore`）。
`results/` が成果物としてコミットされる。

## 実装者向け落とし穴チェックリスト

- Dukascopy URL の**月は0起点**（2024年1月 → `/2024/00/`）。変換は `bi5_url` /
  `bi5_local_path` の1箇所に隔離し、URL文字列そのものを単体テストする
- Dukascopy のローソクファイルは**休場時間帯もフラットな埋め草バー**（volume=0 かつ
  高値=安値、全体の約3割）を含む。parquet 構築時に必ず除外する（SPEC.md §1）
- 週末・祝日の bi5 は HTTP 404 **または** 200で0バイト。どちらも「データなし」として
  ゼロバイトのローカルファイルを置いてレジューム可能にする。デコード時の `LZMAError`
  は破損DL → 削除して1回だけ再取得
- bi5 のフィールド順（O,C,L,H か O,H,L,C か）と除数（JPYクオート=1e3 / 他=1e5）は
  **実データの経験的検証で確定**（OHLC不変条件違反ゼロの解釈を採用）。検証済み実
  データ日を `tests/fixtures/` にコミットしてオフラインテスト化する
- 2026年は部分年（〜6月）。年次集計で年率換算しない
- フラッシュクラッシュ足では `invalid_sl` スキップや `sl_gap` の大幅な不利フィルが
  発生し得る。`risk_dist > ペア中央値×5` のトレードはフラグを立てて残す（削除しない）
- σ=0（完全フラット窓）: strict比較でブレイク不成立、`invalid_sl` ガードで
  `risk_dist=0` を排除。それでも `risk_dist > 0` をアサートする
- ネットワークはプロキシ経由。requests は `REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt`
  を尊重すること（`verify` を上書きしない）。プロキシの 403 CONNECT は「ドメイン
  未許可」— リトライループせず明確なメッセージで失敗させる
- 乱数はすべて `config.SEED` からシード。ポートフォリオのマージは `(timestamp, pair)`
  でソートし決定的にする
