# FX 1分足スキャルピング手法「10万円→2000万円」忠実再現バックテスト計画

## コンテキスト

YouTube動画 (https://www.youtube.com/watch?v=zKxMzCp-CUo) の「SMA400×ボリンジャーバンド1分足スキャルピングで10万円→2000万円」という主張を、**ルール完全再現のバックテストで定量検証**し「本当に稼げるのか」に統計的根拠で答える。リポジトリは空、ゼロから構築。

**役割分担（ユーザー指定・厳守）**: 計画・考察・レポート = **Fable 5**（メインループ）／コード実装 = **Opus 4.8**（`Agent` ツール `model: "opus"`）

**確定事項（ユーザー回答済み）**:
- データ: **Dukascopy**（ユーザーがネットワークポリシーに `datafeed.dukascopy.com` を追加。現状403 → 実装開始時に疎通再確認）
- 範囲: **主要20ペア × 2024-01-01〜2026-06-30（2.5年、約18.6M本）**
- ブランチ: `claude/scalping-backtest-verification-rhxi5n`
- 環境: Python 3.11 / 4CPU / 16GB RAM / 31GB disk。pandas等は PyPI から導入（疎通確認済み）

## 手法スペック（動画ルールの確定版 — SPEC.md としてコミット）

ペア20種: `EURUSD USDJPY GBPUSD AUDUSD NZDUSD USDCAD USDCHF EURJPY GBPJPY AUDJPY NZDJPY CADJPY CHFJPY EURGBP EURAUD EURCHF EURCAD GBPAUD GBPCHF AUDNZD`

指標: SMA400（終値）、BB(20, ±2σ)。σは**母標準偏差 ddof=0**（MT4 iStdDev / TradingView ta.stdev 準拠。ddof=1感度チェックを付録に）

**買い（売りは完全ミラー）**:
1. クロス: `mid[t-1]<=sma400[t-1] かつ mid[t]>sma400[t]` → 買いサイクル開始。**新たなクロス（両向き）は未エントリーサイクルを取消して新サイクル開始**（クロス列がタイムラインを窓 `[t_i, t_i+1)` に分割）。建玉には触れない（約定後放置ルール）
2. ステップ2: サイクル内で最初の「終値 > +2σ」確定足（実体ベース）。**クロス足自身も対象**（検索はクロス足含む）
3. ステップ3: その後（t_break+1以降）最初の「終値 < mid」確定足
4. エントリー: ステップ3足の終値で成行。**1サイクル最初の完成シーケンス1回のみ**。発火でエントリー有無に関わらずサイクル消費
5. SL = エントリー足の −2σ 値、TP = entry + 1.5×(entry−SL)（RR 1:1.5固定）
6. スキップ（理由別に記録）: 建玉あり `position_open` ／ entry<=SL の異常足 `invalid_sl`

**エグジット**（エントリー足の次から走査、両バリアント共通の t_exit）:
- 優先順位: ①始値がSL貫通 → 始値約定 `sl_gap`（曖昧扱いしない） ②同一足で low<=SL かつ high>=TP → **曖昧足**: 保守=SL／楽観=TP ③SLタッチ→SL ④TPタッチ→TP（有利ギャップでもTP価格で約定=保守的） ⑤データ終端 → 最終終値 `eod`
- 週末持ち越しあり。スワップは無視（レポートで制約として明記）

**コスト**: BIDローソクで計算し1往復スプレッド1回控除 `r_net = r_gross − spread_price/risk_dist`。3シナリオ（zero／tight≒低スプ業者／standard≒国内標準）。ペア別表は設計済み（EURUSD 0.2/0.6pips、USDJPY 0.2/0.7、GBPJPY 0.6/1.4、GBPAUD 0.7/2.0 等、config.SPREADS_PIPS に格納）。pip: JPYクオート0.01、他0.0001

**資金管理シム**（初期¥100,000、約定時刻順・同時刻はクローズ先行）:
- (i) 固定リスク¥2,000（純粋統計用） (ii) 動画方式ティア複利: リスク=ティアの2%、資産がティア2倍でラチェット昇格（降格なし） (iii) 連続2%複利
- ポートフォリオ: 全ペア時系列マージ、1ペア1ポジ、ペア間同時保有可（同時数・合計オープンリスク記録）

**検証する主張**: ①勝率がRR1.5の損益分岐40%（コスト前）を有意超過するか（Wilson 95%CI） ②「月利平均10%」 ③連敗実態（動画の"4連敗定期発生"、50トレード窓勝率のブレ） ④「10万→2000万」= プロップチャレンジMC（FTMO型: P1+10%/P2+5%、日次損失5%・最大DD10%、リスク1%/2%、実測R分布と実測トレード数/日でブートストラップ10k回、iid+ブロック）で合格確率・期待コスト・所要期間

## アーキテクチャ（設計エージェント承認済み）

```
├── SPEC.md / README.md / requirements.txt / .gitignore(data/)
├── src/
│   ├── config.py            # 全定数（ペア・pip・スプレッド表・期間・シード・料金）
│   ├── data/{base,dukascopy,histdata_repo,store}.py
│   ├── indicators.py  signals.py  execution.py  costs.py
│   ├── equity.py  metrics.py  montecarlo.py  report.py
├── scripts/{download_data,run_backtest,make_report,spot_check}.py
├── tests/（12ファイル、下記シナリオ網羅）
├── data/     # gitignore（raw bi5 + parquet zstd キャッシュ）
└── results/  # コミット対象: trades/*.csv, summary.json, charts/*.png, report.md
```

**主要契約**: `store.load_pair(pair) → UTC index / OHLCV float64`（ソース非依存）。トレード表スキーマ凍結: `pair, direction, t_cross/t_break/t_entry/t_exit, ts_entry/ts_exit, entry, sl, tp, risk_dist, exit_price_cons/opt, reason_cons/opt(tp|sl|sl_gap|eod), ambiguous, r_gross_cons/opt, hold_minutes` + skips表

**データ層（Dukascopy）**: `https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM0}/{DD}/BID_candles_min_1.bi5`（**月0起点**、URL生成は1箇所に隔離しURL文字列を単体テスト）。LZMA解凍→24byteビッグエンディアン仮説 `>IIIIIf`（秒offset, O, C, L, H, vol）を**経験的に検証**: 実在日をO,C,L,H/O,H,L,C両解釈でデコードし `low<=min(o,c)<=max(o,c)<=high` 違反ゼロの方を採用、価格が妥当レンジ内かで除数(JPY=1e3/他=1e5)も確定。検証済み実データ日をtests/fixtures/にコミットしオフラインテスト化。DL: 8スレッド+Retry+レジューム（404/空=ゼロバイトファイルでマーク）、`REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt`。約18,240ファイル≒200MB、15〜40分。LZMAError=破損→削除・再DL1回

**シグナルエンジン**（全バーPythonループ禁止）: pandas rollingで指標→shift比較でクロス列→サイクル窓ごとに `flatnonzero(close>upper)` 等のソート済みインデックス配列へ `searchsorted` で t_break, t_entry を解決。候補のみのPythonループ（~10³/ペア）で建玉フィルタ＋チャンク化ベクトルTP/SL前方走査（保守/楽観を1走査で同時算出、t_exit共有）。20ペア全体で2分未満、ペア毎逐次処理でメモリ~90MB

## タスク分解

| # | 担当 | 内容 |
|---|------|------|
| 0 | **Fable5** | Dukascopy疎通確認（不通ならユーザーに設定案内して待機）→ SPEC.md+計画+スケルトンを最初にコミット/push（セッション再起動保険）→ pip install |
| 1 | **Opus 4.8** (agent A) | データ層: bi5 DL/デコード/レイアウト経験検証/parquetキャッシュ/QualityReport + データ系テスト。受入=実在3平日+週末1日のデコードがOHLC不変条件・妥当レンジ・平日~1440本を満たす → 全量DLをバックグラウンド起動 |
| 2 | **Opus 4.8** (agent B, Aと並行可) | エンジン: indicators/signals/execution/costs/equity + run_backtest.py + spot_check.py + 合成データテスト一式（下記） |
| 3 | **Opus 4.8** (agent C) | 分析系: metrics/montecarlo/report + make_report.py + 図表生成（英語ラベル、candlestickヘルパー自作） |
| 4 | **Fable5** | 全パイプライン実行 → スポットチェック25トレード+10スキップの独立再計算照合レビュー → サンプルセットアップのチャート目視 → **統計考察・report.md（日本語）執筆** → コミット/push |

各Opusエージェントには本計画＋設計詳細を仕様書として全文渡す。修正が必要なら同エージェントに差し戻し（SendMessage）

## 検証（正しさの担保）

1. **pytest 合成シナリオ**（実装受入条件）: warmup400本無シグナル／クロス足=ブレイク足／ステップ順序強制（先行プルバック無視）／反対クロスの途中取消／ゼロタッチ再クロスのリセット／1サイクル1回制限／スキップ時もサイクル消費／invalid_slスキップ／建玉中スキップ→次サイクルは入る／曖昧足の保守SL・楽観TP同t_exit／SLギャップ始値約定／TP有利ギャップはTP価格／エントリー足自身のヒゲ無視／データ終端eod／ロング・ショート鏡像対称／BB・SMA手計算照合＋ddof配線ガード／RR算術／JPY・非JPYのpip×spread換算12桁精度／重複分デデュープ／bi5往復（合成pack→LZMA→decode）＋実データfixture／ティア昇格ラチェット（降格なし・2段跳び）／同時刻クローズ先行／Wilson既知値／MC決定性+日次損失5%違反シナリオ
2. **スポットチェック**（Task 4でFable5がレビュー）: 固定シードで25トレード+10スキップ抽出、pandasを使わない素朴ループで指標を独立再計算し全ルール成立を機械照合（25/25, 10/10必須）。サニティ: 平均損失≈−1R−spread、sl_gap以外でr<−3なし、勝率25〜65%帯
3. **目視**: sample_setups/ のローソク足PNG（BB・SMA400・entry/SL/TP描画）5枚を動画の説明パターンと照合
4. **データ品質**: ペア別カバレッジ・欠損分・OHLC不変条件違反0・価格レンジ・フラッシュクラッシュ足（risk_dist>5×中央値）フラグ
5. 一括実行: `pip install -r requirements.txt && pytest && python scripts/download_data.py && python scripts/run_backtest.py && python scripts/make_report.py`

## 成果物

- 実装一式＋テスト＋README（実行手順）
- **results/report.md（日本語・Fable5執筆）**: ①概要 ②検証条件（ddof根拠・スプレッド表含む） ③データ品質 ④全体結果（保守/楽観×コスト3種: N・勝率±CI・期待値R・PF・保有時間・トレード/日） ⑤主張検証（40%分岐・月利10%・10万→2000万+プロップMC） ⑥ペア別/年別（2026は部分年と明記） ⑦資金管理3方式の資産曲線・最大DD・連敗分布（実測+シャッフル10k p50/p95） ⑧動画式「最初の50トレード」表と50窓勝率分布 ⑨制約（BID足のみ・スワップ無視・終値約定仮定・スリッページ=スプレッドのみ・プロップ日次境界UTC近似） ⑩結論
- 図表: 資産曲線3種／コスト別資産曲線／DD曲線／月次リターンヒートマップ／rolling50勝率／R分布／保有時間／ペア別期待値バー／MC合格確率／サンプルセットアップ

## リスクと対応

- **Dukascopy未許可のまま**: Task 0で検知 → ユーザーへ手順案内し待機（代替: philipperemy/FX-1-Minute-Data を add_repo でスパースクローン、〜2024年で実施） 
- **セッション再起動**: 計画・SPEC・進捗を都度ブランチにコミットして常時再開可能に
- **bi5構造の想定違い**: 両仮説の経験的判別を受入条件化。判別不能なら tick データ (`{HH}h_ticks.bi5`) から1分足を自前集計する代替経路
- **DL長時間化**: レジューム対応＋主要8ペア先行で早期に暫定結果 → 残12ペア追いがけ
- **フラット窓σ=0・異常足**: strict比較+invalid_slガード+risk_dist>0アサート
