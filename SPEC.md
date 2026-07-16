# SPEC — 1分足スキャルピング手法 完全仕様（実装の唯一の正）

検証対象: YouTube動画 https://www.youtube.com/watch?v=zKxMzCp-CUo で解説される
「SMA400×ボリンジャーバンドの1分足スキャルピング（10万円→2000万円）」手法。
本ファイルが実装のリファレンス。曖昧な点はここに追記して確定させる。

## 1. データ契約

- ソース: Dukascopy BID 1分足ローソク（`BID_candles_min_1.bi5`）
- 期間: 2024-01-01 〜 2026-06-30（UTC）
- ペア: `config.PAIRS`（20ペア）
- `store.load_pair(pair) -> pd.DataFrame`:
  - index: `time` — UTC tz-aware DatetimeIndex、狭義単調増加・重複なし
  - columns: `open, high, low, close, volume` — すべて float64
  - 欠損分は行を持たない（分グリッドへの reindex はしない）
- ローリング計算は**バー本数ベース**（時間ベースではない）。ギャップをまたいでも
  直近 N 本で計算する（MT4/TradingView のチャート挙動と同一）

## 2. 指標

- `sma400[t]` = 直近400本の終値単純平均（`close.rolling(400).mean()`）
- `mid[t]` = 直近20本の終値単純平均（BB中央線）
- `sigma[t]` = 直近20本の終値の**母標準偏差（ddof=0）**
  （MT4 iStdDev / TradingView ta.stdev 準拠。ddof=1 は感度分析のみ）
- `upper[t] = mid[t] + 2.0 * sigma[t]`、`lower[t] = mid[t] - 2.0 * sigma[t]`
- ウォームアップ: 有効値は sma400 が定義される index 399 以降。
  NaN を含む比較はすべて False（シグナル不成立）

## 3. シグナル（サイクルとシーケンス）

`diff[t] = mid[t] - sma400[t]`（両者非NaNのときのみ定義）

- **上抜けクロス** at t: `diff[t-1] <= 0 かつ diff[t] > 0` → 買いサイクル開始
- **下抜けクロス** at t: `diff[t-1] >= 0 かつ diff[t] < 0` → 売りサイクル開始
- クロス列（時刻順）がタイムラインを窓 `[t_i, t_{i+1})` に分割する（最後は `[t_last, n)`）。
  **新たなクロス（どちら向きでも）は未エントリーのサイクルを取消し、新サイクルを開始する。**
  ちょうど 0 タッチ（diff が +,0,+）は同方向クロスが2回発生し、2回目でサイクルが
  リセットされる（仕様として許容）
- **買いサイクル `[s, e)` 内のシーケンス**（売りは完全ミラー）:
  1. `t_break` = `[s, e)` 内で最初の `close[t] > upper[t]` の足（**クロス足 s 自身も対象**）
  2. `t_entry` = `[t_break+1, e)` 内で最初の `close[t] < mid[t]` の足
  3. 両方見つかった場合のみ候補（Candidate）成立。
     **1サイクルにつき最初の完成シーケンス1回のみ**（発火後はサイクル消費）
- 候補の値:
  - 買い: `entry = close[t_entry]`, `sl = lower[t_entry]`, `tp = entry + 1.5*(entry - sl)`
  - 売り: `entry = close[t_entry]`, `sl = upper[t_entry]`, `tp = entry - 1.5*(sl - entry)`

## 4. 執行

候補を t_entry 順に処理（サイクル窓が互いに素なので自然に時刻順）:

- **スキップ（skips表に記録、サイクルは消費済み）**:
  - `position_open`: 同一ペアに建玉が残っている（`t_entry <= open_until`）
  - `invalid_sl`: 買いで `entry <= sl`、売りで `entry >= sl`（異常足）
- **エグジット走査**: `t_entry+1` から前方走査（エントリー足自身の高安は無視）。
  最初に SL/TP のどちらかに触れた足 `t_exit` で確定。判定優先順位（買いの場合）:
  1. `open[t] <= sl` → reason `sl_gap`、約定価格 = `open[t]`（両バリアント共通。
     同じ足で TP に触れていても曖昧扱いしない — 寄付で損切りが先に執行される）
  2. `low[t] <= sl かつ high[t] >= tp` → **曖昧足**: 保守 = SL約定 / 楽観 = TP約定。
     `ambiguous = True`
  3. `low[t] <= sl` → SL約定（価格 = sl）
  4. `high[t] >= tp` → TP約定（価格 = tp。有利ギャップでも tp 価格 = 保守的）
  5. データ終端まで触れず → reason `eod`、価格 = 最終 close
- **保守/楽観バリアントは同一の走査を共有し `t_exit` は同一**（曖昧足でも判定バーは
  同じ。違うのは約定価格と reason のみ）。よって建玉フィルタも両者で同一
- `r_gross = direction * (exit_price - entry) / risk_dist`、`risk_dist = |entry - sl| > 0`
- 週末・祝日は持ち越し（動画の「約定後は放置」ルール）。スワップは無視（§10）

## 5. コスト

BIDローソクのみで計算し、**1往復につきスプレッド1回分**を価格ベースで控除:

```
spread_price = SPREADS_PIPS[scenario][pair] * PIP_SIZE[pair]
r_net = r_gross - spread_price / risk_dist
```

シナリオ: `zero`（動画の暗黙前提）/ `tight`（低スプレッド業者）/ `standard`（国内標準級）。
値は `config.SPREADS_PIPS` が唯一の定義。

## 6. 出力スキーマ（凍結 v1.1）

`results/trades_all.csv` — 1トレード1行:

```
pair, direction(1|-1), t_cross, t_break, t_entry, t_exit,
ts_entry, ts_exit (UTC ISO8601),
entry, sl, tp, risk_dist,
exit_price_cons, reason_cons(tp|sl|sl_gap|eod),
exit_price_opt,  reason_opt,
ambiguous(bool), r_gross_cons, r_gross_opt,
hold_bars, hold_minutes,
r_net_cons_zero, r_net_cons_tight, r_net_cons_standard,
r_net_opt_zero,  r_net_opt_tight,  r_net_opt_standard
```

`results/skips_all.csv`: `pair, ts(UTC), direction, reason(position_open|invalid_sl)`

`results/summary.json`: ペア別＋全体の要約（bars, candidates, trades, skips内訳,
ambiguous数, reason内訳, 勝率などの速報値）

`reason == "eod"` のトレードは勝率・期待値の統計から除外し、資産曲線には含める。

## 7. 資金管理シミュレーション（初期資金 ¥100,000）

トレード列（全ペア時系列マージ）へ R 倍数ベースで適用: `PnL = リスク額 × r_net`。
通貨換算・ロット丸めは行わない簡略化（§10 に明記）。

- **(i) fixed**: リスク額 = ¥2,000 固定（複利なし、純粋統計用）
- **(ii) tiered**（動画方式）: リスク額 = tier_base × 2%。決済時に
  `while equity >= 2 * tier_base: tier_base *= 2`（ラチェット昇格のみ、降格なし。
  2倍跳びも可）。初期 tier_base = 100,000
- **(iii) pct**: リスク額 = エントリー時点の残高 × 2%（連続複利）

共通ルール: イベントは時刻順、**同時刻はクローズ処理がエントリー処理より先**。
エントリー時にリスク額を確定（建玉中の残高変動の影響を受けない）。
equity <= 0 で破産（ruined フラグ、以後の新規エントリー停止）。
記録: 資産曲線、最終資産、最大DD（¥・%）、最長ドローダウン期間、月次リターン表、
最大同時保有数、最大合計オープンリスク。

## 8. 検証する主張とメトリクス

1. **勝率**: RR1:1.5 の損益分岐勝率 40%（コスト前）を Wilson 95%CI 込みで上回るか
2. **月利10%**: 資金管理 (ii)(iii) の月次リターン分布・平均月利
3. **連敗**: 最大連敗の実測値＋シャッフル分布（10,000回、p50/p95）。
   50トレード窓の勝率のブレ（動画の「50回検証」がどこまで当てになるか）
4. **10万→2000万**: (a) 自己資金 tiered シムの到達可否・所要期間、
   (b) プロップチャレンジMC（FTMO型近似: P1 +10% / P2 +5%、日次損失5%（UTC日・
   日初残高基準）、最大DD10%、リスク1%/2%、実測 r_net 分布と実測トレード数/日を
   iid＋ブロック(20)ブートストラップ、10,000シム、固定シード）→ 合格確率・
   期待受験回数・期待費用（`config.PROP_FEE_JPY`）・所要日数分布

統計は {保守, 楽観} × {zero, tight, standard} × {全体, ペア別, 年別} で算出。
2026年は部分年（〜6月）として明記し、年率換算しない。

## 9. 実装確定事項（設計判断）

| 論点 | 決定 |
|---|---|
| BB σ | ddof=0（感度分析で ddof=1 併記） |
| ステップ2の探索開始 | クロス足を含む |
| ステップ3の探索開始 | t_break+1 から |
| TP約定価格 | 常に tp ちょうど（有利ギャップでも） |
| SLギャップ約定 | 寄付価格（不利フィル）、曖昧扱いしない |
| t_exit | 保守/楽観で共有（1走査） |
| エントリー価格 | シグナル足終値（`--entry-next-open` は感度分析用オプション） |
| dtype | float64 固定（float32 禁止: 3桁JPY価格のσとクロス判定が劣化） |
| ローリング | バー本数ベース、グリッド reindex なし |
| ティア | 昇格のみ・降格なし |
| 同時刻イベント | クローズ→エントリーの順 |
| ソース混在 | 禁止（parquet メタデータ `source` で強制） |

## 10. 制約・既知の簡略化（レポートに明記）

- BIDローソクのみ（ASK側の非対称は往復スプレッド1回控除で近似）
- スリッページはスプレッド控除のみ、約定は終値/レベル価格を仮定
- スワップ（週末持ち越し含む）無視
- ロット丸め・pip価値の通貨換算なし（R倍数ベース）
- プロップMCの「日次」は UTC 日で近似（実際はブローカーの CE(S)T midnight）
- 祝日・流動性低下時間帯のフィルタなし（動画にも該当ルールなし）
