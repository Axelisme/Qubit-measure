---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-19-single-qubit-rb-irb-merge.md
  - 2026-11-20-rb-round-reduction-and-fit-quality.md
basis:
  repo:
    before_errorbars: f99619a009e503eb415cff62ccd3b0a594d07fa7
  external: []
---

# RB 每個 depth 的誤差棒

## 結論

普通 two-qubit RB、heralded two-qubit RB 和單 qubit RB 在即時圖與 decay 擬合圖畫 seed 間的樣本標準差。普通 two-qubit RB 的 pooled survival 圖也畫同一種誤差棒。不足兩個 seed 時不畫棒。沒有另畫 shot noise 參考線，也沒有改 Analyze 回傳、Result 或擬合權重。RB 與 IRB 的數值工具併入 `analysis.fitting.rb`，沿用 `analysis.fitting` 的匯出名稱。數值責任見 [ADR-0085](../adr/0085-numerical-fitting-owners.md#rb-圖上的-seed-間標準差)。

## 問題

2026-11-20，使用者說：「rb實驗的liveplot與分析，我希望能給每個點顯示errorbar，因為有些資訊可以從此判斷，比如ding的博倫中提過，coherence error的rb的標準差會隨著gate數變大而增加」。「博倫」指博論。使用者也問：「為什麼有irb.py?我以為都合併了」。前一次合併處理的是[單 qubit 實驗入口](2026-11-19-single-qubit-rb-irb-merge.md)，當時沒有合併 fitting 模組。

## 考慮過的選項與比較

| 誤差棒 | 回答的問題 |
| --- | --- |
| Seed 間樣本標準差，`ddof=1` | 同一個 depth 的不同序列，存活率或投影 IQ 相差多少？ |
| 標準誤，標準差除以 seed 數的平方根 | 等權 seed 平均值估得多準？即使序列差異一樣，增加 seed 也會使棒變短。 |
| Shot noise | 在固定存活率與 shot 數下，有限取樣有多大的波動？不含不同序列造成的差異。 |

另比較了是否畫 shot noise 參考線、是否在 Analyze 回傳中加欄位。模組有三個選項：合成一個 `rb.py`、只改 `irb.py` 的名字、保持兩檔不動。

## 裁決與未採用的方案

使用者於 2026-11-20 對三個顯示問題答「1A 2B 3A」。畫第一項 seed 間標準差，因為要看序列的分散程度，不是平均值的精度。只畫這一組誤差棒，不另外畫 shot noise 參考線，也不把逐點數字放進回傳。Heralded 原本用來加權的 binomial 標準誤繼續供擬合使用。

同日，使用者對模組選項答「A」。兩組工具放在同一個 `rb.py`。只改名或保持兩檔，都不會把數值工具合到同一處。這次沿用[取數歸約](2026-11-20-rb-round-reduction-and-fit-quality.md)後的 Result，不再改資料軸。

## 留下的問題

誤差棒沒有把序列差異與 shot noise 拆開。對存活率 `P` 和 `N` 發 shot，shot noise 約為 `√(P(1−P)/N)`。當 `P` 從接近一降向一半，這一項也會增加。因此棒隨 depth 變長，不能單憑這張圖判定有 coherent error。

## 依據

- 變更前的 RB 與 IRB 數值工具、三個實驗與其 Result 以本 repo `f99619a00` 為基準。
- [數值入口](../../lib/zcu_tools/analysis/fitting/README.md#randomized-benchmarking)定義哪些 seed 參與即時圖和擬合。
