---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-19-single-qubit-rb-irb-merge.md
  - 2026-11-20-rb-depth-program.md
basis:
  repo:
    before_round_reduction: 282e98191d81b677386362c4b174c86cd7d89f25
  external: []
---

# RB 取數歸約與衰減擬合品質

## 結論

三個 RB 實驗把 round 數傳給 acquire。兩個 two-qubit 實驗加總計數，單 qubit 實驗平均 IQ。Result 一律以 (seed, arm, depth) 開頭，不保存 round 軸。兩個 two-qubit 分析回報衰減長度的相對誤差，不再用尾端存活率門檻發警告。取數原則見 [ADR-0083](../adr/0083-program-macros-and-timelines.md#round-在取數端歸約)。

## 問題

舊版主機每個 round 呼叫一次 acquire，並把結果寫入 round 軸。分析先丟掉不完整的 seed／round 配對，再對每個 seed 平均 round。兩個 two-qubit 實驗以 depth 開頭，單 qubit 實驗以 round 開頭。

使用者先問：「还有不懂为什么你要软体扫 ROM 的，这东西不是 acquire 可以直接给的吗？」（ROM 指 round。）接著指出：「我得要提醒你，其实结果的 round 的轴是多余的，我想应该是之前实做的人的瑕疵。round 的轴应该在 round 函数内就已经被平均掉了，而且呢，这个平均应该是 reducer。负责的吧？那你外部再乱好几次就更奇怪了。我可以接受单 Q B 跟 Two Q B 要一致，也就是轴的顺序一致，所以是可以统一移到。最内侧，这样就少了一个维度，还有少了手动平均的难度。」

Plain two-qubit RB 原先在最深點的 reference gg 存活率高於 0.2 時提示加深掃描。這個數字也受初態與讀出影響，不能單靠它判斷衰減長度可否估準。使用者說：「关于警告的部分，我觉得这应该是拟合品质检查的机制，而它应该在分析的回传中就包含了品质的衡量。这不是应该有警告的东西，这应该是写在结果里的东西。」接著問：「fit quality 应该也可以看出来这个拟合的可信度有多少。那既然已经可以看可信度了，还需要看残余量吗？」

## 考慮過的選項與比較

- **在取數時歸約。** 每個 seed、arm 的板上 program 只呼叫一次。這延續[板上 RB 程式改寫](2026-11-20-rb-depth-program.md)的 program 邊界，不改脈衝或 Clifford 序列。Stop 保留已完成 rounds 的總和或平均。Two-qubit 各格以實際計數總和為分母，分析依完整 seed 等權擬合。Heralded 兩組以各格接受的 shot 數計算 binomial 誤差。
- **在分析結果回報衰減長度相對誤差。** 短掃描可以貼合觀測點，但衰減長度和 offset 仍難以分開估計。從 covariance 取衰減長度的標準誤，再除以長度。Heralded 的兩組共用同一 arm 的衰減，所以回報同一個相對誤差。

比較使用 p = 0.98、雜訊 σ = 0.01、11 個 depth 點：

| 最大 depth | R² | 衰減長度相對誤差 |
| ---: | ---: | ---: |
| 30 | 0.993 | 0.30 |
| 50 | 0.997 | 0.14 |
| 80 | 0.998 | 0.08 |
| 200 | 0.999 | 0.035 |

最大 depth 30 時，p 的擬合平均為 0.9752，標準差為 0.0035。到 depth 200 時，平均為 0.9799，標準差為 0.0007。即使 R² 都高於 0.99，短掃描的衰減長度相對誤差仍約為最深掃描的 8.6 倍。

## 被否決的選項與理由

- **保留主機迴圈，或取數後再合併。** 這會保留重複呼叫和手動整理資料。取數能保留已完成的 rounds。實驗不必另外判斷哪些 rounds 跑完。
- **保留警告，或在分析新增最深點的殘餘存活率。** 相同的尾端存活率可以來自不同 offset 或不同衰減長度。殘差與 R² 只描述模型貼合觀測點的程度。使用者接受在 `fit_quality.relative_parameter_errors` 回報衰減長度相對誤差，不加殘餘量欄位。對取數與品質兩項提案答「其他采用建议」。

## 留下的問題

單 qubit 的每個 seed 固定先跑 reference，再跑可選的 interleaved。舊版交替臂順序以平衡漂移。這次沒有量測兩種順序的漂移差異。

舊檔帶 round 軸，新版 Load 因軸數不合拒絕。沿用[單 qubit RB／IRB 合併](2026-11-19-single-qubit-rb-irb-merge.md)的先例，只保留原檔，不做轉換。這次 cfg 欄位和型別不變，保留 cfg schema 版本；結果軸由載入端核對。

## 依據

- 變更前的 program、Result 與分析以本 repo `282e98191` 為基準。
- 合成比較的腳本未入庫。上表及正文保留測試條件和數字。
