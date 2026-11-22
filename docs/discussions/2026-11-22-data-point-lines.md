---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../../zcu_lab/v2/figure-style.md
  - 2026-11-22-presentation-rules.md
  - 2026-11-22-shared-presentation-facts.md
basis:
  repo:
    previous_rule: 9888a2df31f9691562c6c01f0deb8980fba1ad64
    verified_candidate: 53efb4dbc5a0fabb968198b53c067791344a66cc
  external: []
---

# 量測點何時連線與 spectrum 的 error bar

## 結論

使用者選擇用同一條規則處理 Run 和 Analyze 的一維掃描。沒有擬合曲線，量測點用折線連接。有擬合曲線，量測資料只畫點與可用的誤差棒。Spectrum 若能用原 shot 資料估計 SEM，也畫逐點誤差棒。這改掉了[原呈現規範](2026-11-22-presentation-rules.md)的峰位例外。類別軸不連線，IQ 散點與二維圖不套用這條規則。

[ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)記錄裁決和責任分工。實驗呈現指南的 [R8](../../zcu_lab/v2/figure-style.md#r8圖型對應資料liveplot-標明進度)說明線型，[R11](../../zcu_lab/v2/figure-style.md#r11有-shot-sem-的一維資料畫-error-bar)說明誤差棒。

## 問題

Local Z 與 Conditional Phase 的 live 圖原先只有點，操作員要逐點追蹤掃描趨勢。反過來，有些 Rabi Analyze 把實測點和密網格模型各畫成一條線，難以看出哪條是擬合。原 R11 讓判讀峰位的 spectrum 省略誤差棒，即使資料足以估計逐點 SEM。

## 考慮過的選項與比較

只改 live 圖可以少動幾個 Analyze caller，卻不能處理 Analyze 中量測線和模型打架的情況。使用者原先指出 two-tone spectrum、Rabi 和 Local Z 都需要能看出趨勢。對比選項後，使用者在 2026-11-22 答：「如果拟合曲线本身就是连线的，那数据点就不用。但如果没有拟合曲线，那就要连线。」範圍選「全面套用」，不只修點名的幾族。

這次更新沿用[既有 SEM 分工](2026-11-22-shared-presentation-facts.md)。Analysis 用原有的分類計數或 IQ 矩估計 SEM，實驗決定在哪張圖呈現。只存平均值的頻譜無法從平均逆推單發散布。取消峰位例外不等於要補取數或改保存格式。

### Wrapped phase 的線段

直接把 ±180° 切口兩側的代表值連起來，會在圖上畫出假的穿零。例如相鄰點 −175° 和 +175°，兩端是實際估出的相位，中間直線卻不是量測結果。這和原 R0 要求讀者只看圖即可判讀衝突。

Designer 代裁只斷相關線段：相鄰有效代表值相差至少 180°，或中間有缺值，就不連。恰半圈時兩個方向等長，不猜方向。保留原點和既有誤差棒，不插入邊界點。這個例外寫在 R8，適用於 Conditional Phase、Heralded Conditional Phase 和 Local Z 的 wrapped phase 圖，不改 unwrap、π crossing 的紅叉、擬合或寫回。使用者在 2026-11-22 覆核，答「可以。」

## 被否決的選項與理由

- 保留 Analyze 原線型，或只改使用者點名的家族。原本疊在模型上的量測線仍會干擾判讀，也不符合全面套用。
- 仍把跨切口的點直連、只在 guide 提醒。操作員還是會看到假的穿零。
- Wrapped phase 全部改成散點，或補線延伸到 ±180° 邊界。前者丟失其餘有效線段的趨勢，後者會引入未量到的邊界位置，可能和量出的 π crossing 混淆。
- 為了畫圖新增取數統計或改成 unwrapped 軸。這會改動保存契約，或改變原圖要回答的問題。

## 留下的問題

只存平均值的實驗沒有逐點 shot SEM，等家族更新取數時再處理。斷線規則不聲稱稀疏樣本足以辨識真實相位路徑。

## 依據

原始問答與離線 probe 圖未入庫。本篇保留裁決、比較理由和代表例。操作者可從 [Conditional Phase](../../zcu_lab/v2/two_qubit/conditional_phase/guide.md)、[Heralded Conditional Phase](../../zcu_lab/v2/two_qubit/conditional_phase/guide_heralded.md)和 [Local Z](../../zcu_lab/v2/two_qubit/local_z/guide.md)的 guide 找到斷線的判讀方式。
