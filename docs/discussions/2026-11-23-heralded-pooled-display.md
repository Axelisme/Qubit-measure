---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-20-heralded-experiment-pruning.md
  - 2026-11-23-phase-figures-physics-order.md
  - ../../zcu_lab/v2/two_qubit/rb/guide_heralded.md
  - ../../zcu_lab/v2/two_qubit/conditional_phase/guide_heralded.md
basis:
  repo_commit: 2b8f628c31a181dfd9fe187859408c9f4d236c7e
  external_sources: none
---

# Heralded RB 與 conditional phase 合併顯示 gg、ee

## 結論

Heralded RB 的主圖與 Live 每個 arm 顯示一個量：在 gg 或 ee herald 的 shots 中，有多少回到原態。Heralded conditional phase 按實際 spectator 態，先把 ee 的投影對齊 gg，再合併投影計數。兩個實驗把分組曲線移到診斷，存檔、聯合擬合與寫回不變。

## 問題

原圖用 herald 分組呈現量測點，聯合擬合卻在問共用參數。RB 問兩種 herald 態的回歸如何衰減，conditional phase 問兩種分組各自對齊後的共用相位斜率。操作者不必先比較每條分組線才看到這兩個量。使用者說：「有些實驗我通常不在意gg還是ee，我在意的是相同還是相異」，對兩個入口答「兩個入口都合」。

## 選項與比較

- RB 在每個實測深度合併回歸 shots，除以同深度兩組 herald shots 之和。合併擬合線用同樣的逐深度權重組合原聯合擬合的兩個預測。沒有 herald shots 的深度不取權重；有權重的深度之間插值。比例不隨深度改變時，合併線仍是同一個衰減常數的單指數。
- Conditional phase 延續[依物理推導排圖](2026-11-23-phase-figures-physics-order.md)的三格。前兩格使用合併投影。累積相位格用和測點相同的計數合併聯合擬合的預測投影，再取角度。測點的 360° 分支以該線為參考，沒有增加擬合。第三格仍用聯合斜率差決定 π 交點。

## 未採用

- 只改 Live：使用者選擇兩處都改，因為 Analyze 也有相同的判讀問題。
- RB 每個 arm 用一組固定權重：比例隨深度變動時，線與測點權重不同，圖上的差就不是同一個量的殘差。
- Conditional phase 主圖保留 gg offset 線：兩組 offset 不同時，單組參考線和合併點的偏離不代表擬合誤差。
- Conditional phase 主圖只留合併點：沒有參考線，讀者無法在累積相位格把兩支斜率連到第三格的斜率差。

## 留下的問題

Heralded conditional spectroscopy 的四個分支是各自的分析對象，維持分組，不套用此次合併畫法。

## 依據

本 repo `2b8f628c3` 的兩個 heralded 入口、[v2 README](../../zcu_lab/v2/README.md)〈gg／ee 共用參數〉與 2026-11-23 的使用者裁決。當時的裁決原檔未入庫；本文保留顯示取捨及不變的分析行為。
