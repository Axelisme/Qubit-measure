---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#報告只寫本次做了什麼
  - ../adr/0074-flat-readout-module.md#classifier-隨-readout-保存
  - 2026-11-23-singular-matrix-as-absent.md
  - 2026-11-20-recipe-report-polish.md
basis:
  repo_commit: d7911319985a4bfe17aec50b7ecb60953caa88b0
  external_sources: none
---

# 無校正的 Raman／STIRAP 分類圖與摘要

## 結論

Raman GF 的 frequency、gain 和 frequency × gain，以及 STIRAP delay、amplitude、detuning，沒有可用讀出校正時照樣分析已存 g/e/f 計數。有合法完整的 saved cfg 才走這條路。合法缺 classifier、矩陣奇異、矩陣把有效點解成非有限值，都給相同的分類圖與具名摘要。只有後兩種情況警告；沒有新增 Writeback。

摘要只報已套用的校正。沒校正時三個矩陣診斷為 null。STIRAP 的 `f_population_estimate` 和對應座標仍只指校正峰；未校正時改報 `classified_f_peak`，包含 0–1 的 f label 比例與掃描座標。圖和摘要取同一個最大有限點，平手取首點。沒有有限峰就不報峰。

## 問題

Run 需要 classifier 才能分 g/e/f，但 Analyze 讀到的計數已經分好。四個入口沿用 Run 的查找方式，合法 cfg 缺 classifier 或矩陣奇異時都在出圖前中止，無法實現[無矩陣等價](2026-11-23-singular-matrix-as-absent.md)。

## 選項與取捨

- 峰值另具名，recipe 可以在無校正時找到下一次細掃的位置，又不會把分類比例讀成校正結果。共用原欄位再加 `basis` 會改變原欄位量義；只出圖不報峰會讓 recipe 再找一次。照舊拒絕會讓已有計數也不能分析。成功校正時不另報分類峰，本題不需要比較兩個峰。
- 未用的矩陣不出現在診斷。保留原矩陣並加狀態，或把 saved calibration 與 correction outcome 拆成兩個物件，能區分失敗原因，但沒有程式讀者需要。原矩陣仍在 saved cfg，異常原因在 log。
- Reader 判斷奇異，數值 owner 判斷反解結果；只在 reader 檢查會漏掉溢位，逐點遮罩則不等於無矩陣。不猜 Raman mode、不重分類、不以偽逆補校正。

## 留下的問題

Run 仍要求指定 ADC 的 centroid3。Analyze 不放寬錯型、錯 ADC、無效 cfg 或原資料有效性。若日後程式要自動區分未校準與校正失敗，再按[報告原則](../adr/0069-experiment-core-and-frontends.md#報告只寫本次做了什麼)加具名狀態。

## 依據

- 2026-11-23，使用者：「M1、邊界對，D」。M1 指只報已套用校正；D 指另報分類峰。原始問答未入庫。
- 遵循[無矩陣等價](2026-11-23-singular-matrix-as-absent.md)與 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#呈現規範) 的有限不截斷校正、可估 SEM 與圖上事實。縮窄 [ADR-0074](../adr/0074-flat-readout-module.md#classifier-隨-readout-保存) 的缺 classifier 拒絕範圍；Run 不變，這四個 Analyze 只讀 counts。
- 保留[Recipe 報告與執行進度](2026-11-20-recipe-report-polish.md)定下的 corrected f 峰量義，recipe 仍轉出分析 owner 的結果。
