---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#平均-iq-的投影慣例
  - ../../zcu_lab/v2/figure-style.md#r12y-軸畫物理-observable
basis:
  repo: c15a15706e064009c9f620be92ac6f2948fe6806
  external: []
---

# 平均 IQ 圖的投影慣例

## 結論

[ADR-0069](../adr/0069-experiment-core-and-frontends.md#平均-iq-的投影慣例)記錄使用者裁決。同類實驗沿用同一 IQ 轉換。這次先改 mux Ramsey、Echo 與兩種 mux two-tone flux scan 的圖，不改取數或保存格式。

## 問題

Mux Ramsey 的擬合失敗圖直接畫平均 IQ 的幅值，卻沿用成功圖的實部投影軸標。Live 圖也畫幅值。使用者指出，幅值、相位或固定 I、Q 都可能對訊號變化的方向不敏感。沿等幅圓弧擺動的 IQ 可以有明顯條紋，幅值卻不變。

## 考慮過的選項與比較

只改失敗圖軸標能讓文字與曲線相符，但曲線仍可能看不出條紋。讓 Live 與失敗圖改用已在成功擬合及單 qubit Ramsey 使用的主軸投影，則可看到同類對比，不需重做擬合。

二維 flux／頻率圖不必為不同 bias 找一個固定投影方向。單 qubit two-tone flux scan 已先扣掉各 bias 的複數 IQ 背景，再以幅值顯示頻譜對比。兩種 mux two-tone flux scan 沿用這個做法；它們的選線流程原本就使用 phase-aware 背景。

## 被否決的選項與理由

不採只改軸標。那會修正文案，卻保留可能抹掉條紋的量。也不把 PCA 套到所有圖。相位本身是待測量時，改畫一個 IQ 投影會丟失讀者要看的相位。

## 留下的問題

同類其它實驗若仍畫固定分量，先檢查它是否有意顯示診斷分量，再決定是否另改。

## 依據

2026-11-23，使用者：「B，這種投影方式應該作為慣例存在，沒有獨特理由就保持一致」。原始問答未入庫。當時以[單 qubit Ramsey](../../zcu_lab/v2/twotone/time_domain/t2ramsey/core.py)及[單 qubit two-tone flux scan](../../zcu_lab/v2/twotone/flux_dep/core.py)為對照。
