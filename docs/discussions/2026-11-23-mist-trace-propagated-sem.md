---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#能推導的誤差棒都畫
  - 2026-11-23-mist-single-trace-observable.md
  - 2026-11-23-inverted-error-bars-and-figure-facts.md
  - ../../zcu_lab/v2/mist/power_dep/single_trace/guide.md
basis:
  repo_commit: bcf894607f02a0a500b490a7740db1769fe13979
  external_sources: none
---

# MIST 距離的一階誤差棒

## 結論

2026-11-23，使用者選同批 shot SEM。MIST single trace 改用同批的 `[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]` 計算平均 IQ，並把六矩另存為可選的 Native `iq_moments`。Labber 保留原單一 log。Live 與 Analyze 用同一公式，舊檔只有平均 IQ 時保留距離，不畫棒。

距離是 `d = |z − r|`，`z` 是該點平均 IQ，`r` 是參考。對非零距離，令 `u = (z − r)/d`，一階變異為 `σ_d² = uᵀ(Σ_z + Σ_r − 2Cov(z,r))u`。共變異是 shot 樣本共變異除以實收 shot 數。預設參考照原有掃描順序取開頭 5%，向下取整、至少一點。對參考集內的點，它和參考平均的共變異是該點共變異除以參考點數。明填 `g0` 時視為精確。恰零沒有徑向方向，取完整差值共變異最大特徵值的平方根。只有一個參考點時，該點與自己比較的差值恆為零。

## 問題

[前一次裁決](2026-11-23-mist-single-trace-observable.md)保留位移大小及複數參考，但不畫棒。取模後的距離不是逐發距離的平均，不能直接套逐發平均的 SEM。使用者指出，即使只是一階近似，也能在 Live 幫助判斷是否早停。

## 選項與比較

- A。用同批 shot 矩傳遞平均 IQ 的不確定度。含點值、參考及兩者的相關。保存原矩後，重載可用同一個參考和公式重算。使用者採用。
- B。以各輪的平均作樣本來估輪間 SEM。要保存每輪均值，而且至少兩輪才有棒。它衡量的是輪間變化，不是 A 的 shot SEM。

遵循[一維反演量的誤差棒](2026-11-23-inverted-error-bars-and-figure-facts.md)對一階傳遞的取捨，以及同批實收 shot 的 SEM 慣例。這次只推翻[前次裁決](2026-11-23-mist-single-trace-observable.md)的不畫棒、不改取數與保存；觀測量和參考保留。

## 未採用

不畫棒無法把這份不確定度交給操作員。輪間 SEM 不等於這次要畫的 shot SEM。改成有號投影或逐發距離的平均會改變觀測量，也沒有採用。

## 留下的問題

近零時一階估計可能偏小，不能把棒當顯著性判準。跨輪合併的矩含投影雜訊，也吸收各輪平均值的變化，不能分開估計或扣除輪間漂移。

## 依據

- 2026-11-23，使用者：「0257應該能誤差傳遞吧？能推倒為什麼不畫？errorbar對於liveplot判斷早停是很重要的指標」。對 A／B 答「0257a」，同意改用 summed acquisition。
- 當時依據為候選 `bcf894607`。[呈現規範](../adr/0069-experiment-core-and-frontends.md#能推導的誤差棒都畫)記共用規則，[MIST guide](../../zcu_lab/v2/mist/power_dep/single_trace/guide.md)記判讀限制。原始問答及離線驗收工件未入庫。
