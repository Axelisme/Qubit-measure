---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../../zcu_lab/v2/figure-style.md#r11一維資料的逐點誤差棒
  - 2026-11-23-inverted-error-bars-and-figure-facts.md
basis:
  repo_commit: 3beb30d615620410a42f14003b3f9e3ddf13737b
  external_sources: none
---

# 已存統計量的第一層誤差棒

## 結論

第一層只補用已存統計量就能算出的誤差棒。Reset bath length 的 Live 和 Analyze contrast、readout power 的 Live 和 Analyze R1 contrast 都加棒。Reset flux gain／length 只在 Analyze 的 ratio、residual 與選定點加棒。五個 chain 實驗只改 Analyze joint residual 圖。上方量測點的棒是 pooled SEM，不是各點獨立估計的 SEM；下方的 ±1 參考帶表示同一個 pooled SEM。

不改取數、存檔、擬合或 Writeback。舊檔沿用原讀取方式。統計量不足以算出某個有限點的 SEM 時，只省略該棒。

## 問題

盤點找出 15 組原本沒畫棒的圖，依取得必要統計量的成本分成四層：使用已存統計量、加存 reducer sums、改用 summed 取數、保留 fit covariance。第一層的資料已隨 Result 保存，重載後可用同一方式畫棒。

## 選項與比較

- 先做第一層。使用者選擇先用既有資料加棒，不把存檔或取數變更綁進同一批。Designer 按這個範圍決定三個家族的圖。
- 一次做四層。能涵蓋更多圖，但還要修改取數與存檔，所以沒有採用。
- 使用 Gaussian 或 Wishart 近似替 SNR、Fsep、QND 等協方差類指標畫棒。這需要先決定能否接受分布假設，所以沒有採用。

審查發現，跨輪累加的矩沒有保留各輪均值，不能把輪內抽樣的散布和輪間均值的變動分開。Designer 更正措辭。這些棒包含投影雜訊，但不另外估計或扣除輪間漂移。輪間平均值的變化會併入散布，所以不能據此估計漂移。這項更正不改算法。

## 留下的問題

第二至四層另案處理。若要估計逐點 chain covariance 或區分輪內和輪間散布，需先決定要保存的統計量。

## 依據

- 2026-11-23，使用者：「1 截斷保留 CPMG 對 metadata 調查開 誤差棒先做第一層」。Designer 代裁涵蓋的圖、用既有 CycleSums 畫 pooled SEM，以及合併矩的措辭。當時的盤點原始檔未入庫。
- 遵循 [ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)的 shot SEM 規則，以及[一維反演量的誤差棒](2026-11-23-inverted-error-bars-and-figure-facts.md)採用的一階傳遞。使用者同日指出誤差棒可用於判斷早停；這一批先改已有必要統計量且列入本次範圍的 Live 圖。
