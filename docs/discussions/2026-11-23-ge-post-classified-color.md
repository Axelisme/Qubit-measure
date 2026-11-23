---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-22-ge-post-preparation-majority.md
  - 2026-11-22-mux-fit-preparation-color.md
  - 2026-11-23-figure-fixed-geometry.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: d7911319985a4bfe17aec50b7ecb60953caa88b0
  external_sources: none
---

# GE Post 以分類結果上色

## 結論

使用者選擇在 Single 和 Mux GE 的 Post 圖，以分類結果上色。圖仍按 g／e 準備分格；藍、紅、灰表示 g、e、Other。Analyze Fit 和 Diagnostic 保留準備色。現行讀圖規則見[實驗呈現指南](../../zcu_lab/v2/figure-style.md#r4尺度跟著資料走)。

## 問題

Post 原本用圖例表示準備態，兩格各只用一種顏色。準備為 e 卻被分到 g 的點仍是紅色，必須對照分類線確認。

## 選項與取捨

保留準備分格，把同一格的各發按分類上色。準備態由標題識別；e 準備格的藍點指出分到 g 的位置。沿用 [GE 密度圖的六次方混色](2026-11-22-ge-diagnostic-layout.md)，避免以畫點順序決定重疊位置的顏色。這次遵循[Fit 的準備色先例](2026-11-22-mux-fit-preparation-color.md)和[固定存檔幾何](2026-11-23-figure-fixed-geometry.md)。

## 未採用

沿用[每格只畫準備色](2026-11-22-ge-post-preparation-majority.md)可把顏色與分類邊界分開，但讀者不能靠顏色找出誤分的位置。使用者本次取代該上色及圖例規則，保留分類線、兩個矩陣、條件數與各準備的觀測比例。

## 留下的問題

沒有待裁事項。離線 PNG 可驗讀圖，不驗 GUI 圖磚。

## 依據

2026-11-23，使用者說「0235對於analysis圖用準備色，對於post analysis用分類後色」，在分類色與維持準備色之間選「1a」。原始討論未入庫。
