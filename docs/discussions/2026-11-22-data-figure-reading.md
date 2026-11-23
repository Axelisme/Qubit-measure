---
date: 2026-11-22
status: superseded
superseded_by: 2026-11-23-figure-fixed-geometry.md
related:
  - ../adr/0067-gui-application.md#experiment-figure-area
  - ../adr/0069-experiment-core-and-frontends.md#圖形產物與呈現
  - ../../lib/zcu_tools/gui/app/measure/README.md#figure-presentation
  - ../../lib/zcu_tools/gui/app/measure/ui/README.md
basis:
  repo_commit: 72c25702f3966f779aac2d5d7c72dd518f62c88e
  gallery_precedent: 4912c7f146543c77fc274e9d4efcd90cccc22200
  in_page_selection_precedent: 5dbc9a9995139505aac50ae569d93fce1274ff87
  external_sources: none
---

# 在 Measure Data 頁讀原圖

## 結論

保留 Data 的縮圖、階段分組與分頁。操作員用滑鼠或鍵盤啟動 `View figure`，在同頁讀選中的原生 Figure。讀完回原頁次和按鈕。Designer 代裁採用這個入口，待使用者覆核。操作規則見 [Measure Qt views](../../lib/zcu_tools/gui/app/measure/ui/README.md)。

## 問題

使用者提醒，Save 頁可能較窄，不能假設視窗是 4:3 或一味縮小字。當時在實際 313×234 的 Qt 卡上，兩張 MIST 圖的文字都讀不清。操作員要在按 Save 前看懂刻度、單位、圖例與色棒，不是先存檔再核對。原始調查工作檔未入庫。

## 考慮過的選項與比較

既有 A gallery 保留多圖概覽。同頁選圖的先例讀的是已存 PNG；這次借用選圖手勢，不沿用其檔案或按視窗無下限縮圖的做法。按需在 Qt owner thread 重繪一張新 raster，空間不足就在閱讀面內捲動。Figure 和 canvas 留在原 owner，Save writer 與工件契約也不移動。

## 被否決的選項與理由

- 只提高縮圖 DPI 再縮回卡片，不會放大畫面上的字。把所有卡片放大，則會失去多圖概覽。
- 切回 Run／Analysis 或先看已存 PNG，不能在 Data 完成保存前核對。Run Save 也不產生 PNG。
- 借用原生 canvas 取代右側 gallery，單圖可用，多圖仍有窄兩欄的問題。也會把 Data 讀圖接到原 pane 的選取和互動狀態。

## 留下的問題

共用驗收用合成 adapter 走預設與窄兩種頁寬。另用既有 Lookback 結果和原分析選項，讀兩張真實 Analyze 圖的 I/Q 和 0.3944 µs，不重採集。共用框架通過不代表各實驗的個別圖已通過可讀性驗收。

## 依據

使用者於 2026-11-22 提出窄 Save 頁的排版要求。既有 gallery 與同頁選圖先例的 Git 版本列在 frontmatter。Figure owner 和回呼生命週期見 [ADR-0067](../adr/0067-gui-application.md#gui-callback-lifetime)；原生 Figure 與保存的分工見 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#圖形產物與呈現)。

## 後記，2026-11-23

[圖磚沿用存檔幾何](2026-11-23-figure-fixed-geometry.md)取代本文的捲動閱讀與當時的 R7 驗收方式。縮圖選圖、分頁和同頁 `View figure` 入口保留。
