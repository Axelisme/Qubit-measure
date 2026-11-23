---
date: 2026-11-22
status: superseded
superseded_by: 2026-11-23-ge-post-classified-color.md
related:
  - 2026-11-22-ge-diagnostic-layout.md
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0072-module-boundaries.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: b76c2814aa104465467afe7a3850cf65e39daf70
  external_sources: none
---

# GE Post 的準備顏色與分類邊界

## 結論

Designer 裁定兩個 GE Post 各用一張原始 IQ 圖。每格按 g／e 準備點數取色，票數平手為紫色。以獨立的線和標記顯示分類邊界和中心。半徑重疊時保留裁切的圓弧和共用的中垂線段。觀測分類比例按準備分列，兩個矩陣與條件數保留。數值、寫回和具名圖產物不變。這是代裁，尚未經使用者覆核。

## 問題

舊 Post 分成兩張單準備 IQ 圖，點色代表分類結果。同一圖只有一種準備，無法實行[依準備點數取色](2026-11-22-ge-diagnostic-layout.md)並比較兩種準備。若直接共格改色，原分類色的含義也不能沿用。

## 選項與取捨

同格可用一個顏色讀局部的準備多數。若一格有 90 個 e 準備和 10 個 g 準備，該格為紅色；紅色不代表 90 個點被判為 e。要讀 g 準備被判成 e 的整體比例，看 g prep 那一列。等距或不在半徑內的 Other 仍由分類規則決定，與準備票數平手無關。

[四格先例](2026-11-22-ge-diagnostic-layout.md)的分解、直方圖和 assignment 屬於 Primary，這次沒有改它。依 [ADR-0069 的呈現規範](../adr/0069-experiment-core-and-frontends.md#呈現規範)分開準備顏色、分類線與報表。依 [ADR-0072 的核心與接線分工](../adr/0072-module-boundaries.md#實驗計算與-gui-接線)，兩個 core 各自排圖，共用 GE 的畫圖能力，分類和矩陣計算由既有數值工具負責。

## 未採用

維持兩張逐點分類圖，無法在同一格比較準備。兩張單色密度圖也不能同格比較，且會失去原有的逐點分類色。另加逐準備分類詳圖則增加需保存和驗收的具名圖，當時沒有確立逐點判讀需求。

## 留下的問題

共格圖不能逐點定位少數準備的分類結果。若桌面走查確立這項需求，再比較分類詳圖。實際 Post-analysis、Data 的 View figure 和原生尺寸匯出 PNG 留給使用者簽收。

## 依據

2026-11-22 的裁定與原始討論未入庫。操作者的讀圖入口為[單 tone GE guide](../../zcu_lab/v2/singleshot/ge/guide.md#how-to-read-the-results)與 [mux GE guide](../../zcu_lab/v2/mux/singleshot_ge/guide.md#read-the-plots-and-use-the-result)。舊圖與分類計算見上述基準 commit 的兩個 GE core 與 [singleshot 數值工具](../../lib/zcu_tools/analysis/singleshot.py)。

## 後記（2026-11-23）

使用者確認 Post 依原本配置分開兩種準備：「我认为 post 图片原本就是分开的，那就用分开的。」共格多數取色未採用。左上只畫 g 準備的藍色密度，左下只畫 e 準備的紅色密度；兩格都疊分類線，右欄保留兩個矩陣。原討論未入庫，[現行圖形規則](../../zcu_lab/v2/figure-style.md#r4尺度跟著資料走)說明顏色與分類的分工。
