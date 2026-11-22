---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-ge-gate-split-from-diagnostic.md
  - ../adr/0072-module-boundaries.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: af1ebbd9fb6437e88e0c7a080eca7a07795ed951
  external_sources: none
---

# GE 診斷圖保留四格

## 結論

使用者選版面 A。診斷圖保留四格，左上用原始 IQ 並標擬合中心。左下保留合併的直方圖與模型，將 assignment 移到兩列之間。右上、右下各顯示一種準備的擬合成分。總標題保留同次擬合的 SNR、標準誤及中心警示。每個 IQ bin 取點數較多一組的顏色；直方圖柱子不畫邊框。

## 問題

[上一批](2026-11-22-ge-gate-split-from-diagnostic.md)把 assignment 放在左下縱軸標籤旁。將標籤改成 12 pt 仍在 360×375 px 的磚上碰字。逐點疊畫也讓後畫的準備態蓋住前一組，遮住了兩種準備的分布差異。

## 選項與取捨

A 為 assignment 留出列間空間，不用移除分解或縮短 `Fraction per bin`。各準備各自歸一後，柱高就是該準備落在一個 bin 的 shot 比例。依點數多者取色讓藍、紅兩團能分開；同數取中間色，總點數只改變不透明度。使用者看過 360×375 px 和 800×800 px 的參考圖後選 A，並說「左上应该不用转正，就原始的方向就好了，一样标出拟合中心。」

依 [ADR-0072 的實驗與 GUI 分工](../adr/0072-module-boundaries.md#實驗計算與-gui-接線)，兩個 core 各自決定圖名和何時畫圖，並選用擬合 owner 的 renderer。擬合 owner 只讀同次的 covariance，不把它加入分析或保存欄位。兩個 core 與兩個 adapter 共用 lab 的中心判讀，adapter 仍用既有規則決定寫回初選。診斷圖不消費 writeback note，也不反向決定選取狀態。

## 未採用

沒有把分解拆成另一張圖。使用者說「保留4格，每一格其实都还蛮重要的」。把 assignment 放進 IQ 圖內則讓窄磚的文字太小。沒有採用依比例混色：兩種準備接近時，兩團都會偏紫，看不出哪組為主。使用者也要求移除黑色直方圖邊框，「這樣看不到後面的顏色」。

## 留下的問題

本次先恢復 `diagnostic` 和 `diagnostic_i`。使用者選擇的密度上色也適用於 GE 的 live、原 IQ fit、Post 散點；那些圖另行接線，保留現有方向。離線磚只能檢查版面，不能代替操作員在 Analysis 與 Data 的實際查看。

## 依據

版面與上色由使用者於 2026-11-22 裁定，參考 PNG 與原討論未入庫。操作者的讀圖入口為 [單 tone GE guide](../../zcu_lab/v2/singleshot/ge/guide.md) 和 [mux GE guide](../../zcu_lab/v2/mux/singleshot_ge/guide.md)。等比例與柱子的畫法記在 [圖形呈現指南](../../zcu_lab/v2/figure-style.md#r4尺度跟著資料走)。
