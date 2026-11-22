---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-22-ge-diagnostic-layout.md
  - 2026-11-22-ge-post-preparation-majority.md
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0072-module-boundaries.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: 3ab0bd596a885fc335eacee5f887ad6a3630c87c
  external_sources: none
---

# Mux GE Fit 用準備色，圖例標出 probe 來源

## 結論

Designer 代裁（2026-11-21 授權），待使用者追認。Mux Analyze Fit 沿用同一 tone 的擬合準備順序，與 Diagnostic 一樣用藍表示 g 準備、紅表示 e 準備。Fit 的兩個色項各標出對應的 probe row。成功擬合仍標 threshold 和中心；有密度底圖的失敗圖仍標失敗原因。Live 保留 probe off 藍、probe on 紅。

## 問題

兩個持久化的取數列分別為 probe off 和 probe on。Live 只知道哪一列已取數。Analyze 才讀 **Initial State**，將原始兩列對應到 g／e 準備。當時 mux Fit 繼續用原始取數順序，Diagnostic 和 Post 已按準備解讀。選 `excited` 時，同一批 probe-off 點在 Fit 表示藍，到了 Diagnostic 和 Post 卻算 e 準備、表示紅。變的是顏色所指的身分，不是原資料或 I/Q 座標。

## 選項與取捨

採用準備色，並在 Fit 圖例加兩個短標。選 `excited` 時，圖例是 `g prep (probe on)`、`e prep (probe off)`；選 `ground` 時，off 對應 g、on 對應 e。操作員看 Fit 的圖例就能回到 Live 找同一取數列。準備標籤描述一整批 shots 的準備方式，不等於單次真實態或分類結果。這一改動不碰擬合、原始座標、圖名、持久化或寫回。

沿用[診斷圖的 γ=6 銳化混色](2026-11-22-ge-diagnostic-layout.md)和[分格 Post 的準備色](2026-11-22-ge-post-preparation-majority.md)。使用者說「颜色的标签是与初始化制备绑定的」。Live 沒有 Analyze 選項，不能把它的取數色標作準備色。這依循 [ADR-0069 的呈現規範](../adr/0069-experiment-core-and-frontends.md#呈現規範)：同一身分用同一端點色，從取數轉到準備解讀時，在圖上寫出對應。圖例仍由 mux core 組裝，不替通用 renderer 增加實驗語彙，遵循 [ADR-0072 的實驗與 GUI 分工](../adr/0072-module-boundaries.md#實驗計算與-gui-接線)。

## 未採用

- Fit 沿用 off／on 色。從 Live 到 Fit 可用顏色追取數列，但切到 Diagnostic 或 Post 時，同一組點會換色。先前只改 Post 的討論保留當時的 Fit 實作，沒有另立 Fit 必須沿用取數色的規則。
- Fit 只寫 g prep／e prep。分析圖之間的顏色一致，但操作者要從設定或 guide 回推 probe row。Fit 已拿到這個對應，兩個短標就能把它交代在圖上。

## 留下的問題

較長的圖例要在 12×9 in 存檔圖上確認沒有蓋住資料。窄磚驗整張等比嵌入，不用縮小圖磚判原生字級。當時尚未走查桌面畫面。

## 依據

原始調查、挑戰與裁決未入庫。基準 commit 的 [mux GE core](../../zcu_lab/v2/mux/singleshot_ge/core.py) 可查 Fit 與 Diagnostic 的輸入順序。操作見 [mux GE guide](../../zcu_lab/v2/mux/singleshot_ge/guide.md#read-the-plots-and-use-the-result)，顏色和圖磚規則見[圖形呈現指南](../../zcu_lab/v2/figure-style.md)。
