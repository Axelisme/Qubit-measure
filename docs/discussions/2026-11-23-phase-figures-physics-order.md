---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../../zcu_lab/v2/figure-style.md#r4尺度跟著資料走
  - ../../zcu_lab/v2/figure-style.md#r10圖放理論重點採集設定放摘要
  - ../../zcu_lab/v2/two_qubit/conditional_phase/guide.md
  - ../../zcu_lab/v2/two_qubit/conditional_phase/guide_heralded.md
  - ../../zcu_lab/v2/two_qubit/local_z/guide.md
basis:
  repo_commit: 84905b20391af1c65cee6db4b3a8242cc6248886
  external_sources: none
---

# Conditional phase 與 Local Z 的圖照推導順序

## 結論

Conditional phase 的 plain 與 heralded 主圖依次畫三件事：

1. 在一個實測頻率下，比較 spectator 準備為 g 或 e 時，target 在 X–Y 平面的 Bloch 向量。每個向量標示 CZ 次數 `n`。
2. 在同一頻率下，畫兩種準備的累積相位對 `n`。擬合直線的斜率是每個 CZ 的相位。
3. 畫兩條斜率之差 `θe − θg` 對頻率。以 0–360° 視窗把 180° 放在中央，標記 π 交點及現有寫回規則是否能產生提案。

Local Z 的主圖左格畫兩個 target 的 Bloch 向量隨 `n` 旋轉，單位圓供比較。右格把兩條累積相位曲線放在同一軸，標示斜率 ± fit SE 和 correction。Correction 是斜率的負值。

X/Y、contrast、reduced χ²、herald 分布和殘差各留在對應實驗的診斷圖。Herald 分布的色階採實測資料範圍。Conditional phase Live 只顯示已完整的頻點，可能批次出現。Local Z Live 隨完成的 `n` 更新累積相位。這次不改取數、擬合、交點判斷、寫回或存檔。

## 問題

使用者在 2026-11-23 表示看不懂 heralded conditional phase 的 Analyze 與 Live，也看不懂 Local Z 的圖。原 Analyze 把相位包在 ±180°，π 交點正落在接縫。另一張圖分開畫 g、e 的斜率，讀者得自己相減。Herald 熱圖固定用 0–1，Live 只畫原始 quadrature。Local Z 的 correction 則藏在各格標題。

Conditional phase 準備 spectator 為 g 或 e，以 Xπ/2 把 target 旋到赤道。施加 `n` 個 CZ 後，用獨立 shots 量 X 或 Y 投影。兩個投影各取 `σ = P(e) − P(g)`，方位角是 `atan2(σY, σX)`。固定相位偏移進入截距，g、e 兩條斜率的差才是每個 CZ 的條件相位。CZ 次數差值的最大公因數須為 1。只量 `n = 1, 3, 5` 會讓相差 180° 的斜率混疊。

Heralded 版在前置讀出後保留 gg 和 ee，減少熱布居混合。ee 組的觀測先對調 spectator 標籤並反轉 `σ`，再和 gg 組共同擬合斜率。兩組可以有各自的 contrast 與截距。前兩格以 gg 觀測與截距呈現向量和相位，第三格使用 joint 斜率之差。獨立 gg-only 結果放診斷圖。

Local Z 讓另一顆 qubit 維持在名目 g，兩顆輪流當 target。兩條累積相位的斜率因此分別對應兩顆的單 qubit 相位。

## 選項與比較

- A。保留原圖分工，只改相位視窗、直接畫兩條斜率之差，並把熱圖縮到資料範圍。交點會好找，但向量、累相與斜率仍在不同圖。
- B。把向量、累相與相位斜率按推導順序排在主圖，診斷另放。Local Z 沿用同樣的 Bloch 與累相元件，使兩顆的 correction 能在同一圖例讀到。

B 遵循[把理論關注量放在圖上](../../zcu_lab/v2/figure-style.md#r10圖放理論重點採集設定放摘要)和[色階隨資料走](../../zcu_lab/v2/figure-style.md#r4尺度跟著資料走)。Ding 博論是既有圖面範本；這裡取其呈現物理觀測量的做法，不複製其他實驗的座標或圖格。

## 未採用

使用者選 B，並說：「B吧，但你先介紹這個實驗的細節與原理好了，我有點忘了，還有local z的圖也是看不懂」。同日對 Local Z 併入答「採用建議」。A 未採用，因為它不能讓讀者從量測向量跟到相位差。

## 留下的問題

Conditional phase 仍以 CZ 次數為外圈，每個 train 掃整段頻率。最後一個 train 完成時，多個頻點可能一起具備全部投影。現行 Live 不支援依這條曲線在掃描中途提早停或擴窗。要做到這點須另改排程。Contrast 仍為 `hypot(X, Y)`，它的 shot SEM 另作一次改動。

## 依據

改版程式與 guide 的依據為本 repo `84905b203`。原先給使用者的物理解說沒有入庫；本篇保留影響圖面裁決的序列、相位與 herald 分組理由。
