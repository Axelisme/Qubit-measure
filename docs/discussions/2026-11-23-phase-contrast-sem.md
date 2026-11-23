---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#能推導的誤差棒都畫
  - 2026-11-23-phase-figures-physics-order.md
  - 2026-11-23-mist-trace-propagated-sem.md
basis:
  repo_commit: bb687cf5b3b37ee7379fb790f3d0e241bfeaa634
  external_sources: none
---

# CZ phase 的 contrast 標準誤

## 結論

Local Z、Conditional phase 及 heralded Conditional phase 的 Live 和 Analyze 都畫 contrast 的 ±1 shot SEM。Live 另開 contrast 圖，保留原本的相位圖。Conditional phase 的 contrast 圖只看最新有完整 X/Y 對的 train，不用等整個頻點的相位擬合完成。

## 問題

相位圖重排後，Analyze 的 X/Y 與相位已有棒，向量長度仍只顯示中心值，Live 則沒有這項診斷。兩個投影分開取數，既有計數足以估計各投影平均的抽樣誤差。

## 選項與比較

依 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#能推導的誤差棒都畫) 傳遞同批 shot 統計量。Contrast 非零時用徑向的一階傳遞，恰零時遵循 [MIST 的規則](2026-11-23-mist-trace-propagated-sem.md)，用變異最大的方向。這裡 X/Y 獨立，因此最大方向的 SEM 就是兩個投影中較大的 SEM。計算由分析層擁有，各實驗保留自己的圖與取數範圍。

## 未採用

不只在 Analyze 補棒，否則操作員無法在採集中判讀 contrast 的抽樣誤差。不調整相位 fit 權重或取數排程，因為這次估計只用來呈現。

## 留下的問題

一階近似在弱 contrast 可能偏小，棒不能當作顯著性判準。棒也不包含分類器或擬合參數的不確定度。

## 依據

[Phase 圖形重排](2026-11-23-phase-figures-physics-order.md#留下的問題) 將 contrast 的 shot SEM 留給後續改動。延後及此次實作範圍的原始裁決未入庫。
