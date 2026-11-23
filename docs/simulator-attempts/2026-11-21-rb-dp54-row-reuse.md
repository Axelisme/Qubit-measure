---
date: 2026-11-21
status: rejected
commit: 666eb48ac
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# Heralded RB 的 DP5(4) 列內重用與共同步長

**讀者：** framework-dev、agent。查哪些 DP5(4) 成本值得重用，及為什麼沒有把投影 histories 綁在同一步長。

## 核心思路

先拆出單列的 DP5(4) 開銷，找可逐位不變地重用的 RHS、Hamiltonian 和不變量。另用合成微基準試把多條 projection history 的交換子疊成一批，估共同步長能否省掉單列控制成本。每條 history 仍有自己的 generator。

## 實測

量測版本 `666eb48ac`。重建 heralded RB 的 reference 臂、seed 685721、depth 1、point 1、history `[1]`，只計被拒後走 CPU DP5(4) 的 400 ns 雙 tone 列。暖機後原始 apply 為 739.779 ms。扣除計時探針成本後，把所有已辨認的不變工作視為可移除，估計最多省 14.29%。這是從該列分項耗時算的樂觀估計，不是實作後的加速，也不涵蓋未辨認的重用。

同一 commit 上以 seed 685721 產生 24 組 24×24 複數 density 及各自的 Hermitian Hamiltonian。只計交換子，BLAS／Numba 單執行緒、非排他。暖機後 NumPy stacked 為 64.181 µs／批，Numba 編譯逐列為 63.354 µs／批。微基準不含 dissipator、包絡或步長控制。原型與原始工作檔未入庫。

## 為何採用或否決

沒有找到達 15% 研究門檻的列內重用。批量乘法比逐次 Python／NumPy 呼叫快，卻沒有優於較接近原路徑的編譯逐列。共同步長還可能讓部分 histories 多做 trial，沒有足夠收益證據，兩條路都停在診斷。

## 留下的線索

評估批次計算，先比已編譯逐列，而不是用 Python 呼叫開銷當基準。應分開量可完全重用的工作與必須重算的狀態相依項。後續的[縮短區間起始步長](./2026-11-22-dp54-interval-step-seed.md#核心思路)針對被拒的第一個 trial，不需要不同 histories 同步。
