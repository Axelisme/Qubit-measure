---
date: 2026-11-23
status: rejected
commit: 8cd18afcd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 放寬 RB 跨 depth 的前綴候選

**讀者：** framework-dev、agent。查同 seed 的 RB 已共享哪些演化，以及為什麼沒有放寬候選。

## 核心思路

同一個 arm、同一 seed 的較深 Clifford 序列可包含較淺序列的前綴。原本的 `PrefixResolver.certify` 只從非零起點的單 tone 常數列找共用邊界。F 把所有非零起點的列都列為候選，仍要求前綴簽章相同。F-max 只留每群最後一個共同邊界，避免反覆從頭求前綴。

## 實測

在 `8cd18afcd` 比較原規則 K 與兩個未入庫原型。輸入是離線 plain RB 的 reference 與 CZ-interleaved 兩臂，depth `[0,12,24]`、序列 seed 701、模擬 seed 123、2 reps、1 round、CPU 單 worker、合成分類器。各規則各跑一次完整 `run_stage`，K 為 97.594 s、F 為 103.395 s、F-max 為 103.361 s，均為牆鐘。各輪的 FTF preparation cache 都從空開始，K 的 Numba cache 是冷的，F 與 F-max 是暖的。因此這不是同暖度的速度排名。

同輪收據中，共用只發生在 depth 12 與 24 之間。K 在 CZ-interleaved 臂已共用完整的 12-Clifford 前綴。兩臂合計，K 執行 19 次前綴求解，F 執行 174 次，F-max 只執行 2 次，但整輪仍沒有比 K 短。原始工作檔未入庫。

## 為何採用或否決

保留既有 `certify`。放寬候選增加了可認出的邊界，卻沒有在目標 workload 上顯示整輪收益。編譯端建前綴樹及逐 gate 重用 channel 也沒有足夠收益證據，未實作。

## 留下的線索

量候選規則時，兩側要用同暖度，並分開記首臂初始化與雙 tone 路由成本。Heralded RB 的診斷在 reference 臂完成前就逾時，不能用 plain RB 的共用範圍推定投影鏈也能共享。
