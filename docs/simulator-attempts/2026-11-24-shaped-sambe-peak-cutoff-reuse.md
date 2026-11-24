---
date: 2026-11-24
status: adopted
commit: d1408edb4383cbab7a4aaeb80cdf49e9ccc98fcd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# Shaped Sambe 在一次 cutoff 決策內重用峰值 Fourier 陣列

**讀者：** framework-dev、agent。查閱兩個 heralded 實驗的配對結果，以及為何只重用 cutoff-3 陣列。

## 核心思路

[Shaped Sambe](../../lib/zcu_tools/simulate/evolution/kernels/shaped_sambe.py) 先在峰值振幅比較 cutoff 2 與 3。若誤差超過原門檻，才比較 3 與 4。如果兩次比較都在當前有號載波做精確建構，原路徑會算兩遍相同的 cutoff-3 Fourier 陣列。模型、drive port、峰值振幅上界和子步長都由同一個 spectral family 固定。重用這張陣列把該次決策的峰值特徵分解從四次減為三次。

這個暫存只留完整 key 精確相同的 cutoff-3 Fourier 陣列。一次 cutoff 決策結束便釋放，不跨列或跨 `Engine.evolve`。它不留特徵向量、其他載波的陣列或插值結果。既有 guard 的公式與門檻、carrier workspace 的 key 與插值都不變。

## 實測

### 基線定位 preparation 成本

在 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a`，用雙 qubit 普查 manifest `c2b6b9e0522e` 和合成 fixture 跑離線 FTF。設定是 seed 123、`time_scale=0`、CPU affinity 0–3、1 個 worker、1 條原生執行緒。各場各跑一次完整 Run。Phase 沿用 spectroscopy 的快取。

在 11 格的 `conditional_spectroscopy_heralded`，shaped Sambe 的 prepare 與 apply 各花 42.0102、15.9674 CPU s。在 33 格的 `conditional_phase_heralded`，兩項各花 77.4208、23.7164 CPU s。這批不是下方配對的改前數字。Routing 記錄的 cache hits 計的是用了既有 preparation 的邏輯列，不能用來算特徵分解的次數。

### 三頻點只計數

以同一基線 commit 和 fixture，只把 `conditional_spectroscopy_heralded` 的 11 個頻率點縮為 5784、5800、5816 MHz。衍生 manifest 是 `87919bc3c114`。在排他 CPU 0–7 上，以同 seed、`time_scale=0`、1 個 worker 和 1 條原生執行緒跑一次完整 Run。觀察器只計 preparation 中成功的 shaped Sambe 特徵分解，不計 fractional tail。計數 key 由能量、耦合、有號載波和 cutoff 決定。每個 WorkPlan 屬於一次 evolve，同一 key 出現在不同 WorkPlan 時分開計。

這次有 3,138 次求解，各 WorkPlan 內的不同 key 加總為 2,748 個。多做的 390 次占求解次數的 12.43%。計數未區分重算是否發生於同一次 cutoff 決策。390/3138 不是候選命中率，也不是 CPU 或 core 耗時比例。

### 完整 Run 的排他配對

比較基線 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a` 與候選 `d1408edb4383cbab7a4aaeb80cdf49e9ccc98fcd`。兩側用同一份雙 qubit 普查 manifest `c2b6b9e0522e` 和合成 fixture。FTF seed 123、`time_scale=0`、排他 CPU 0–7、1 個 worker、1 條原生執行緒，關閉 OpenCL，不掛計數器。每側從獨立的空快取啟動一個程序，按表列順序各跑一次完整 Run。Phase 是冷啟動首場。Spectroscopy 接在 phase 之後，已暖機。

Prepare 是該場所有完整 routing 記錄的 shaped Sambe prepare CPU 秒數總和。Core 是 compile、evolve、readout 與 materialize 的牆鐘總和，除以 Result 格數。

| 場景 | Result 格數 | Core s／格，前 → 後 | Shaped Sambe prepare CPU s，前 → 後 | 非 evolve core s／格，前 → 後 |
| --- | ---: | ---: | ---: | ---: |
| `conditional_phase_heralded`，5798–5802 MHz | 33 | 5.6793 → 5.5276 | 80.9368 → 76.8360 | 0.7616 → 0.7599 |
| `conditional_spectroscopy_heralded`，5784–5816 MHz | 11 | 9.6173 → 9.1434 | 41.0232 → 36.1349 | 0.5348 → 0.5338 |

六個結果陣列逐元素相同。Phase 比對 `gate_counts`、`frequencies` 和 `sums`；spectroscopy 比對 `dc_values`、`frequencies` 和 `sums`。六個陣列的最大絕對差都是 0，低於 `1e-2` 的對照門檻。兩側的 routing 計數也相同。Phase 有 3 筆完整記錄、52,800 個 shaped Sambe 列。Spectroscopy 有 1 筆完整記錄、14,410 個 shaped Sambe 列。這一組配對只和舊路徑比，未和獨立參考解比對。單次配對也不能估計長期加速。原始 fixture、計數、profile 和配對收據未入庫。

## 為何採用

採用這項窄重用。兩場的 prepare CPU 和每格 core 耗時都下降，結果陣列相同。

兩場仍未達 [ADR-0078 的速度目標](../adr/0078-evolution-module-boundary.md#模擬器速度目標) 0.1 core s／格。表中的非 evolve 耗時由同一批的 `(core_s - evolve_s) / point_count` 算出，不是另一次量測。如果其他階段耗時不變，即使把 evolve 降到零，候選的 phase 仍要 0.7599 core s／格，spectroscopy 仍要 0.5338。只優化這條演化路徑無法讓這兩場達標。

## 留下的線索

保留一次 cutoff 決策的暫存，不擴成跨列或跨 evolve 的 live cache。三頻點計數沒量到窄重用實際省下多少次求解，也沒量到保留整個 WorkPlan 的特徵向量是否划算。若要繼續追 0.1 core s／格，先拆開量這兩場的 compile、readout 和 materialize，再決定要減哪項成本。這次配對未量其他場景，也未外推至全尺寸掃描。
