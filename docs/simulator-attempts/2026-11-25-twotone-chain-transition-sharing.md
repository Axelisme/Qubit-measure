---
date: 2026-11-25
status: rejected
commit: 13aa91a3c25169d7ed003e0b187e71c1cad5cf85
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/experiment
discussion: none
superseded_by: null
---

# FTF twotone chain 原生時序相同，轉移列仍不同

**讀者：** framework-dev、agent。查找 FTF twotone chain 為何不能按 program 時序共用轉移列。

## 核心思路

若同一 RF 點的非終端 projective cells 有相同的轉移列，就能按來源態共用求解結果。先將控制時序對到 program 原生時鐘，分辨排程差和 FTF 支撐區間差，再比較實際的 population matrices，並檢查 probe 相位轉換是否能重建完整 density。[既有共同相位重用](2026-11-24-projective-cell-phase-reuse.md)只接受單一 span 且沒有 qubit drive 的 cell。本次的 cells 含 qubit probe，各有三個 Engine spans，不符合該 guard。終端 cell 仍須保留完整 density，不列入共用候選。

## 實測

在 `13aa91a3c25169d7ed003e0b187e71c1cad5cf85` 跑兩次離線 `twotone/chain`，輸入為 297.742 MHz 單點、12 reps、1 round、每 dwell 48 讀。兩輪都用 core 3、單 worker 和單原生執行緒。首輪完整 Run 用 74.263 process CPU s，其中 46 個非終端 cells 的轉移求解用 42.459 CPU s。第二輪用同一輸入擷取實際轉移矩陣與各來源態的 density；下列比對和時序追查均用這一次擷取，未再跑 FTF。原始輸入清單、profiler 與矩陣擷取檔未入庫。

這次擷取使用的 [FTF `drive_track`](../../lib/zcu_tools/experiment/readout_plan.py) 將 drive 支撐區間的起訖各自對齊到 2 ns，讀出邊界則保留 program 時間。現行 planner 對 drive 與 readout 都保留編譯後的起訖，見 [ADR-0084](../adr/0084-simulation-integer-time-base.md#時間契約)。在這次擷取中，event 17 和 44 的相對 probe start 看似差 0.01116071428 ns；映回 [430.08 MHz tProc 模擬時鐘](../../tools/data/experiment_wall_time/board-profile.yaml)，兩者都在各自讀出開始後第 904 tick。按 tProc 與 generator 的原生時鐘整理全部非相位時序後，46 個非終端 cells 只有一個 program schedule key。只改 probe-start 欄位而保留 FTF 的 duration 和 span 邊界時，仍有 46 個 keys。

| 同一輪擷取的 46 個非終端 cells | 結果 |
| --- | ---: |
| 完整矩陣逐值相等，或最大元素差在 `1e-10` 內 | 0／1,035 對 |
| 只比兩邊都求解的同來源態列，差在 `1e-10` 內 | 0／18,626 對 |
| 完整矩陣的最大 population 元素差 | 0.3108105484 |
| 兩邊都求解的列，最大 population 元素差 | 0.0014767061 |
| 兩邊都求解的來源態，對齊 probe 相位後的最大 density 元素差 | 0.0236457636 |

完整矩陣的最大差來自 projection floor。Event 0 跳過來源態 18，留下 identity row；event 39 實際求解了該列。即使只比共同求解的來源態，仍有差。用 `phase_graph` 的 probe 電荷對 density 作對角共軛，不會改變 population，對齊後的完整 density 也不相等。這輪的 probe 求解走 `floquet_period`，`single_tone_rwa` guard 未放行。RWA 電荷不能把這批 full-drive Floquet 結果轉成精確共變的通道。

## 為何採用或否決

按首輪 CPU 分布作 Amdahl 外推，若 46 個非終端 cells 都能共用同一份解，整輪的理想上限是 2.13 至 2.29 倍。第二輪實際測得可精確共用的 cells 為 0／46。由此估算，本輸入的精確列快取整輪只有 1.00 倍，且未計查找成本；這不是裝上快取後的計時。只用同一個 program schedule key 不能證明 FTF 轉移列相等。

## 留下的線索

這次沒有固定 FTF support 再重跑，也沒有固定時序而只改 probe phase。因此無法分開計算支撐區間取整、相位和 projection floor 各自造成多少差異。後續可固定原生時鐘支撐區間，保留 full-drive Floquet 路徑，分別比對固定與改變 probe 相位的輸入，並記錄各 cell 實際求解哪些來源態。本次只量單一 RF 點。換 RF 點會改變 probe frequency，不能直接共用完整轉移列。Board profile 是模擬設定，沒有量實體 DAC onset。
