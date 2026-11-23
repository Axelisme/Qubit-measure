---
date: 2026-11-18
status: adopted
commit: c0555bd24
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-18-ftf-dynamic-dimension.md
superseded_by: null
---

# FTF acquire 動態維度

**讀者：** framework-dev、agent。查 FTF acquire 動態維度 的做法與證據。

## 核心思路

用最強一點做上限 pilot，只有預估節省足以抵驗算時才試小維度。

## 實測

原討論列出 main 與 integration 基準，但未將每輪正式數字釘到各自候選樹，不轉錄倍率。 見[原討論](../discussions/2026-11-18-ftf-dynamic-dimension.md#數字)。

## 為何採用或否決

使用 pilot 的上限解作該點輸出，避免為驗算不過的情況浪費整批計算。相同 preparation 可保留選到的維度。

## 留下的線索

適用於終點讀出且無早期 cavity drive 的 acquire。Pilot 點不是其他點的逐點證明。
