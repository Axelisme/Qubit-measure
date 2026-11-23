---
date: 2026-11-23
status: adopted
commit: 574e4ad8d
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-23-ftf-rwa-probe-memo.md
superseded_by: null
---

# FTF sampled RWA 探針快取

**讀者：** framework-dev、agent。查 FTF sampled RWA 探針快取 的做法與證據。

## 核心思路

一次 preparation 內重用完全相同的 sampled RWA 探針結果，每次 evolve 仍獨立選路。

## 實測

基準的 full-drive DOP853 探針用 175.29 CPU s（`ffd2a0c6`，離線 `two_qubit/chevron` 四點、單 worker 的一次 warm Run，呼叫執行緒 CPU 總和）。這不是快取後的耗時。 見[原討論](../discussions/2026-11-23-ftf-rwa-probe-memo.md#問題)。

## 為何採用或否決

快取可省重複的參考積分。原討論的 profile 只定位瓶頸，未提供落地後的加速倍率。

## 留下的線索

快取隨 preparation 退役。重用的是探針差值與通過結果，不是演化輸出或 capture。
