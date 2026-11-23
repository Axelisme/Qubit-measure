---
date: 2026-10-15
status: partial
commit: 70becd65c
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/experiment/ftf
discussion: ../discussions/2026-10-15-offline-propagator-library.md
superseded_by: 2026-10-16-ftf-simulation-speedup.md
---

# 離線 propagator library

**讀者：** framework-dev、agent。查 離線 propagator library 的做法與證據。

## 核心思路

把重複建表移到離線，執行期載入並插值。

## 實測

早期 CPU 原型及計時未入庫，無確切量測 commit。 見[原討論](../discussions/2026-10-15-offline-propagator-library.md#結論)。

## 為何採用或否決

此 commit 是後續入庫的較窄 FTF plateau library，並非原 CPU 原型或完整 GPU 建庫路線的驗收。後來改採 runtime 分派與快取並移除離線表（`c68e9401a`）。

## 留下的線索

離線攤提需要長期穩定的模型與輸入。若成本轉向每一場掃描的 runtime 判斷，重新比較何時建表。
