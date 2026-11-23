---
date: 2026-11-18
status: adopted
commit: fe394793f
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-18-gate-direct-duration-price.md
superseded_by: null
---

# Full-drive gate 的 CPU 競品估價

**讀者：** framework-dev、agent。查 Full-drive gate 的 CPU 競品估價 的做法與證據。

## 核心思路

按 row duration 估算 CPU 競品，避免在不回本的列先建多 tone gate。

## 實測

原討論未釘訓練收據的量測 commit；估價重校不是端到端加速驗收。 見[原討論](../discussions/2026-11-18-gate-direct-duration-price.md#後記加入-drag-lab-訓練點)。

## 為何採用或否決

共同的 duration 估價替代無法照顧長短列的固定係數；DRAG lab 納入後重校端點。

## 留下的線索

原 flat-top lab 連段仍是留出場景。不要把訓練誤差當成留出場景也通過。
