---
date: 2026-11-14
status: adopted
commit: 5214e2275
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-14-t6-direct-strategy.md
superseded_by: 2026-11-15-t6-channel-removal.md
---

# t6 的 direct 與 channel 分派

**讀者：** framework-dev、agent。查 t6 的 direct 與 channel 分派 的做法與證據。

## 核心思路

用 acquire 的 key 重用次數比較 channel 建構攤提與旋轉座標 direct 積分。

## 實測

原討論的速度驗收早於最後一版 ownership 重構，沒有釘每側量測 hash，不轉錄。 見[原討論](../discussions/2026-11-14-t6-direct-strategy.md#耗時驗收)。

## 為何採用或否決

當時採用按完整 acquire 的 key 重用次數在 direct 與 channel 之間擇一。Direct 的局部容差隨每段預算調整。後來才[移除 channel 與定價分支](./2026-11-15-t6-channel-removal.md)，保留 direct。

## 留下的線索

完整 acquire 的重用與單列解的成本不同。以同 session 的配對判斷，不把跨時段基準當成前後對照。
