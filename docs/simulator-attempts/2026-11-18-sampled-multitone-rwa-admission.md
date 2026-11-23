---
date: 2026-11-18
status: adopted
commit: ea668f77c
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-18-sampled-multitone-rwa-admission.md
superseded_by: null
---

# Sampled multitone RWA 恢復准入

**讀者：** framework-dev、agent。查 Sampled multitone RWA 恢復准入 的做法與證據。

## 核心思路

用逐列條件與同形群抽查控制誤差，只有預估回本才付探針成本。

## 實測

正例 wall 中位加速 3.149 倍（`9a09218c` → `28afb820f`，CPU 7 workers、d=24 的 `simultaneous_drive_retained24` 24 點、三組 ABBA）。 見[原討論](../discussions/2026-11-18-sampled-multitone-rwa-admission.md#考慮過的選項與比較)。

## 為何採用或否決

採用較晚基準的速度門檻，並保留未過探針時的原路由。測得收益只屬於該候選和正例。

## 留下的線索

另驗拒收路徑的探針成本。RB 縮樣沒走此快路，不能作 RWA 加速證據。
