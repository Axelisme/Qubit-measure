---
date: 2026-11-18
status: adopted
commit: 6bc6f98d1
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulators/single_fluxonium
discussion: ../discussions/2026-11-18-remove-private-replay.md
superseded_by: null
---

# 移除 private replay

**讀者：** framework-dev、agent。查 移除 private replay 的做法與證據。

## 核心思路

移除逐位重算及舊 archive 相容要求，讓求解器不必為除錯改變執行形狀。

## 實測

原討論沒有單獨量測移除 replay 的 runtime 收益。 見[原討論](../discussions/2026-11-18-remove-private-replay.md#問題)。

## 為何採用或否決

原要求一度使開發中的 pilot 提高啟用門檻，並限制如何沿用上限解。改存當次輸入、路由與結果資訊，不保留重算承諾。

## 留下的線索

新診斷先記當次判斷所需資料。不要為重算舊 archive 綁住求解器的批次形狀。
