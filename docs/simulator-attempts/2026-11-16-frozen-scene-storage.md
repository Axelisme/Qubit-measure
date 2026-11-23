---
date: 2026-11-16
status: adopted
commit: f7c0cd082
modules:
  - lib/zcu_tools/simulate/evolution
  - tools
discussion: ../discussions/2026-11-16-frozen-scene-storage.md
superseded_by: null
---

# 凍結場景跨 worktree 共用

**讀者：** framework-dev、agent。查 凍結場景跨 worktree 共用 的做法與證據。

## 核心思路

把大份數值保存在 repo 共用快取，Git 只留可核對身份的小清單。

## 實測

舊 bundle 大小與 dirty-tree 計時未綁到確切量測 commit，不轉錄數字。 見[原討論](../discussions/2026-11-16-frozen-scene-storage.md#問題)。

## 為何採用或否決

同一份數值可跨工作樹重用，不讓大檔留在 Git 歷史。遷移沿用原 payload，避免重建模型和參考解。

## 留下的線索

缺檔由 producer 重建，hash 不同就停。這減少驗證資料的重算與儲存，不是 production evolve 加速。
