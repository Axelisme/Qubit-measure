---
date: 2026-11-15
status: adopted
commit: 3d47858e7
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-15-t6-channel-removal.md
superseded_by: null
---

# t6 移除 reusable channel

**讀者：** framework-dev、agent。查 t6 移除 reusable channel 的做法與證據。

## 核心思路

退役代表工作負載未選用的 channel 分支，保留 t6 direct。

## 實測

原討論未釘工作負載比較的量測 commit，不轉錄耗時。 見[原討論](../discussions/2026-11-15-t6-channel-removal.md#問題)。

## 為何採用或否決

既有工作負載沒有證據支持保留兩套計算與定價機制，已由使用者裁定移除。

## 留下的線索

保留最後一道 family 與移除其內部做法不衝突。若出現新的高重用場景，再量 direct 是否仍合適。
