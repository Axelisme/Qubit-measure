---
date: 2026-11-19
status: adopted
commit: d7c523e2d
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-19-shared-multitone-pricing.md
superseded_by: null
---

# 成形雙 tone 共同估價

**讀者：** framework-dev、agent。查 成形雙 tone 共同估價 的做法與證據。

## 核心思路

以群總 CPU 秒數比較 gate 與 sampled RWA，建 gate 前按剩下的用量重算回本條件。

## 實測

原討論核對逐列選路，未量整次演化的加速。 見[原討論](../discussions/2026-11-19-shared-multitone-pricing.md#留下的問題)。

## 為何採用或否決

固定先試某個方案會在另一方案較便宜時多付一次探針或建構。分攤後重核可避免少數消費者攤不起 gate。

## 留下的線索

只在兩候選真的重疊時換成共同尺度，別把 CPU 秒和 GPU wall 直接相比。
