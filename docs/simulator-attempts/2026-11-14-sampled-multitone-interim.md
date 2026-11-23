---
date: 2026-11-14
status: paused
commit: 7766f1cd5
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-14-sampled-multitone-interim.md
superseded_by: null
---

# 暫停 sampled multitone admission

**讀者：** framework-dev、agent。查 暫停 sampled multitone admission 的做法與證據。

## 核心思路

暫停兩個 multitone 快路徑的 sampled 列，保留原 fallback 及 RWA 的 constant pairs。

## 實測

原討論主要驗證拒收與 fallback，未給可追溯的加速數字。 見[原討論](../discussions/2026-11-14-sampled-multitone-interim.md#結論)。

## 為何採用或否決

先停用沒有合格證據的適用域。後續只恢復取得證據的 sampled RWA，不是同時放回兩個 family。

## 留下的線索

暫停保留原 family 身分與定價，量表須把未執行標成缺席，不能填零誤差。 續見 [sampled RWA 恢復准入](./2026-11-18-sampled-multitone-rwa-admission.md)。
