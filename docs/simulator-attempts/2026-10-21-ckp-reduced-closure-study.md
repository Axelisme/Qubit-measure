---
date: 2026-10-21
status: partial
commit: 14c13fbea
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulate/adc
discussion: ../discussions/2026-10-21-ckp-reduced-closure-study.md
superseded_by: null
---

# CKP reduced closure 取捨

**讀者：** framework-dev、agent。查 CKP reduced closure 取捨 的做法與證據。

## 核心思路

比較哪些腔場歷史可壓進有限維 closure，避免完整 Lindblad 的成本。

## 實測

Reduced closure 原型的確切量測 commit 未釘。此檔 commit 是仍有 CKP 精度缺口的半古典部分落地，不是 displaced-Fock 的量測版。 見[原討論](../discussions/2026-10-21-ckp-reduced-closure-study.md#考慮過的候選)。

## 為何採用或否決

Displaced-Fock 的局部精度可過但成本不合格，單 Gaussian 也無法表示部分 coherence。半古典先交付，替代路線另做 joint Fock。

## 留下的線索

局部 Pe 合格不足以認證整張 CKP map 的中心偏移。比較前要保留 flip 造成的非真空腔激發。 續見 [Joint Fock CKP](./2026-10-21-joint-fock-performance-study.md)。
