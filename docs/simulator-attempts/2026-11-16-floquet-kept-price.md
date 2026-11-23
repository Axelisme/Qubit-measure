---
date: 2026-11-16
status: adopted
commit: e00bd6ca7
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-16-floquet-kept-price.md
superseded_by: null
---

# Floquet 保留與價格重校

**讀者：** framework-dev、agent。查 Floquet 保留與價格重校 的做法與證據。

## 核心思路

保留有配對收益的 Floquet 路徑，重校與直接積分比較的成本。

## 實測

原討論未把保留研究或價格重校各自釘到量測 commit；不把 repo basis 當成兩批共同量測版本。 見[原討論](../discussions/2026-11-16-floquet-kept-price.md#依據)。

## 為何採用或否決

Committed 批次的局部加速支持保留，但不改 guard，也不據此固定三個 family 的順序。

## 留下的線索

保留研究與估價校準用不同輸入。選路應涵蓋短段拒收及建構成本，不拿其中一批代表另一批。
