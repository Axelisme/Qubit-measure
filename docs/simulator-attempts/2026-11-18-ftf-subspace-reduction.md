---
date: 2026-11-18
status: rejected
commit: c04fb3823
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: ../discussions/2026-11-18-ftf-subspace-reduction.md
superseded_by: null
---

# FTF 的 Engine 內子空間約化

**讀者：** framework-dev、agent。查 FTF 的 Engine 內子空間約化 的做法與證據。

## 核心思路

比較低能截斷、逐點選態與誤差界，尋找便宜又可靠的自動約化。

## 實測

在 `c04fb3823` 上的研究原型中，固定 DC，對 F1 單一常數 tone 做三組五點掃描。最低能量 24 態相對 32 態的完整 density 最大絕對元素差，15／15 點均 `≤1e-3`。這是該次截斷比較的門檻，不是整體精度目標 `2e-3`。見[原討論](../discussions/2026-11-18-ftf-subspace-reduction.md#單-tone-ftf-掃描)。

## 為何採用或否決

未找到便宜又嚴格的逐點 guard。雙 qubit 長序列也不適合依稀疏分量選態，因為主分量很快佔滿基底。

## 留下的線索

若由呼叫端選維度，須將 pilot 與驗算成本一起算。這與 Engine 內逐點保證是兩個不同目標。
