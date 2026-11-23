---
date: 2026-11-14
status: adopted
commit: 3af2e8cae
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-14-common-frame-step-pricing.md
superseded_by: null
---

# common_frame 步長成本歸 dispatch

**讀者：** framework-dev、agent。查 common_frame 步長成本歸 dispatch 的做法與證據。

## 核心思路

移除 kernel 的 2 ns 經濟性限制，由 dispatch 按步數和建構攤提選路。

## 實測

2 ns 步長下限移除（設定值）。精度對照未釘量測 commit，不轉錄誤差。 見[原討論](../discussions/2026-11-14-common-frame-step-pricing.md#後記)。

## 為何採用或否決

精度 guard 留在 kernel，是否回本由選路決定。Pinned family 因此仍可執行數值合格的短步長列。

## 留下的線索

以更正後的精度對照判讀，且別拿恢復適用域當成整批加速證據。
