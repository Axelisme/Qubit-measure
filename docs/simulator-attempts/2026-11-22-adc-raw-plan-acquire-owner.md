---
date: 2026-11-22
status: adopted
commit: b6bdb8024
modules:
  - lib/zcu_tools/simulate/adc
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulators/single_fluxonium
discussion: ../discussions/2026-11-22-adc-raw-plan-acquire-owner.md
superseded_by: null
---

# ADC raw plan 在 acquire 內重用

**讀者：** framework-dev、agent。查 ADC raw plan 在 acquire 內重用 的做法與證據。

## 核心思路

把確定性的 ADC raw plan 綁到 acquire session，換批或結束才退役。

## 實測

Raw materialize 343.737 → 159.082 s（`e602f0dc6` → `a4cd9f1bd`，同 fixture 的完整 `chain_power`，非排他單 worker、各一個 Run 的 materialize 累計）。 見[原討論](../discussions/2026-11-22-adc-raw-plan-acquire-owner.md#留下的問題)。

## 為何採用或否決

重用計畫而不重用亂數結果。這個對照量的是 session owner 及退役接線完成的候選，不是單一建立 plan 的 commit。

## 留下的線索

改 prepared batch 應使 plan 失效；不把單 worker 的 Run 外推到 GUI 多 worker 路徑。
