---
date: 2026-10-18
status: adopted
commit: 7b45bfca9
modules:
  - lib/zcu_tools/simulate/adc
  - lib/zcu_tools/simulators/ftf/environment
discussion: ../discussions/2026-10-18-readout-discrete-sums.md
superseded_by: null
---

# Flat integrated readout 離散和

**讀者：** framework-dev、agent。查 Flat integrated readout 離散和 的做法與證據。

## 核心思路

對 ADC 樣本的條件場直接求幾何級數，省掉產生整條 trace 後再加總。

## 實測

原型未入庫，原討論只確認 `b0cb487f3` 基準可取得，未確認原型 commit。不轉錄成對收益。 見[原討論](../discussions/2026-10-18-readout-discrete-sums.md#考慮過的選項與比較)。

## 為何採用或否決

Flat、均勻格點且系統可安全分解時採用。成形包絡、trace 或不可靠的 eigensystem 走原 field 路徑。

## 留下的線索

不要用連續積分替代離散 ADC 格點。保留 fallback 並量完整 acquire，避免只從 reducer 的收益推論總耗時。
