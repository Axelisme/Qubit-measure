---
date: 2026-11-21
status: adopted
commit: 2930a184c
modules:
  - lib/zcu_tools/simulate/adc
  - lib/zcu_tools/simulators/ftf/environment
discussion: ../discussions/2026-11-21-mist-cavity-reuse.md
superseded_by: null
---

# MIST 腔場端點重用

**讀者：** framework-dev、agent。查 MIST 腔場端點重用 的做法與證據。

## 核心思路

擴充 ADC 的 span memo，讓未作用的 pulse 振幅不阻止端點場重用。

## 實測

候選完整 Run 為 163.3 s（`767c478b8`，排他 CPU 0–7、seed 123、101×101 點、1000 reps × 10 rounds 的 `mist/flux_dep`）。 見[原討論](../discussions/2026-11-21-mist-cavity-reuse.md#結果)。

## 為何採用或否決

只存二階矩無法省掉端點求解。落地後由同一 owner 儲存兩種產物，FTF 仍逐段選態。

## 留下的線索

不同版本的 Run 差不能全歸因於 memo。對 shaped 或無法證明相等的 span 保留原求解。
