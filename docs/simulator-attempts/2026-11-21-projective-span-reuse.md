---
date: 2026-11-21
status: adopted
commit: ed770fb56
modules:
  - lib/zcu_tools/simulate/adc
  - lib/zcu_tools/simulators/ftf/environment
discussion: ../discussions/2026-11-21-projective-span-reuse.md
superseded_by: null
---

# Projective 腔場 span 重用

**讀者：** framework-dev、agent。查 Projective 腔場 span 重用 的做法與證據。

## 核心思路

由 ADC 一次取二階矩與各態端點場，完全相同的 span 才重用。

## 實測

獨立端點解從 29,921 次降到 0（`afbed1182` → `aef142668`，seed 123、`chain_length` 2.2 µs 首點、Engine ordinal 0–4 前綴的計數）。 見[原討論](../discussions/2026-11-21-projective-span-reuse.md#結果)。

## 為何採用或否決

真實前綴的 density 相同，減少了重複的端點求解。比較在指定 evolve 返回後停止，沒有測整次 acquire。

## 留下的線索

用真實歷史分岔量命中率與 RSS，再選 memo 容量；不要沿用 MIST 的命中率。
