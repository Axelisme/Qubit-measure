---
date: 2026-10-16
status: adopted
commit: 11f28b048
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/experiment/ftf
  - lib/zcu_tools/simulate/adc
discussion: ../discussions/2026-10-16-ftf-simulation-speedup.md
superseded_by: null
---

# FTF 每點演化一次再抽樣

**讀者：** framework-dev、agent。查 FTF 每點演化一次再抽樣 的做法與證據。

## 核心思路

以 runtime 分派與快取取代離線表和重算，將物理演化與 shot 抽樣分開。

## 實測

舊路徑單個 128 ns twotone 點為 2.9 s（`e7f32506c`，單點原型的改前基準，不是完整 sweep 或落地版）。 見[原討論](../discussions/2026-10-16-ftf-simulation-speedup.md#問題)。

## 為何採用或否決

同一份演化結果不再隨 reps 和 rounds 重複求解；抽樣仍保留統計波動。

## 留下的線索

先確認時間花在重複的 physics、時間線建構還是單點求解，再選擇哪一層重用。
