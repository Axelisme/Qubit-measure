---
date: 2026-11-22
status: adopted
commit: e2cf9727f
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - zcu_lab/simulation_envs/FTF
discussion: ../discussions/2026-11-22-ftf-projection-weight-floor.md
superseded_by: null
---

# FTF 樣板提高 projective floor

**讀者：** framework-dev、agent。查 FTF 樣板提高 projective floor 的做法與證據。

## 核心思路

提高新工作副本的 floor，減少低機率來源態的續演化。

## 實測

最大完整 density 差 6.921e-6（`cc6b252c0`，FTF 六列含 `chain/amp_rabi` 原表單首點、heralded CP、chain power、chain length、舊 phase 與一個 heralded RB seed；同 seed、同一 Markov 模型的 floor `1e-5` 對 `0`）。 見[原討論](../discussions/2026-11-22-ftf-projection-weight-floor.md#考慮過的選項與比較)。

## 為何採用或否決

只改樣板，不覆寫既有設定。Markov 同模型對照補上了原始表單首點的驗收，不代表舊 history 模型已補測。

## 留下的線索

比較 floor 時固定模型。不能用 Markov 對照替代帶場 history 的參考解。
