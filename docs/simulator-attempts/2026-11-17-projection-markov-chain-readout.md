---
date: 2026-11-17
status: adopted
commit: 4710837ef
modules:
  - lib/zcu_tools/simulate/adc
  - lib/zcu_tools/simulators/single_fluxonium
  - lib/zcu_tools/simulators/six_fluxonium
discussion: ../discussions/2026-11-17-projection-markov-chain-readout.md
superseded_by: null
---

# Single／six projective Markov chain

**讀者：** framework-dev、agent。查 Single／six projective Markov chain 的做法與證據。

## 核心思路

以逐態轉移取代 history 展開，讓輸出不再隨 read depth 組合增長。

## 實測

Producer 70.0 s（`60cc37798`，單次 `ChainAmpRabiExp` 的 51 點、16 次 read event、21 態，8 GiB／無 swap；只包 producer，不含整次 acquire）。 見[原討論](../discussions/2026-11-17-projection-markov-chain-readout.md#依據)。

## 為何採用或否決

由 ADC 按轉移表抽樣，保留逐事件分佈且不枚舉所有 histories。低機率來源態可逐點近似為留在原態。

## 留下的線索

Producer、integrated、raw 分開量。不同維度的單次計時含快取差異，不能據此擬合縮放率。
