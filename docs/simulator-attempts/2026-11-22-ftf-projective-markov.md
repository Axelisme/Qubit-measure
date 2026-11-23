---
date: 2026-11-22
status: partial
commit: cc6b252c0
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulate/adc
discussion: ../discussions/2026-11-22-ftf-projective-markov.md
superseded_by: null
---

# FTF 真空續演化與 Markov chain

**讀者：** framework-dev、agent。查 FTF 真空續演化與 Markov chain 的做法與證據。

## 核心思路

投影後清零腔場，讓同點、同層、同 DC 態只續演化一次。

## 實測

獨立同模型參考通過 41／75 個 key（`e602f0dc6` 候選，兩個 floor，兩個縮樣讀出掃描及 heralded conditional phase 的首讀）。 見[原討論](../discussions/2026-11-22-ftf-projective-markov.md#量到什麼還缺什麼)。

## 為何採用或否決

使用者接受先交付。對照沒有補齊所有獨立參考，舊 history 路徑仍保留。

## 留下的線索

清場改物理模型，chain 改表示。分開比較兩步；raw 的逐發配對不能從 compact 邊際還原。
