---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0073-measure-form-derivation.md
  - ../adr/0074-flat-readout-module.md
  - ../../zcu_lab/templates.py
  - ../../zcu_lab/v2/measure-authoring.md#readout-分析與-writeback
  - ../../zcu_lab/v2/onetone/README.md
basis:
  repo_commit: 3d73c92afee38e41c783b10777a7ca248344617c
  external_sources: none
---

# 讀出範本在 ADC 窗口前後各留 0.015 µs

## 結論

使用者在 2026-11-21 說：「我通常会前后各留零点零一五，总共多出零点零三的长度。我认为这作为预设模板的生成常数值是可接受的。」因此新建 `readout` 保留 1.0 µs 的 pulse，把 ADC window 縮至 0.97 µs。從「前後各留 0.015 µs」推得 trigger offset 為飛行時間加 0.015 µs。`readout_dpm` 也用這個 offset，並將 pulse 設為最佳化後的 ADC window 加 0.03 µs。

DPM 寫回沿用本次 Run 的 trigger offset，只改擬合的欄位與 pulse 長度。同一個 0.03 µs 長度差也用於普通 onetone 表單的 ADC window 初值。Onetone 的 ring-up trigger 規則不變。

## 問題

舊範本在 ADC 開窗前後各留 0.05 µs，共 0.1 µs。DPM 寫回另維護 0.1 µs，普通 onetone 表單用自己的 0.01 µs 差值。這些初值不能從同一處調整，也不符合使用者量測時保留的時間。

## 考慮過的選項

- 保留既有長度差可以沿用舊範本，但仍比使用者需要的 0.03 µs 多 0.07 µs。
- 只改新建範本會讓 DPM 寫回和普通 onetone 表單繼續使用不同長度差。
- 採用同一個長度差。Lab 的範本持有這個數字，表單和 DPM 寫回引用它，框架不定義 lab 的校準起點。

此裁決沿用 DPM 的「pulse = ADC window + 固定長度差」，只改長度差。它遵循 ADR-0073 的簡單初值與 ADR-0074 的獨立 trigger、ADC window 欄位。先前[讀出鏈時序討論](2026-11-21-chain-readout-spacing.md)要求 pulse 不要剛好貼齊關窗；範本改在開窗前與關窗後各留少量時間。這裡不改讀出鏈每點的長度或間隔。

## 被否決的選項

不保留 0.1 µs 的舊初值，因為它超過使用者需要的餘量。也不由各呼叫者分別定義長度差，否則改了範本仍要逐處同步。

## 留下的問題

這是起始值，不會重設已保存的 module，也不保證有既有 ring-up 延遲的 module 仍在 ADC 窗口兩端對稱。改變 trigger 或 ADC window 後，操作者仍須核對 pulse 是否覆蓋整個窗口。

## 依據

- 使用者在 2026-11-21 說明常用的前後餘量。原始討論未入庫。
- [量測作者指南](../../zcu_lab/v2/measure-authoring.md#readout-分析與-writeback)記錄 DPM proposal 保留與改寫的欄位。
- [One-tone 初值](../../zcu_lab/v2/onetone/README.md)記錄普通 onetone 表單的 trigger、ADC window 與 waveform 初值。
