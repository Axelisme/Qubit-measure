---
date: 2026-11-21
status: active
superseded_by: null
related:
  - 2026-11-21-chain-length-fixed-margin.md
  - 2026-11-21-readout-template-window-margin.md
  - ../adr/0073-measure-form-derivation.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../../zcu_lab/v2/twotone/ro_optimize/rpi_length/guide.md
basis:
  repo_commit: c45c98a2fb834eca834b213aff2ec4105b8c5d0c
  external_sources: none
---

# Rπ length 沿用讀出模組的長度差

## 結論

使用者在 2026-11-21 確認 Rπ length 與 chain length 的問題相同：「不过它看起来问题是一样的，应该要比照办理。」Rπ length 延用 [chain length 的長度規則](2026-11-21-chain-length-fixed-margin.md)。每點的 pulse 長度為 ADC window 加上可編輯的 `length_margin`。Open 或 Reset 從擷取模組帶入差值。沒有讀出模組才用範本。Cfg schema 從 3.0 升到 4.0，不轉換舊 cfg。

## 問題

Rπ length 原本在每個 ADC window 加 0.11 µs，和讀出模組原先的長度差無關。用含 ring-up 的讀出模組時，掃描會縮短原先為延後開窗保留的 pulse。

## 考慮過的選項

採用模組差值，讓掃描的 pulse 和 window 一起增減，同時保留操作員編輯差值的能力。這沿用 chain length 的共用初值，也遵循 [ADR-0073](../adr/0073-measure-form-derivation.md) 的固定來源與簡單推導。不使用 `ro_optimize/length` 的固定 RF 長度。後者用同一個 pulse 長度比較不同 ADC window，目的不同。

## 被否決的選項

保留 0.11 µs 或一律改用範本的 0.03 µs，兩者都會捨棄模組已有的差值。0.03 µs 只用於建立範本，以及沒有讀出模組可擷取的表單初值。

## 留下的問題

差值不會修好原模組的觸發時序。R→π 的等待仍從 RF pulse 結束起算，遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md) 的 timeline 規則，不新增等待 ADC 關窗的步驟。既有 Rπ sums 與分析契約不變。

## 依據

- [Rπ length guide](../../zcu_lab/v2/twotone/ro_optimize/rpi_length/guide.md) 記錄了表單、序列與保存契約。
- 使用者指示兩個 ADC window 掃描採同一長度差政策。原始任務工作檔未入庫。
