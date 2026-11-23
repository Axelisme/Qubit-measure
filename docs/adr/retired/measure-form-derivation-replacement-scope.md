---
status: superseded
---

# Measure 表單推導的局部取代範圍

> 摘自 [ADR-0073](../0073-measure-form-derivation.md#局部取代範圍)。以下保留歷史原文；現行規則見 [ADR-0079〈決策狀態與問題〉](../0079-form-field-transitions.md#決策狀態與問題)。

## 局部取代範圍

[ADR-0079](../0079-form-field-transitions.md#決策狀態與問題) 局部取代本篇關於普通切換補 fresh defaults、數值葉 command 改 parent，以及僅保存 input 並丟棄不相容葉的規則。欄位轉換與保存還原以該篇為準。[Form core](../../../lib/zcu_tools/gui/form/README.md) 提供 `select` 與 `restore_input`，Measure 的 tab 由 Form owner 還原已保存的原始輸入，見 [Measure owner](../../../lib/zcu_tools/gui/app/measure/README.md#sources-and-workflow)。其餘 owner 邊界不變。
