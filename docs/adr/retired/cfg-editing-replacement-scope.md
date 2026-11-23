---
status: superseded
---

# Cfg 編輯模型的局部取代範圍

> 摘自 [[0065]] 的[〈局部取代範圍〉](../0065-cfg-editing.md#局部取代範圍)。以下保留歷史原文；現行規則見 [[0079]] 的[〈決策狀態與問題〉](../0079-form-field-transitions.md#決策狀態與問題)。

## 局部取代範圍

**規則。** 欄位轉換與保存還原以 [ADR-0079](../0079-form-field-transitions.md#決策狀態與問題) 為準。

**例子。** 普通切換沒有 converter 時只改 selection，不補 fresh defaults。

**邊界。** 局部取代也涵蓋改動 parent 的數值葉 command，以及只保存 input、丟棄不相容葉的舊規則。其餘 owner 邊界不變。

**依據。** 取代項目與實作狀態由 [ADR-0079](../0079-form-field-transitions.md#決策狀態與問題) 記錄。
