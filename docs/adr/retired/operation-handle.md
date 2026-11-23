---
status: superseded
---

# Operation handle 舊模型

> 摘自 [[0002]] 的[〈2. Operation handle（歷史設計）〉](../0002-version-table-async-handle-off-main.md#2-operation-handle歷史設計)。以下保留歷史原文；現行規則見 [[0066]] 的[〈問題與決策〉](../0066-operation-lifecycle.md#問題與決策)。

## 決策

### 2. Operation handle（歷史設計）

**規則。** `OperationRunner` 分開建立用於追蹤的 handle 與可選的 exclusion lease。舊 handle 模型已由 [[0066]] 接替。

**例子。** GUI 的 SoC connect 可經 runner 同時使用 handle 與 lease。Wire `soc.connect` 走同步路徑。

**邊界。** Handle 不證明持有 lease。建立、settle、取消與關閉的分工以 [[0066]] 和 [session owner](../../../lib/zcu_tools/gui/session/README.md) 為準。

**依據。** 舊模型見 Git `dc74c1e08`，現行分工見 [[0066]]。舊模型的原始討論未入庫。
