---
status: draft
---

# 模擬器獨立成 package

已核准、尚未排程的方向。現行規則在 [ADR-0072](../0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明)。

## 方向

2026-11-15，使用者說：「先告訴你，我預計將模擬器功能獨立出zcu_tools成獨立的package，後者依賴前者，目前作為封閉子模塊可以」。
拆分後，模擬器 package 不依賴 `zcu_tools`，`zcu_tools` 依賴模擬器。

## 已具備的前提

`simulate` 已不 import `zcu_tools` 的其他 package，由 `.importlinter` 的 `simulate-closed` 檢查。
Acquisition 邊界型別在 `simulate.boundary`，numba 快取根目錄由 `simulate` 自己解析。

## 尚未決定

- 新 package 的名稱、發佈方式與版本規則。
- `simulators`（FTF、single、six 的模型與 environment）跟著拆出、留在 `zcu_tools`，或另成 package。
- `simulate` 內與量測框架共用的部分（例如 `simulate.fluxonium` 的 prediction 工具被 analysis、gui 使用）是否全部跟著拆出。

開始拆分時，先把以上各項帶回使用者裁決。
