# MCP recipes

Recipe 是串接 GUI 操作的 generator，量測與分析由它開啟的 adapter 執行。
Adapter 定義 cfg、表單與結果，recipe 選來源、設參數並決定 Run／Save／Analyze／writeback 順序。
新增流程從 [recipe 教學](../tutorial/04-recipe.md) 開始，先依 [實驗教學](../tutorial/03-experiment.md) 註冊 adapter。

## 找到流程與註冊清單

[__init__.py](__init__.py) 的 `RECIPES` 是有序清單，不靠檔案掃描。
各 recipe 模組的 `DEFINITION` 擁有 tool 名稱、description、input_schema、callback、adapter mapping 與摘要宣告。
檔名描述量測用途，找流程時從清單的 import 找 owning module，不從檔名推導 MCP tool 名稱。
`helpers.py` 與 `mux_helpers.py` 提供共用步驟，不是獨立 recipe。
Import 不執行量測。MCP 組合根把清單注入框架，新增後要重啟 MCP server。
`recipe_guide(recipe="...")` 透過 adapter mapping 讀 operator guide，不回傳 recipe 程式碼。
完整 API 見 [recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)，mux 操作與摘要見 [mux guide](mux-guide.md)。
