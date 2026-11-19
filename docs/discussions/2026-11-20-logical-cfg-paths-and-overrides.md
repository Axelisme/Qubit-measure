---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-10-21-recipe-interface-redesign.md
basis:
  repo:
    main: 0d871cbd5
---

# Agent 只用 logical cfg 路徑；recipe overrides 支援 value、select、command

## 結論

- Agent 只看到 logical 路徑的 tab cfg 方法：`tab.read_recipe_field`、`tab.read_elements`、`tab.edit_recipe_fields`、`tab.select_recipe_option`、`tab.run_recipe_command`、`tab.check_recipe_ready`。`tab.read_elements` 的目標與回傳 row 也改用 logical 路徑。
- Canonical 路徑的 `tab.describe_cfg`、`tab.edit_cfg`、`tab.run_cfg_command` 從 agent 的 catalog 隱藏，GUI 內部照常 dispatch。`tab.get_cfg` 保留公開，是唯一回 canonical 路徑的方法，用來讀完整診斷。裁決記在 [ADR-0068〈Method 與一致性〉](../adr/0068-remote-transport.md#method-與一致性)。
- 每筆 recipe override 是 path 加上 `value`、`select`、`command` 其中一個。`select: null` 清空可清空的 selector；`command` 接受該欄位宣告的任何 command。`RecipeTab.apply_overrides` 依序套用，失敗時報出 `overrides[i]` 與 path，前面已套用的保留，不回復。
- `overrides` 移出 `RecipeInputs` 基底。lib 便利層提供 `Overrides` 欄位型別，42 個 recipe 各自宣告 `overrides: Overrides`。裁決記在 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 這次一併涵蓋「用 override 切換 waveform style」的需求：`{"path": "modules.cz", "select": "Inline: cosine"}`。

## 問題

一次 two-qubit CZ 量測任務之後的檢討裡，agent 在同一張 tab 上看到兩套路徑：

- canonical：model-field 路徑，含生成的 inline wrapper，例如 `tab.describe_cfg`、`tab.edit_cfg`；
- logical：省略 wrapper 的欄位路徑，例如 `tab.read_recipe_field`、`tab.edit_recipe_fields`。

兩套方法都列在 `rpc_list`，agent 會選錯，`tab.read_elements` 更是一半 canonical、一半 logical：目標路徑是 canonical，`fields` 是 logical。另外，recipe override 只能設值。換 waveform style 必須經過 select，因為 select 才會跑轉換器，所以 override 換不了 style。

使用者要求：「5b我想找一個使用者低負擔，實作也簡單，擴充性強的方案」。

## 考慮過的選項

| 選項 | 內容 | 規模（估計） |
| --- | --- | --- |
| X1 | agent 只看到 logical 路徑和四個動詞；canonical 的三個方法隱藏；`read_elements` 改 logical | 程式 30–70 行、測試 50–100 行 |
| X2（採用） | X1，加上 override 支援 select 與 command | 比 X1 多程式 65–230 行、測試 60–140 行；42 個 recipe 都要改 |
| Y | 只改文件，叫 agent 用 logical 動詞；canonical 方法留在 `rpc_list` | 文件 30–60 行 |

使用者：「我傾向X2，這樣行為才一致。」

X2 讓 override 和 tab 上的四個動詞對應，同一件事只有一種寫法。

**overrides 放哪裡。** 2026-10-21 的裁決是「overrides不該出現在base class中，超過負責的範疇了」，但當時的 `RecipeInputs` 基底實際上帶著 `overrides`。使用者先提議放寬：「我說overrides不該出現在base class中可以考慮放寬，作為中性屬性，作為慣例可以存在。」接著又問：「為什麼overrides不能作為慣例存在？不作為base class屬性，但幾乎每個recipe都會宣告他，這樣不就沒問題了嗎？」

比較過兩個寫法：

- 留在基底，當中性屬性：要放寬 2026-10-21 的裁決。
- 每個 recipe 自己宣告，lib 只提供 `Overrides` 型別與 `apply_overrides`（採用）：不必放寬裁決，作者也看得見這個欄位。

使用者以「好，我沒問題了，全部採用建議」採用後者，連同 X2 的行為與實作計畫。

### 先例與立場

- **ADR-0068 把 `tab.run_cfg_command` 列為公開方法，remote README 寫明 describe／edit 維持 canonical 契約：** 使用者推翻，改為 internal。ADR-0068 已有「`internal` 不列入 catalog、dispatch 保留」的機制，這次沿用。
- **ADR-0075「overrides不該出現在base class中」：** 遵循。這次把實際帶著 overrides 的基底改回符合裁決。
- **ADR-0079 與 form adapter 的 logical 路徑、selector、command 機制：** 遵循。X2 沒有新增 owner 層機制。
- **ADR-0065 與 ADR-0073 中 recipe 經 canonical 讀取的文字：** 區別。前者已經過時；後者的 recipe 段落是建議，不是裁決。

## 被否決的選項與理由

- **X1：** 換 style 仍要在 tab 上直接呼叫 `tab.select_recipe_option`，override 和 tab 動詞不一致。
- **Y：** 兩套寫法都還看得到，下一個 agent 仍可能選錯。
- **overrides 留在基底當中性屬性：** 需要放寬 2026-10-21 的裁決，而每個 recipe 自己宣告就能達到同樣效果。
- **失敗時整串回復：** 要新增 tab 的快照與還原，約 50–100 行。只有 value 的 override 原本就是每筆各一次 RPC，同樣沒有原子性，所以 X2 沒有讓情況變差。

## 留下的問題

- 順序由作者或 agent 負責：先改某個分支裡的欄位、後選那個分支，會失敗，或者值被轉換器丟掉。
- 實作前的盤點未入庫：當時 logical 讀取已涵蓋 `tab.describe_cfg` 的單層 metadata，主要只缺全表 diagnostics，而全表 diagnostics 由 `tab.get_cfg` 提供。

## 依據

- 程式：`lib/zcu_tools/mcp/measure/recipe_api.py`（`FieldEdit`、`Overrides`、`RecipeInputs`、`RecipeTab.apply_overrides`）、`lib/zcu_tools/gui/app/measure/remote/method_entries/tab.py`、`lib/zcu_tools/gui/form_adapter/_logical.py`。
- 文件：[measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)、[remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。
- 實作前的調查與實作驗證紀錄未入庫；必要事實已寫進本文。
