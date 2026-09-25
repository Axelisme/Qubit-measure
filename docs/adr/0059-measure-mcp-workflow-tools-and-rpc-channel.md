# ADR-0059：measure-gui MCP 以 workflow tools 加 live RPC channel 取代 1:1 generated tools

**狀態：** proposed（草稿，待使用者決定「待決問題」後改為 accepted）
**關聯：** 修訂 [[0014]] 決策 4 的工具生成方式；沿用 [[0002]]（version guard / operation handle）、[[0013]]（remote adapter 為第二個 View）、[[0035]] 與 [[0047]]（tool error 契約）。

## Context

measure-gui MCP 目前暴露 81 個 tool：50 個由 `METHOD_SPECS` 1:1 generated，31 個 hand-written（bundle、lifecycle、override）。另有 12 個 wire method 為 `internal`。這個形狀有三個問題：

1. **選擇負擔與雙路徑。** 同一件事常有 bundle 與細粒度兩條路，例如 `gui_tab_run` 與 `gui_tab_set_cfg` + `gui_tab_run_start`、`gui_tab_analyze_review` 與 `gui_tab_analyze_start` + `gui_tab_get_analyze_result` + `gui_tab_writeback_list`。agent 每一步都在 81 個名字中挑選，run-measure-gui skill 必須額外教「優先用哪條」。
2. **tool 面在 MCP 啟動時凍結。** MCP server 在自己的 process import `METHOD_SPECS` 並產生 tool，policy table（`GUARD_DEPS`、`READ_REVEALS`、`OPERATION_KEY_OF`）也是 MCP 端的 import。GUI 端新增或修改 wire method 後，即使 `gui_stop` + `gui_launch` 重啟 GUI，agent 仍看不到也呼叫不到新形狀；Claude Code session 內無法重啟 stdio MCP server。這讓 agent 無法以 e2e 方式驗證自己的改動。
3. **exposure 層在維護第二份 API。** generated tool 已靠 `tool_name` 別名改名（`context.ml_get` → `gui_context_ml_list`、`tab.writeback_preview` → `gui_tab_writeback_list`），MCP 名稱與 wire 名稱逐漸分岔。

外部參考：Blender MCP 只暴露少量觀察 tool、截圖與 `execute_blender_code` 逃生口；Anthropic〈Writing tools for agents〉建議合併成 workflow 形狀的 tool，不把 endpoint 1:1 暴露。measure-gui 驅動實體儀器，任意程式碼執行會繞過 permit、version guard 與 operation handle，因此只採用「少量 workflow tool + 通用通道」的形狀，不採用 code execution。

## Decision

### 1. 三層 MCP surface

| 層 | 內容 | schema 來源 | 會隨 GUI 重啟更新 |
| --- | --- | --- | --- |
| **Workflow tools** | primary flow、MCP-only 組合、硬體安全動作 | MCP 端 typed `inputSchema` | 否 |
| **RPC channel** | `gui_rpc_list`、`gui_rpc_describe`、`gui_rpc_call` | live GUI 的 `rpc.catalog` | 是 |
| **Dev** | `gui_debug_operations`（讀 MCP session 狀態） | MCP 端 | 否 |

一個 wire method 對 agent 恰好只有一個入口：綁定到某個 workflow tool、經 RPC channel 呼叫，或 `internal`（agent 不可達）。三者互斥，不存在雙路徑。

**Workflow tool 的收錄準則**，至少符合其一：

- 屬於 run-measure-gui skill 教的 primary flow；
- 是 MCP-only 組合（bundle、lifecycle、operation wait），wire 上沒有單一對應；
- 直接改變硬體狀態，typed schema 與說明文字本身是安全措施。

依此準則的初始清單（約 26 個，最終以待決問題 1 的使用數據校正）：

| 類別 | Tools |
| --- | --- |
| Lifecycle / orient | `gui_launch`、`gui_stop`、`gui_bridge_connect`、`gui_bridge_detach`、`gui_overview`、`gui_screenshot` |
| Setup | `gui_soc_connect`、`gui_project_apply`、`gui_context_create`、`gui_context_switch` |
| Tab workflow | `gui_tab_open`、`gui_tab_run`、`gui_tab_analyze_review`、`gui_tab_get_cfg`、`gui_tab_set_cfg`、`gui_tab_get_figure`、`gui_tab_writeback_apply`、`gui_tab_save_data`、`gui_tab_save_image` |
| Operation | `gui_op_wait`（吸收 `gui_op_poll`：`timeout=0` 即 poll） |
| Hardware | `gui_device_connect`、`gui_device_apply` |
| User | `gui_prompt_user` |
| RPC channel | `gui_rpc_list`、`gui_rpc_describe`、`gui_rpc_call` |

其餘能力（context ml/md、editor、predictor、arb waveform、value、device 查詢與取消、tab 細粒度讀寫、post-analyze 等）改經 RPC channel。原本只為包裝單一 wire method 而存在的 hand-written tool 移除；仍帶 MCP 端組合邏輯者（例如 `gui_context_ml_inspect` 的 open/read/discard）在遷移時逐一判定：升為 workflow tool，或把組合下放為 GUI 端 wire method。

### 2. `rpc.catalog`：catalog 由 live GUI 提供

GUI 新增 wire method `rpc.catalog`，回傳每個非 `internal` method 的 `method`、`description`、`params`（沿用 `schema_property` 的 JSON schema）、`timeout_seconds`、exposure，以及決策 3 的 guard policy。

- `gui_rpc_list(domain?)` 回傳 method 名稱與一行說明，以 `domain`（`tab`、`context`、`device`⋯）過濾，控制 context 用量。
- `gui_rpc_describe(method)` 回傳完整 params schema 與說明，承接原 generated tool description 的使用指引。
- `gui_rpc_call(method, params)` 以 catalog 驗證 method 存在與 exposure，再走既有 guarded `send_gui_rpc`；transport timeout 取 catalog 的 `timeout_seconds` 加 slack，`operation.await` 仍要求 caller 明確給 timeout。
- MCP 在每次 connect／launch 後重新取 catalog 並快取；呼叫到已不存在的 method 以 tool error `unknown_method` 回報。

參數驗證仍在 GUI 端 `validate_params` 完成（dispatch 前、handler 前），因此 RPC channel 在 MCP 端雖為弱型別，實際仍有 typed validation；錯誤沿 [[0035]]／[[0047]] 的 stable `reason` 回報。

### 3. guard policy 隨 method 宣告，由 catalog 攜帶

`GUARD_DEPS`、`READ_REVEALS`、`OPERATION_KEY_OF` 從 MCP 端 `session_policy.py` 移到 measure-gui 的 `RemoteMethodEntry`，以宣告式欄位表示：guard deps 與 reveals 維持現有 pattern 字串，operation key 由 lambda 改為同語法的 template（例如 `"device:{name}"`）。`MeasureMcpSession` 從 catalog 讀取這些欄位並以現有邏輯執行。

欄位放在 app-specific 的 `RemoteMethodEntry`，不放共用層 `MethodSpec`，維持 [[0014]]「version guard 只在 main、共用層零知識」的邊界。GUI 本來就擁有 version table 並檢查 `expected_versions`，由它宣告 method 依賴哪些 resource 讓 policy 與 handler 同處一地；guard 仍由 MCP session 組裝 `expected_versions`、由 GUI 判定。

### 4. exposure enum

`McpExposure` 改為：

- `TOOL`：由 spec 產生一個 workflow tool（取代原 `GENERATED` 中被收錄者）；
- `RPC`：預設，只經 RPC channel 可達；
- `OVERRIDE`：綁定 hand-written workflow tool，`gui_rpc_call` 以 `reason="use_tool"` 拒絕並指名 tool；
- `INTERNAL`：agent 不可達。

`tool_name` 別名只允許 `TOOL` 使用；RPC channel 一律以 wire 名稱定址，不再產生 MCP 專屬別名。

### 5. 不採用 code execution

不提供在 GUI process 執行任意 Python 的 tool。所有 agent 動作都經 wire method、permit、version guard 與 operation handle；這是 measure-gui 與 Blender 類 app 在安全模型上的根本差異。

## Consequences

- agent 預設面對約 26 個 tool，primary flow 沒有替代路徑；長尾能力經 `gui_rpc_list` 按需發現。
- GUI 端 wire method 的新增與修改，經 `gui_stop` + `gui_launch` 即可由 agent 以 `gui_rpc_call` e2e 驗證；只有 workflow tool 與 MCP 端程式碼的改動仍需重啟 MCP server。搭配 `gui_launch` 指定 worktree（另案）即可覆蓋 lane 開發。
- 長尾操作多一次 `gui_rpc_describe` 往返；MCP 端失去長尾 method 的 JSON schema 型別提示，錯誤延後到 GUI 端 `INVALID_PARAMS`。
- wire 新增 `rpc.catalog`，`WIRE_VERSION` 遞增。
- run-measure-gui skill 與 MCP server instructions 需依新 tool 面改寫；移除的 tool 名稱不保留相容別名。
- fluxdep／dispersive／autofluxdep 的唯讀 MCP 各約 5～10 個 tool，不在本 ADR 範圍。

## Alternatives considered

- **維持 81 個 tool，只靠 Claude Code deferred tool loading。** token 成本已低，但選擇負擔、雙路徑與凍結問題都不變。
- **可重載 stdio proxy（`dev_reload` + `tools/list_changed`）。** 能讓全部 tool 熱更新，但依賴 client 在 session 中途重抓 tool 清單，且在 MEASUREMENT 實際路徑上加入開發用機制。
- **Blender 式 code execution。** 繞過 permit／guard／operation handle，對實體儀器不可接受。
- **toolsets 分組（GitHub MCP 式）。** 減少預設數量，但被關閉的 toolset 仍在 MCP 啟動時凍結，且 agent 無法在 session 中途切換。
- **guard policy 留在 MCP 端。** RPC channel 呼叫新 method 時 MCP 不知道其 guard deps，GUI 端新增的 mutating method 需重啟 MCP 才受 guard 保護，違背決策 2 的目的。

## 待決問題

1. **Workflow tool 清單以實際使用數據校正。** `logs/mcp/measure/*-calls.jsonl`（本機 call log，未進 repo）統計各 tool 的呼叫次數與序列，確認初始清單有無遺漏或多收。
2. **guard policy 移往 GUI 端是否接受。** 這改變 policy 的所在 process，但不改變 [[0014]] 的共用層邊界。
3. **移除 tool 的過渡方式。** 本 repo 預設不加相容邏輯；若有外部腳本或 skill 版本依賴舊名，需要一次性切換的時間點。
