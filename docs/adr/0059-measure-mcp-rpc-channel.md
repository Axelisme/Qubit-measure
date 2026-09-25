# ADR-0059：measure-gui MCP 的 RPC channel——由 live GUI 提供 method catalog

**狀態：** accepted（未實作）
**關聯：** [[0060]]（特化 tool）；[[0002]]（version guard / operation handle）、[[0013]]（remote adapter 為第二個 View）、[[0014]]（共用 transport 與 app policy 邊界）、[[0035]] 與 [[0047]]（tool error 契約）。

## Context

measure-gui 的常用操作由特化 tool 提供（[[0060]]）；其餘 wire method 使用頻率低但仍需可達。MCP server 隨 agent session 啟動一次、在 session 內不能重啟，GUI 則可以隨時重啟。若可呼叫的 method 清單與其 guard policy 由 MCP 端在啟動時決定，GUI 端新增或修改的 method 在 GUI 重啟後仍無法被 agent 使用。

## Decision

### 1. 三個通用 tool

| Tool | 行為 |
| --- | --- |
| `rpc_list(domain?)` | 列出所有非 `internal` 的 wire method 與一行說明；已綁定特化 tool 的 method 標示對應 tool。`domain`（例如 `tab`、`context`、`device`）用於過濾。 |
| `rpc_describe(method)` | 回傳該 method 的參數 schema 與完整說明。 |
| `rpc_call(method, params)` | 呼叫該 method，回傳其結果。 |

### 2. catalog 由 live GUI 提供

GUI 提供 wire method `rpc.catalog`，回傳每個非 `internal` method 的 `method`、`description`、`params`（由 `ParamSpec` 產生的 JSON schema）、`timeout_seconds`、exposure 與 guard policy。MCP 在每次連上 GUI 後重新讀取並快取 catalog；三個通用 tool 只依 catalog 運作，因此 GUI 端的 method 變更在 GUI 重啟後即可呼叫，不需重啟 MCP。

### 3. exposure 隨 method 宣告

每個 method 在 measure-gui 的 `RemoteMethodEntry` 宣告一種 exposure：

| exposure | 意義 |
| --- | --- |
| `rpc`（預設） | 經 `rpc_call` 呼叫。 |
| `tool(names…)` | 由列出的特化 tool 使用；`rpc_call` 以 `reason="use_tool"` 拒絕並指名 tool。 |
| `internal` | agent 不可達，不出現在 catalog。 |

每個 method 對 agent 只有一個入口。

### 4. guard policy 隨 method 宣告

version guard 相依（guard deps）、讀取後揭露的 resource（reveals）與 operation key 以宣告式欄位寫在 `RemoteMethodEntry`：guard deps 與 reveals 為 resource pattern 字串，operation key 為同語法的 template（例如 `"device:{name}"`）。catalog 攜帶這些欄位，MCP session 依此組裝 `expected_versions`、更新已觀察版本並記錄 operation handle；是否過期由 GUI 判定。

這些欄位屬於 measure-gui 的 app policy，放在 app-specific 的 `RemoteMethodEntry`，不放共用層的 `MethodSpec`（[[0014]]）。

### 5. 驗證與錯誤

- 參數由 GUI 端 `validate_params` 在 handler 前驗證；錯誤以 stable `reason` 回報（[[0035]]、[[0047]]）。
- 呼叫 catalog 中不存在的 method 回 `reason="unknown_method"`。
- transport timeout 取 catalog 的 `timeout_seconds` 加上固定 slack；需要呼叫者指定等待時間的 method（例如 `operation.await`）由參數給定。

### 6. 不提供程式碼執行

不提供在 GUI process 執行任意程式碼的 tool。所有 agent 動作都經 wire method，因而受 permit、version guard 與 operation handle 約束。

## Consequences

- 新增 wire method 時只需在 `RemoteMethodEntry` 宣告 exposure 與 guard policy；預設即可經 RPC 呼叫。
- MCP 端不再持有 method 清單與 guard policy 表；兩者的唯一來源是 GUI。
- `rpc.catalog` 為新增的 wire method，`WIRE_VERSION` 遞增。
- `rpc_call` 在 MCP 端不做參數型別檢查，型別錯誤於 GUI 端回報。
