# 0014 — GUI 三 app 共用「純傳輸層」：gui/remote 的 NdjsonRpcEndpoint + McpBridge，policy 留在各 app

**狀態：** accepted（**已實作**，Stage E）。承 [[0013]]——RemoteControlAdapter 是與 MainWindow 平級的第二個 View；本 ADR 把它**拆成兩半**：純傳輸（共用）+ dispatch policy/domain（各 app 自有），adapter 退化成「跑在共用 endpoint 之上的 router」。
**關聯：** [[0002]]（async emit / off-main 是 measure-gui-only policy 之一）、[[0002]]（version-table guard 同上）。EventBus 收斂見決策 4。

## 背景

measure-gui（main）的 remote 子系統先成形（[[0013]]）。隨後 fluxdep-gui / dispersive-gui 兩個只讀 tool_gui 各自「複製 gui 再寫領域」（見記憶 [[兩 GUI 搬進 app/]]），於是出現**三份近乎相同的傳輸碼**：

- GUI 側：socket 生命週期 + accept loop + NDJSON 分幀 + per-client writer/outbound queue + `wire.version`/`auth` 握手 + reply 編碼 + push fan-out + Qt 主執行緒 marshal。
- MCP 側：socket 連線 + reader thread + RID 路由 + pending map + pid 檔 + 啟動/停止 GUI 子進程 + stdio 協定 loop + 由 method-spec 表生 MCP 工具。

三份各自演化會漂移。但**不是每樣東西都該共用**：measure-gui 有 fluxdep/dispersive 沒有的 policy——version-table guard（[[0002]]）、operation handle 追蹤、診斷渠道（[[0013]] 決策 2）、off-main 阻塞 handler（[[0002]]）。把這些一起抽進共用層，會讓只讀 app 背上用不到的機制、也讓共用層沾上 measure-gui 的實驗語義。

## 決策

**共用層（`gui/remote`）只擁有「純傳輸」——bytes 怎麼進出 socket，不碰任何 request 的「意義」。每個 app 保留 dispatch policy + domain。measure-gui-only 的 guard/operation/diagnostic 留在 main，永遠不進共用層。** EventBus 機制同樣抽到 `gui/event_bus`，但 main 的 enum scheme 刻意不收斂。

### 1. 邊界：傳輸 vs policy/domain

| 關注點 | 歸屬 | 理由 |
|---|---|---|
| socket / 分幀 / 握手 / push fan-out（GUI 側） | `NdjsonRpcEndpoint`（共用） | 純機制，與 method 集無關 |
| socket / RID / 生命週期 / stdio loop / 工具生成（MCP 側） | `McpBridge` + module helpers（共用） | 同上 |
| 哪些 method 存在、validation、handler 跑在哪條執行緒 | 各 app 的 router | dispatch policy 是 app 私事 |
| domain handler、EventBus 訂閱/序列化、events.* | 各 app | 領域不共用（決策 D2） |
| **version guard / operation 追蹤 / 診斷渠道 / off-main** | **只在 main** | measure-gui-only 的實驗語義，共用層零知識 |

判準：**「換一個 app 是否還需要它」**。socket 框架換 app 不變→共用；version guard 只有 measure-gui 的並發語義要它→留 main。endpoint 在握手之後就把 parsed/authed 的 `Request` 交給注入的 router，到此為止——它對 method 語義零知識。

### 2. NdjsonRpcEndpoint + EndpointRouter seam

endpoint 持 socket 那一側的全部狀態（server thread / selector loop / socketpair-wake / per-client `ClientLink` 含 recv buffer + outbound queue + sole-writer thread + drop-budget backpressure），內建只懂 `wire.version` / `auth` 兩個握手（只需注入的版本常數 + token）。其餘交給 **`EndpointRouter` Protocol**（每 app 實作的接縫）：

- `route(link, request)`：處理一條 parsed/authed/非握手 request——app 決定 method 集、validation、handler 跑在哪（主執行緒 marshal 由 app 自理）。
- `on_client_open(link)` / `on_client_close(link, *, on_main_thread)`：掛/釋放 per-connection app 狀態。`on_client_close` 帶 `on_main_thread` kwarg：drop 在 IO 執行緒（False）、`stop()` 在 Qt 主執行緒（True）——endpoint 擁有並回報執行緒上下文，router 自決 marshal vs direct（例如 main 要回收 editor session 必須在主執行緒）。
- per-connection 的 app 語義狀態（event 訂閱集、editor session）掛在 `ClientLink.app_ctx`，endpoint 從不讀它。

每個 RemoteControlAdapter 現在**持有一個 `NdjsonRpcEndpoint`**、實作 `EndpointRouter`——[[0013]] 的「第二個 View」退化成「跑在共用 endpoint 上的 router」。

### 3. MainThreadDispatcher 是共用 marshal primitive，_dispatch_on_main 留各 app

`MainThreadDispatcher`（一個 `invoke` signal 接 `QueuedConnection` 的 QObject）是把任意 callable 排上 Qt 主執行緒的**唯一原子機制**——三 app 逐字相同，故共用。但**「怎麼 dispatch」（policy）不共用**：

- fluxdep/dispersive 的 `_dispatch_on_main` 是裸 marshal（無 guard）。
- main 的 `_dispatch_on_main` 在主執行緒 `_run` 內疊上 version guard（`_guard_versions`，放在 `_run` 內讓 compare-and-act 對主執行緒 atomic / 無 TOCTOU）、`off_main_thread` 阻塞分支、`_track_editor_lifecycle`。

marshal 的「機制」抽出去，marshal 的「policy」留下來——同 [[0013]] 「實際怎麼送達是每個 View 內部私事」的延伸。

### 4. McpBridge class + MCPBridgeConfig + on_event hook + 注入 send_fn

> **後續（2026-06-08，mcp/ 整併, c8eb1a03）**：`McpBridge` 與各 app 的 MCP server entry 已搬到 `zcu_tools/mcp/`（`McpBridge`→`mcp/core/bridge`、entry→`mcp/<app>/server.py`），`mcp` 成為 `gui.remote` 的**使用方**（wire primitives framing/errors/wire/param_spec/method_spec + rpc_endpoint 留 `gui/remote`）。`MCPBridgeConfig` 再拆出基底 `McpServerConfig`（無 launch 欄位，給 agent-memory 這類無子進程 server；GUI bridge 仍用 `MCPBridgeConfig`）。**本 ADR 的決策不變**（傳輸機制抽共用、policy 留各 app），僅位置/打包調整。

MCP 側對稱地抽成 **`McpBridge` class**——一個進程的 socket 狀態全是 instance attr（socket / reader thread / RID cond + pending map / 子進程 + pid 檔），**不是 module global**。它暴露 `send_rpc_raw`（低階收發，無 policy）+ `connect`/`disconnect`/`launch`/`stop` + `wire_version_note`。三個注入點讓 app 不必碰傳輸：

- `MCPBridgeConfig`：per-app 常數（name/prefix/port/版本/instructions/pid+log 檔/run-script）。
- `on_event` hook：reader 把 event-push 行交給它，沒給就丟棄（只讀 app 丟、measure-gui 接進診斷佇列）。
- `send_fn` 注入：`make_forwarder` 用它發 RPC——只讀 app 傳 `send_rpc_raw` 的薄 error-raising wrapper；measure-gui 傳它**自己的** guarded `send_gui_rpc`（疊上 [[0002]] guard）。

module helper（`coerce_arg`/`make_forwarder`/`generate_tools`/`assemble_tools`/`run_stdio_loop`）由 method-spec 表生工具面。結果：fluxdep/dispersive 的 `mcp/<app>/server.py` 是薄入口（config + bridge + 3 個 lifecycle 工具）；measure-gui 保留全部自有 policy——guard bridge、operation 追蹤、診斷佇列（接 `on_event`）、22 個 override 工具，**並用自己的 stdio loop**（measure-gui 的 reason-tag error 契約 + piggyback 格式，不是共用的 `run_stdio_loop`）。

measure-gui 的 MCP server 使用 app-local `MeasureMcpSession` 擁有 measure-only policy state：version reveal cache、診斷佇列、operation debug projection 與 guarded send flow。guard dependency / read reveal table 是 immutable policy module，由 session 持有並執行；diagnostics queue 是 session implementation detail；semantic-key → handle index 只作 `gui_debug_operations` 的 latest-handle-per-resource 投影，不是 wait/poll 主路徑。`McpBridge` 仍是純 transport adapter，不承擔 measure-gui policy。

### 5. EventBus payload-type-key 收斂，main enum scheme 刻意不收

`BaseEventBus` / `BasePayload`（`gui/event_bus`）以 **payload 型別**為 key：`subscribe(SomePayload, cb)` 讓 cb 的 payload 型別自動推導、payload 永遠配不錯 event（型別即 key）。fluxdep / dispersive / autofluxdep 三者收斂到此（autofluxdep 8 payload，dead `PROJECT_CHANGED` 順手移除）。

**main 刻意不收**：measure-gui 用 `@overload` keyed on event enum 的另一套 scheme。強收會把 measure-gui 既有 wire 序列化攪進共用層、收益不抵 churn——同 [[0013]] D2「domain per-app」基調。

## 拒絕的替代方案

- **把 version guard / operation / 診斷一起抽進共用層**：fluxdep/dispersive 用不到，會背上 measure-gui 實驗語義；共用層必須對 method 意義零知識。否決。
- **endpoint 直接懂 method 集（不要 router seam）**：method 集是 app 私事，endpoint 一旦懂就綁死一個 app。router Protocol 把「握手後」整段外推。
- **McpBridge socket 狀態用 module global**（沿用 [[0013]] 現狀）：阻擋多實例與測試注入。改為 instance attr。
- **`_dispatch_on_main` 也抽進共用層**：marshal 的 policy（guard/lifecycle/off-main）是 per-app 的，只有 primitive（MainThreadDispatcher）逐字相同。抽 primitive、留 policy。
- **main 也收斂到 BaseEventBus / 用共用 stdio loop**：churn 大於收益，且攪入 measure-gui 既有契約。domain per-app。

## 後果

- 三 app 共一套傳輸碼，漂移面從 3 收到 1；新 tool_gui 只寫 router + domain + 薄 mcp_server 入口。
- measure-gui 的 policy（guard/operation/diagnostic/off-main/editor 生命週期/自有 stdio loop）原封不動留在 main，共用層對它零知識——未來看到「為何 guard 不在共用層」，本 ADR 解釋邊界判準。
- `RemoteControlAdapter` 從「第二個 View 本體」退化為「持 endpoint 的 router」；[[0013]] 的 View 接口拆分（DiagnosticSink/RenderHost/RenderView）與診斷 fan-out 不變，只是 push 改走 `endpoint.broadcast`。

## 已知後續（Theme 1）

`McpBridge` 已有 transport seam：注入 `Transport` Protocol（真實實作 `SocketTransport`，測試以 `FakeTransport` 經 `set_transport` 注入），socket I/O 退到接縫之後。measure-gui server 的 module global（`_LAST_SEEN` / `_OP_BY_KEY` / diagnostics queue / bridge instance）不移到 bridge instance；它們收斂到 app-local `MeasureMcpSession`，讓共用 bridge 持續對 method 語義零知識，同時退役 remote 測試裡的 module-global monkeypatch。
