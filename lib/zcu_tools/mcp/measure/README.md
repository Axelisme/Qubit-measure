**Last updated:** 2026-07-11 (canonical cfg paths)

# `zcu_tools/mcp/measure/`

measure-gui 的 MCP entry，負責把 MCP tool call 轉接到 live measure-gui
RemoteControlAdapter。此 package 是 app-local policy 層，不是共用 transport。

## 邊界

Cfg agent直接複製`gui_tab_get_cfg`/`gui_editor_get_cfg`列出的canonical leaf path；不加入
`.sweep`或`.value` wrapper。Batch path diff是成功後final net結果，失敗後重新讀cfg reconcile。

- `server.py` 是 MCP bootstrap / aggregation facade：保留 standalone preflight、
  server instructions/config、session/bridge setup、guarded `send_gui_rpc`、
  compatibility exports、domain tool table aggregation、stdio loop hooks。
- `zcu_tools.mcp._standalone.bootstrap_standalone_server()` 是所有 standalone
  MCP entry server 共用的最小啟動 helper：在 entry import `zcu_tools.*` 前把
  repo `lib` 加進 `sys.path`，並用一致的 stderr + `SystemExit(1)` 做 dependency
  preflight。各 app server 只保留自己的 required modules 與錯誤訊息。
- `tool_context.py` 提供 override handlers 的 late-bound runtime context 與共用 helper。
  hand-written override handlers 透過 context provider 解析目前的
  `server.send_gui_rpc`；generated tools 仍由 `generate_tools(..., send_gui_rpc)`
  維持 import-time capture。
- `tools_*.py` 依 domain 共置 hand-written handler、override schema 與
  manual MCP tool schema；`server.py` 只聚合這些 manual tool tables。
- `exposure.py` 從 GUI wire contract 的 `METHOD_SPECS[*].mcp` 推導 generated /
  internal / override exposure plan，並在 assembly 時 fail-fast 檢查 generated
  tool collision、manual/generated collision、override tool 缺漏，以及
  internal/override method 誤用 `MethodSpec.tool_name`。
- `session.py` 擁有 measure-only policy state：diagnostics piggyback queue、
  optimistic-concurrency baseline、guarded send flow、operation handle capture，以及
  `gui_debug_operations` 使用的 latest-handle projection。
- `session_policy.py` 放不可變 policy table 與純 helper：version guard deps、
  read-reveal table、start-op semantic key mapping、stale key 語義化。
- arbitrary waveform tools 是 agent-friendly generated RPC aliases：
  `list_arb_waveform`、`get_arb_waveform_preview`、`set_arb_waveform`。
  它們操作 qubit-scoped `.npz` asset store；`set_arb_waveform` guard deps 是
  `arb_waveforms`，list/preview 只 reveal `arb_waveforms`，validation/collision/missing
  以 tool error 的 stable `reason` 回報。
- `gui_value_list` / `gui_value_read` 是 generated read-only RPC tools，對應
  `value.list` / `value.read`。它們是 resolve-once value source 逃生通道；
  因來源可能投影 context/device/predictor，不列入 read-reveal table。
- `gui_editor_set` / `gui_tab_set_cfg` 的 scalar `value` 可傳
  `{"__kind":"value_ref","key":"device.flux.value","type":"float"}`，其中
  `flux` 是具名 registered device。
  bridge 不解這個 tag；GUI 端 `CfgEditorSession` / `LiveModel` 立即解析成 direct
  scalar，失敗以 stable RPC/tool error 回報。
- `tab.load_data` / generated `gui_tab_load_data` 是同步 mutation：guard deps 是
  `tab:{tab_id}`、`tab:{tab_id}:result`、`tab:{tab_id}:analyze`、`context`；不依賴
  SoC、device、cfg 或 save path，且不進 operation-handle table。
- load failure 以 `precondition_failed` 搭配 stable `reason` 呈現：
  `invalid_data_file`（canonical/adapter 不相容）、`unsupported_load`、
  `data_file_read_failed`；agent 不需要 parse traceback 或 raw Python exception。
- 一般 generated / hand-written RPC 的 transport timeout 以 `MethodSpec.timeout_seconds`
  加少量 slack 為準；`operation.await` 與 `notify.await` 必須由 caller 明確傳入
  動態 timeout。GUI handler timeout 代表可預期的 bounded wait 結果，transport
  timeout 代表控制 socket 已失去可信度，MCP bridge 會關閉該 socket 並讓下一次 call
  重新連線。
- `McpBridge` 只屬於 `zcu_tools.mcp.core` 的 transport adapter；measure-gui policy
  不下放到 bridge。
- Wire method MCP exposure policy 由
  `zcu_tools.gui.app.main.services.remote.method_entries` 的 method entry 宣告：
  default `generated` 產生 1:1 RPC tool；`internal` 保留 wire method 供 bundle /
  lifecycle 內部使用；`override` 指向一個或多個 hand-written MCP tools。
  MCP-only lifecycle/bundle/debug tools 不硬塞進 method policy。

## 測試注意

Remote/MCP 測試會建立 loopback socket；受限 sandbox 可能需要 unsandboxed execution。
headless 測試環境通常需要 `QT_QPA_PLATFORM=offscreen`、
`QT_QPA_PLATFORMTHEME=`、`MPLBACKEND=Agg`。
