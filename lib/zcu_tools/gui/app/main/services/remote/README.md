**Last updated:** 2026-06-17 (dev/debug tools `gui_debug_screenshot`(window+dialog)/`_versions`/`_operations` + 3-tier tool classification (recommended/on-demand/dev); MCP figure folds into run/analyze finished (the op's own plot), gui_tab_get_current_figure now rarely-needed; base-tools-pure + 4 explicit bundle tools `gui_run_stage1..4` — the base tools no longer fold (each returns only its own op's result); the bundles fold the next decision's input and are the recommended flow. Earlier: `gui_run_stage` macro + first-use adapter-guide fold, set_field number-coercion fix, `gui_tab_new` folds editor_id+paths+cfg_summary, run/analyze finished replies fold a figure path, analyze summaries gain `*_err` + amp_rabi summary key→`pi_gain`. Prior: SoC connect is now a SYNCHRONOUS wire RPC `soc.connect` — the async `connect.start` + soc operation-handle is removed; make_soc_proxy fail-fast 1s COMMTIMEOUT; GUI async connect button untouched; WIRE 35, GUI 42, MCP 47)

# `zcu_tools/gui/services/remote/` — RemoteControlAdapter

> **MCP 搬遷（2026-06-08, c8eb1a03）**：本目錄的 `mcp_server.py`（MCP 端 entry）已搬到 `zcu_tools/mcp/measure/server.py`；共用 `McpBridge`→`zcu_tools/mcp/core/bridge`。**本目錄現只留 GUI-進程側 handler**（`service`/`dispatch`/`events`/`method_specs`/`wire_version`）——MCP entry 是它的使用方。`MCPBridgeConfig` 拆出基底 `McpServerConfig`（measure 仍用子類）。下文 `mcp_server`、`gui/remote/mcp_bridge.py` 等舊位置按此對映。

## Purpose

本地 NDJSON RPC 介面 + event push，讓 automation agent / 測試 script / 外部 orchestrator 透過 socket 驅動 GUI。所有 GUI / Controller 操作都會被 marshal 回 Qt main thread。本 service 不直接碰 Qt widget；依賴既有 `Controller` façade 與 `EventBus`。

## Transport / Policy 分層（Stage E）

Transport 機制與 measure-gui 專屬 policy 分屬兩層：

- **`RemoteControlAdapter`（GUI 端）持一個共用 `NdjsonRpcEndpoint`**，自己是 **router**：實作 `EndpointRouter` seam（`route` / `on_client_open` / `on_client_close`），把 dispatch policy + EventBus 訂閱/序列化 + diagnostic / editor 變更流接上 endpoint 的 `broadcast`。endpoint 負責 socket、framing、writer、handshake、Qt marshal primitive；adapter 負責「呼叫誰、怎麼守、誰能訂」。
  - **router *scaffolding* 本身也已抽到共用層**：`gui/remote/control_service.py` 的 `RemoteControlServiceBase`（+`SubscriptionCtx`）擁有三 app 共通的機制——`route` 骨架（events.subscribe/unsubscribe/list + `_route_extra` hook + METHOD_REGISTRY 查找 + ParamSpec 驗證 + dispatch）、`_dispatch_on_main`（含 off-main 分支與兩個政策 seam `_guard`/`_after_success`）、EventBus subscribe/serialize/broadcast loop。`RemoteControlAdapter` **subclass 它**，domain 經 `__init__` 注入（method registry / event serializers + wire_event_name / 版本 / server_name），policy 經覆寫窄 seam 接上。**事件 key**：三 app 現**統一用 payload `type` 為 key**（#6 起 measure-gui 也退役自製 enum bus、改建在共用 `BaseEventBus`）；base 對 key 仍泛型不檢視（彈性保留，現只是 key 同型、`wire_event_name` 統一回 `payload_type.EVENT.value`，wire 名不變）。fluxdep/dispersive 是 read-only，**零覆寫**（連 `_get_bus` 都用 base 預設 `ctrl.bus`）；measure-gui 覆寫 `_get_bus`(`ctrl.get_bus()`)、`_new_client_ctx`(`_ClientCtx` 含 editor sets)、`_route_extra`(editor.*)、`_guard`(`_guard_versions`)、`_after_success`(`_track_editor_lifecycle`)、`_on_client_close_extra`(`_reclaim_editors`)、`_extra_start`/`_extra_stop`(editor change stream + diagnostic sink)。
- **`mcp_server`（MCP 端）持一個共用 `McpBridge`**（注入 `MCPBridgeConfig` + `on_event` hook），bridge 負責 socket / reader thread / RID routing / pid files；mcp_server 保留全部 measure-gui-only policy：version guard bridge、operation handle 簿記、diagnostic queue（接成 bridge 的 `on_event`）、override 工具、figure helpers。stdio loop **走共用 `run_stdio_loop`**：measure-gui 的 measure-only 行為（per-session logging、diagnostic piggyback、reason-tag error）經 `on_start` / `on_each_reply` / `on_error` / `server_version` hook 注入；read-only apps 不傳這些 hook，得到 bare loop。

兩個 adapter 都遵守 [[State main-thread invariant]]；version guard / off-main dispatch / editor lifecycle 等是 measure-gui 比 fluxdep/dispersive 多出的政策。

## Module Layout

```
services/remote/
├── __init__.py         — re-export ControlOptions / RemoteControlAdapter / RemoteError / DialogName
├── events.py           — EVENT_SERIALIZERS: payload type → wire payload（requery-hint schema；payload 定義在各 domain events module：`app/main/events/{tab,run}.py` + `session/events.py`）
├── dialogs.py          — DialogName enum + parse_dialog_name
├── method_specs.py     — Qt-free 線路 method 契約表（ParamSpec schema / timeout / desc）；mcp 端 import 它鑄 tool schema
├── path_resolver.py    — resolve_and_set（dotted path → live LiveField mutation）+ list_settable_paths（live 樹,帶 under 子樹過濾 + verbosity full/compact/paths）/ list_spec_paths（靜態 spec 樹,ModuleRef/WaveformRef 只列 .ref + allowed choices,不穿透 variant 內部欄位——哪個 variant 是 live 預設屬 value 層 make_default_value(ctx) 決定,static context-free 無從得知；要看 variant 欄位走 build tab + list_paths）
├── dispatch.py         — METHOD_REGISTRY: dotted-name → MethodSpec(handler, timeout, desc)
├── wire_version.py     — 本 app 自持的 WIRE_VERSION / GUI_VERSION + 各自 changelog（不在共用 wire 層）
└── service.py          — RemoteControlAdapter（measure-gui 的第二 View / driving adapter）；subclass 共用 `gui/remote/control_service.RemoteControlServiceBase`，本檔只留 `_ClientCtx` + policy seam 覆寫（guard / editor lifecycle / diagnostic / render_view）
```

NDJSON framing（`encode_line` / `decode_line` / `MAX_LINE_BYTES`）、`Request`/`Response` dataclass + typed coercion helper、`ErrorCode` enum + `RemoteError` 都在共用層 `gui/remote/`（`framing.py` / `wire.py` / `errors.py`）；stdio MCP bridge entry 在 `mcp/measure/server.py`（含本 app 的 guard / operation / diagnostic policy + 自有 stdio loop）。

**Transport 已抽到共用 `gui/remote` 層（Stage E）**：socket lifecycle / accept loop / NDJSON framing / per-client writer + outbound queue / `wire.version`+`auth` handshake / reply 編碼 / `broadcast(line, predicate)` push fan-out / `MainThreadDispatcher` Qt marshal primitive / `ClientLink` 都在 `gui/remote/rpc_endpoint.py` 的 `NdjsonRpcEndpoint`；**router scaffolding** 在 `gui/remote/control_service.py` 的 `RemoteControlServiceBase`；MCP-server-side socket（`send_rpc_raw` + connect/disconnect/launch/stop + reader thread / RID routing / pid files）在 `mcp/core/bridge.py` 的 `McpBridge`。**measure-gui 是三個 app 中最豐富的**：把 transport + router scaffolding 都委給共用層後，本目錄只保留 measure-gui 專屬的 dispatch policy 覆寫（guard / off-main / editor lifecycle / diagnostic / render_view）+ domain（dispatch / method_specs / events / dialogs / path_resolver）。

## Key Design Decisions

### Threading

- **Server thread**（daemon, `RemoteControlServer`）擁有 listen socket 與 `selectors.DefaultSelector`。Accept、讀 NDJSON、parse、dispatch 都在這條 thread。
- **Per-client writer thread**（daemon, one per connection）是該 client socket 的**唯一**寫者；從 `queue.Queue`（cap 256）取 pre-encoded lines。Reply 與 event push 都走同一條 queue，自然依 enqueue 順序序列化。
- Marshal 到 main thread 走內建 `MainThreadDispatcher(QObject)`：service 用 `dispatcher.invoke.emit(closure)`，Qt queued connection 保證 closure 在 main thread 跑（`QTimer.singleShot` 從非 main thread 不可靠）。
- 同步等結果用 `threading.Event.wait(timeout=spec.timeout_seconds)`；timeout / `RemoteError` / 任何 Controller exception 都轉成 typed error envelope。
- 每連線一次只有一個 in-flight RPC（單純化，不支援 pipelining）。
- **Outbound queue overflow**：滿 256 → drop + WARN log；連續 `_QUEUE_DROP_BUDGET`（8）次 drop → 主動 close client（避免 wedged reader 永久堆積）。
- **Shutdown 順序**（必須在 Qt main thread 呼叫，idempotent）：(1) unsubscribe EventBus；(2) enqueue sentinel 給每個 writer + join 2s；(3) self-pipe 喚醒 server thread + join 2s；(4) 關所有 socket。

### Event Push

- `service.start()` 在 main thread 對 `EVENT_SERIALIZERS` 每個 payload type 註冊一個 callback；callback 在 main thread serialize payload，對每個 subscribed client enqueue。wire name 來自 `payload_type.EVENT.value`，全集由 `test_remote_event_dialog_view.py` 的 wire-name lock 測試鎖定。
- Wire schema 採 **requery hint**：複合物件（MetaDict / ModuleLibrary / SocHandle / DeviceSetupSnapshot / BaseDeviceInfo / FluxoniumPredictor）**永不**上 wire，改送 `{"requery": ["<rpc method>"]}`，client 自行再查。
- Event requery hint 必須指向已註冊 RPC method，且 payload 或 requery method 必須提供足夠 scalar context 讓 client 可實際查到當前狀態。
- Serializer 回 `None` 表示「不 push 該事件」。callback 包 `try/except` 不 re-raise 進 EventBus emit。
- Per-client subscription 是 `_ClientCtx.subscribed: set[str]`（lowercase event name，天然 dedup）；`events.subscribe` 白名單檢查，未知事件 → `invalid_params`。`events.subscribe` 是純訂閱代理——但 **mcp 端不訂閱任何 event**（Phase 120c-2 起只收 diagnostic，見下方 Diagnostic Push），所以 RPC 端的訂閱機制現只供 GUI 內部 / 未來用。

### Diagnostic Push（ADR-0013，獨立於 EventBus / 訂閱）

- `RemoteControlAdapter` 是 ctrl 的 **DiagnosticSink**（`start()` 時 `add_diagnostic_sink(self)`）。ctrl 經 `notify_diagnostic(severity, title, message)` fan-out（severity ∈ error/info；info 無 title）→ adapter 編成 `{"event":"diagnostic","payload":{severity,title,message}}` **無條件**（不看 `subscribed`）push 給每個 client。
- **不經 EventBus**：診斷的產生是 ctrl→sink 直推（無 `bus.emit`），因為「報告故障的通道不該是故障系統」。wire 上與 event 同形態（`{"event":...}`），但 RPC 端不 gate。
- **mcp 端只收 diagnostic（Phase 120c-2）**：GUI 端 EventBus 註冊 + push 全保留，但 `_deliver_event` **丟棄所有非-diagnostic event**（agent 不曝露 resource-change event）；只 `event=="diagnostic"` 入 `_DIAGNOSTIC_QUEUE`，piggyback 在每個 tool reply。無 `gui_events_*` / `gui_editor_subscribe/unsubscribe` 工具。agent 偵測 async 完成走 `gui_*_poll`/`gui_*_wait`，偵測 resource 變動靠版本 guard 撞牆（帶語義 stale 清單）。
- **piggyback（mcp 端）**：任意 tool result 附第二個 content block，`_drain_pending()` drain 兩 queue（自上次 call 累積的 events+diagnostics）。agent 每次操作順路收 background 通知，不必單獨 poll；poll 仍保留給閒置等待。**transport 本來就 live-stream**（per-client writer thread），buffer 在 mcp 端不在 RPC 端，故 piggyback 純 mcp 端、不動 wire。

### Wire Format（NDJSON）

```
Request  → {"id":"<str>","method":"<dotted-name>","params":{...}}
Reply    ← {"id":"<str>","ok":true,"result":{...}}
         ← {"id":"<str>","ok":false,"error":{"code":"<code>","message":"..."}}
Push     ← {"event":"<name>","payload":{...}}        # 無 id / 無 ok
```

Client 用 `id` 區分 reply（有 id）vs push（有 event、無 id）。Error codes 是 closed enum：`unknown_method`、`invalid_params`、`controller_error`、`precondition_failed`、`timeout`、`unauthorized`、`busy`、`internal`、`shutting_down`。

### Version Handshake（三版本：一個比較 + 兩個顯示）

no-auth 的 `wire.version` 回 `{"wire_version": N, "gui_version": M}`(auth 閘門**之前**可探);mcp 端另持自己的 `MCP_VERSION`。三個都給 agent 看，但只有 wire 被**比較**：

- **`WIRE_VERSION`(=33,`wire_version.py`)**:mcp↔RPC **接口契約**(method 集合 / params / event 序列化)。mcp **pin 並比較**;不符 = 兩端講不同協議 → **硬 MISMATCH**。**只在契約變更時 bump**。本 app 自持(`wire_version.py`)而非共用 wire 層——每個 GUI app 各自演化其 wire 契約。
- **`GUI_VERSION`(=40,`wire_version.py`)**:GUI 進程**代碼版本**。**只顯示、不比較**(代碼版本是該進程自己的屬性,mcp 不該 pin)。bump 任何想能看出 reload 的 GUI 變更,**含不動 wire 的純內部變更**(拆分用意:內部變更 bump 它而非 WIRE_VERSION,契約版本不動)。
- **`MCP_VERSION`(=47,`mcp/measure/server.py`)**:mcp bridge 自己的**代碼版本**。只顯示;bump 想能看出 reconnect 是否載入了 bridge 端修改的變更(**含純 mcp-side convenience 工具**,如 batch fan-out 不動 wire 但 bump MCP_VERSION)。

mcp `_wire_version_note()` 只比 wire(硬 MISMATCH),gui/mcp 版本純顯示——一致→` wire v35 (mcp==gui); gui code v42, mcp code v47.`；pre-split GUI 缺 gui_version 顯示 `v?`。**用途**:agent 改完碼 reconnect/launch 後直接看三個數字對不對,自己判斷有沒有 reload，不靠 bridge 斷言（注意 ToolSearch 顯示的 schema 是 reconnect 當下快照，可能滯後；以 banner 版本 + 實際行為為準）。**mcp tool error 末尾附 `reason: <tag>`**（Phase 130，MCP 23）——precondition 失敗的 `GuiRpcError.reason`（no_run_result/no_project…，已在 wire envelope）透到 agent text，免 parse 句子分支。WIRE_VERSION 歷史（摘要；逐版完整以 `wire_version.py` 為準）:v33 SoC connect 改同步 wire RPC（`connect.start`+soc operation-handle → `soc.connect`，主線程 connect 後直接回 `{soc:{description,is_mock}}`；operation.await/poll 去 "soc" key；GUI 非同步 connect 按鈕〔`start_connect`+signals〕不動）、v32 加 predictor.set_model_params（從 typed EJ/EC/EL〔GHz〕+ flux_half/flux_period/flux_bias 憑空建+裝 predictor，走 in-memory install_predictor seam、無 params.json；MCP tool 自動生成；predictor.info reply 加 EJ/EC/EL 供讀回）、v31 operation.await cancelled→結構化回覆（cancelled 回 {reason:completed, status:cancelled, feedback?}，failed 仍 raise PRECONDITION）、v30 加 notify.open + notify.await（agent→user prompt，two-RPC：notify.open 主線程 mint token + 開 non-modal dialog、notify.await off-main 阻塞到 user Reply/Dismiss/QTimer；兩者皆 `_NON_GENERATED_METHODS`，agent 只見手寫 `gui_notify_user` 工具，見 ADR-0025）、v29 加 analyze.cancel（取消互動 analyze、清 is_analyzing 使 tab 可關；互動 analyze 與 run 分離故 run.cancel 不涵蓋）、v28 dialog.screenshot 參數 dialog_name→name（與 dialog.open/close 對齊）、v27 加 project.info（讀 ExpContext project 身分 chip/qub/res+result_dir/database_path，無 project fast-fail no_project）、v26 移除 device.active_setups（active_operations 為嚴格超集，含 setup entries）、v25 device active-op 列舉 singular→plural（active_setups/active_operations 各回全部 in-flight、+kind/device_name）、v24 figure/screenshot 收斂（移 view.screenshot、看圖只剩 tab.get_current_figure、加 save.post_image）、v23 complex writeback 標量無損往返（{"__complex__":[re,im]}）、v22 post-analysis RPC（post_analyze.start + get_post_analyze_params/result）、v21 tab.figure_screenshot→tab.get_current_figure(同 handler，當前顯示的圖)+save.{data,image,result} 回寫入路徑({data_path[,image_path]})取代 {ok}/{}、v20 progress 查詢統一 operation.progress(operation_id)、移除 run.progress/device.setup_progress；context.new 無 project→precondition_failed(no_project)、v19 context.new 去裸 value/unit 改 bind_device(白名單 unit、只讀 device 值)+clone_from label、v18 save.both→save.result+精簡回傳(context.new 回 label、save.* 回 {ok}、connect 只折 soc description、run-finished tab 折 {tab_id,interaction})、v17 去 session.persist/restore(persist 改 lifecycle 驅動)、v13 加 app.shutdown(優雅關閉 GUI,走正常 window-close,無 OS kill)、v12 加 adapter.guide(讀 adapter 行為導覽)、v11 加 soc.info(讀 soccfg)、v10 device setup↔run 對齊(device_setup_changed 拆 started/finished、+device.setup_progress)、v9 run.progress 加 raw n/total(progress SSOT 重構)、v8 cfg path 去 value 段+set_field 回 removed/added、v7 拆 run_lock_changed→run_started/run_finished、v6 去 cfg.set_field+diagnostic push(ADR-0013)、v5 device.setup_spec、v4 去 set_ml_* raw RPC(ADR-0011)、v3 去 device.set_value+operation_id、v2 ml.list_roles/rename。

**改 wire 層後的標準驗證流程**:改完碼 → `/mcp reconnect measure-gui`(重啟 MCP server)→ 重新 `gui_launch`(拿到新 GUI 子進程)→ 看回傳的 `wire vN (mcp==gui); gui code vX, mcp code vY` —— wire 不符=MISMATCH(契約)；gui/mcp code 數字對不對自己判斷有沒有 reload(改 GUI 內部記得 bump GUI_VERSION、改 bridge 記得 bump MCP_VERSION,才看得出 reload)。注意 reconnect 會連帶讓舊 GUI 子進程失聯但**不殺它**——舊進程仍佔著 port。**`gui_launch` 有 pre-flight：port 已被佔(舊 GUI 還在)直接 fail-fast**（MCP_VERSION 6 起），明說「port 已佔，先 gui_stop / kill 舊 run_measure_gui.py，或換 port」——避免舊版的「靜默連到殘留舊進程(舊碼、卻回新 pid)」陷阱（握手版本若同版看不出，只有實際效果能分辨）。launch 預設 port 8765 期望**空閒**（起新的）；`gui_connect` 預設 8765 期望**已有 GUI 在聽**（連現有的），無 GUI 時報 `No GUI is listening on 127.0.0.1:<port>`。兩者預設對稱、語義相反。

### Dialog API

- `DialogName` enum（`setup` / `device` / `predictor` / `inspect` / `startup`）是 wire-stable 識別。
- `MainWindow._open_dialogs: dict[DialogName, QDialog]` 是唯一 registry；`open_dialog` / `close_dialog` 是 UI click 與 remote 共用的單一進入點。所有 dialog 一律 **non-modal**（`open()` not `exec()`），否則 modal 會凍住 event loop 導致 marshal deadlock。
- Startup dialog 由 `app.py` 建立後以 `MainWindow.register_dialog(STARTUP, dlg)` 納入 registry。

### Event / Dialog / View 方法（Phase 81a）

| Wire method | 說明 |
| --- | --- |
| `events.subscribe` / `events.unsubscribe` | per-connection 訂閱集合增刪 |
| `events.list` | 列白名單事件 + 本連線已訂閱 |
| `dialog.open` / `dialog.close` / `dialog.list_open` | 開 / 關 / 列 dialog |
| `view.snapshot` | active tab、context label、status、open dialogs 的 JSON summary |

行長上限 `MAX_LINE_BYTES = 1 MiB`。`encode_line` 與 `decode_line` **都以 UTF-8 編碼後的位元組數**（不是字元數）做上限檢查，因此兩端對稱；多位元組字元（中文、例外訊息）不會繞過出站限制。超過 → encode 端 `INTERNAL` / decode 端 `INVALID_PARAMS`。Root 必須是 JSON object。`device.connect` / `device.disconnect` 的 ParamSpec **已宣告**（`type_name, name, address, remember` for connect；`name, remember` for disconnect）；`remember` 是 BOOLEAN optional default True，與 `optional_bool(params,"remember",True)` 對齊，符合 mcp 端「有才傳」行為，validate_params 現在在 handler 執行前強制型別。

### Security

- 預設 bind `127.0.0.1`（loopback）。`--control-allow-external` 才會 bind `0.0.0.0`，且必須有非空 token，否則 startup `RuntimeError`。
- Token check：第一條訊息必須是 `{"method":"auth","params":{"token":"..."}}`，用 `hmac.compare_digest`；未通過 auth 前任何其他 method 回 `unauthorized`。**例外:`wire.version` 是 no-auth 探針**(handshake 用,見 Wire Version Handshake),auth 前即可呼叫。Token 未設時跳過 auth（loopback 不強制）。
- 文件對使用者明示：「無 token 意味同機其他 process 都可控 GUI」。

### Method Registry

| Method | Controller call | Reply |
| --- | --- | --- |
| `tab.new` | `new_tab(adapter_name)` | `{tab_id}` |
| `tab.close` | `close_tab(tab_id)` | `{}` |
| `tab.set_active` | `set_active_tab(tab_id)` | `{}` |
| `tab.list` | iterate `list_tab_ids` | `{tabs:[{tab_id, adapter_name}]}` |
| `tab.snapshot` | `get_tab_snapshot` → 萃取 scalar；含 `editor_id`（tab 的共享 cfg-editor session,未 populate 為 null）| summary dict |
| `tab.get_cfg` | `get_tab_cfg_schema` → `schema_to_raw` | `{raw}` |
| `tab.update_cfg` | `raw_to_schema(base, raw)` → `update_tab_cfg` | `{}` 見下方 ⚠️ 語義 |
| `run.start` | `start_run(tab_id)` (RPC fire-and-forget；mcp `gui_run_start` 加 short-wait degrade) | `{operation_id}` |
| `run.cancel` | `cancel_run()` | `{ok:true}` |
| `analyze.cancel` | `cancel_analyze(tab_id)` | `{ok, cancelled}`（取消互動 analyze、清 is_analyzing 使 tab 可關；`ok` 恆 true=呼叫成功，`cancelled`=是否真取消到；無進行中互動 analyze → `{ok:true, cancelled:false}` graceful no-op）|
| `run.running_tab` | `get_running_tab_id()` | `{tab_id: str\|null}` |
| `save.data` / `save.image` / `save.result` | 對應 Controller method（回寫入路徑：data 是 `.hdf5`+suffix，同步可知）| `{data_path}` / `{image_path}` / `{data_path, image_path}` |
| `save.post_image` | `save_post_image(tab_id, image_path?)`（鏡像 `save.image`，存 post-analysis 圖，需 post 結果）| `{image_path}` |
| `context.use` / `context.labels` / `context.active` | 對應 | scalar |
| `context.new` | `new_context(bind_device?, clone_from?)`——**不收裸 value/unit**：`bind_device`（連線中的 flux device 名）由白名單（FakeDevice→none、YOKOGS200→A）決定 unit、**只讀** device 當前值命名 context（不 set），未在白名單 Fast-Fail；`clone_from` 是既有 context label（clone 其 ml/md）。對齊 notebook 與 setup dialog 流程 | `{label}` |
| `state.has_project` / `has_context` / `has_active_context` / `has_soc` | 對應 | `{value: bool}` |
| `soc.connect` | ParamSpec(kind, ip?, port?) → `coerce_connect_request` → `connect_sync`（**同步**：主線程 connect + 全部 post-connect 副作用〔State write、soc version bump、`SocChangedPayload`→FLUX-AWARE-MOCK provisioning〕完成後才回；remote 經 make_soc_proxy 1s COMMTIMEOUT fail-fast；連接失敗→controller_error）| `{soc:{description, is_mock}}` |
| `soc.info` | `get_soc_info()` → `describe_soc(soccfg)`（精簡 per-channel 表，非 QICK 原生長 description）+ `json.loads(dump_cfg())` + `is_mock`；未連線→precondition_failed | `{description, cfg, is_mock}` |
| `project.info` | `get_exp_context()` 的 project 身分；無 project→precondition_failed(no_project)。唯一暴露 project 身分的 wire query（mcp `_assemble_overview` 折入 `project={chip,qub,res}`） | `{chip_name, qub_name, res_name, result_dir, database_path}` |
| `startup.apply` | ParamSpec(chip/qub/res, result_dir?, database_path?) → `StartupProjectRequest` → `apply_startup_project`。**省略 result_dir/database_path → handler 用 `derive_project_paths(chip,qub,cwd)` 填預設 per-qubit 根**(`<cwd>/result\|Database/<chip>/<qub>`，= setup dialog 填名字時預填的)→ runnable，非 DRAFT。setup dialog(UI)保留 empty=DRAFT 互動路徑 | `{ok}` |
| `device.connect` / `disconnect` | 對應 DeviceService command（回 operation token） | `{operation_id}` |
| `device.reconnect` / `forget` | 對應 DeviceService command | `{}` |
| `device.setup` | 讀目前 `BaseDeviceInfo`，套 `with_updates(**updates)` 後 `start_setup_device`（含設輸出值 `updates={"value":..}`，無獨立 set_value） | `{operation_id}` |
| `device.setup_spec` | 從 live `info.model_fields` 導出 setup `updates` 可用欄位（name/type/choices(Literal→enum)/current/settable，protected type/address 標 settable=false）；device 須連線（類比 `adapter.cfg_spec`） | `{fields:[...]}` |
| `device.cancel_operation` | `cancel_device_operation(name)` | `{}` |
| `device.list` / `device.snapshot` | device summary；`snapshot` 含 `info`（`BaseDeviceInfo.to_dict()`，JSON-safe scalar，agent 可讀 live 參數值如 source `value`） | `{devices}` / `{snapshot}` |
| `device.active_operations` | `get_active_device_operations()` | `{active_operations: [{device_name, kind, name, type_name, address, status, error}, ...]}`（Phase C 並發：列**全部** in-flight op，按 device name 排序；`kind` = device_connect/disconnect/setup；進度走 operation.progress(operation_id)）|| `operation.await` | `await_operation(operation_id, timeout)` | `{status}`；blocking 等該 operation settle；failed/cancelled→PRECONDITION_FAILED（攜 `reason=outcome.status`，poll 端讀此分 cancelled/failed）、timeout→TIMEOUT。off_main_thread。mcp-internal（agent 走語義 wait 工具）|
| `operation.progress` | `get_operation_progress(operation_id)` → `ProgressService.bars_for_operation(id)` | `{active, bars:[{token,format,maximum,value,percent,n,total}]}`；run + device setup 通用（按 operation_id，取代舊 owner-keyed run.progress/device.setup_progress）。mcp-internal——`gui_*_poll` 在 running 時自動折進回傳，不曝露為 agent 工具 |
| ~~`cfg.set_field`~~ | **移除（ADR-0013 F11 / wire v6）**：tab cfg 編輯走該 tab 的 editor session — `editor.set_field(editor_id, …)`（editor_id 由 `tab.snapshot` 取），與 form attach 同一棵 model | — |
| `context.get_md` / `get_md_attr` / `get_ml` | scalar / name-only context query | keys / value / names |
| `ml.list_roles` | `get_role_catalog().list_meta()` | `{roles:[{role_id,label,item_kind}]}` |
| `ml.create_from_role` | `create_from_role(item_kind, role_id, name)` 一次性憑空建（撞名 fail-fast=PRECONDITION） | `{}` |
| `gui_launch` (MCP only) | subprocess `script/run_measure_gui.py --control-port` + wait for port。**跨平台**:用 `sys.executable`(=bridge 自己的 venv python，含 gui extra)而非硬編碼 `.venv/bin/python`；detach 用 POSIX `start_new_session` / Windows `CREATE_NEW_PROCESS_GROUP` | `pid` + ready msg |
| `gui_stop` (MCP only) | **先發 `app.shutdown` RPC 優雅關閉**(GUI 走正常 window-close: persist/devices/cleanup，無 OS kill，跨平台)→ 等 `timeout`(預設 10s)退出 → disconnect socket。超時且 `timeout_kill=false`(預設)回報仍在跑(可重試)；`timeout_kill=true` 才 force-kill(`proc.kill`/`os.kill`，跨平台)。`app.shutdown` 走 `RenderView.request_shutdown`(主線 `_perform_close`，跳過 device-setup modal) | closed/still-running/killed msg |
| `app.shutdown` | `RenderView.request_shutdown()`(主線馬上回，singleShot 延後關閉讓 RPC reply 先送出，再 `_perform_close`) | `{shutting_down:true}` |
| `tab.get_current_figure` | `take_figure_screenshot(tab_id)` → render `plot_stack.currentWidget().figure`（當前顯示的圖：run 2D map / analysis fit / post-analysis figure；post sub-tab 共用此 container）。**唯一看圖路徑**。PNG 用固定小尺寸（`figure_export.SCREENSHOT_FIGSIZE/SCREENSHOT_DPI`，省 token），render 暫存還原 figure 尺寸不永久改 | `{png_b64, bytes}` or `{saved_to, bytes}` |
| `predictor.load` | `load_predictor(LoadPredictorRequest)` | `{}` |
| `predictor.set_model_params` | `set_predictor_model_params(SetModelParamsRequest)`（從 typed EJ/EC/EL〔GHz〕+ flux_half/flux_period/flux_bias〔device units〕憑空建+裝 predictor，走 in-memory `install_predictor` seam，**無 params.json**；`flux_period==0`→PRECONDITION〔value↔flux affine 奇異〕。MCP tool `gui_predictor_set_model_params` 由 spec 自動生成）| `{}` |
| `predictor.clear` | `clear_predictor()` | `{}` |
| `predictor.predict` | `predict_freq(PredictFreqRequest)` | `{freq_mhz}` |
| `predictor.info` | `get_predictor_info()` | `{info: {path, flux_bias, flux_half, flux_period, EJ, EC, EL} \| null}` |
| `tab.get_analyze_result` | `get_tab_analyze_result(tab_id)` → `result.to_summary_dict()` | `{summary: dict \| null}` |
| `tab.get_analyze_params` | `get_tab_snapshot().analyze_params` → `dataclasses.asdict()` | `{analyze_params: dict \| null}` |
| `analyze.start` | `ctrl.analyze(tab_id, updated_params)` fire-and-forget；FIT→worker，**INTERACTIVE→主線程經 View 掛互動 picker、不起 worker**（Controller `_start_interactive_analyze`，token=analyze lease，Done 才 settle）| `{operation_id}` |
| `tab.get_post_analyze_result` | `get_post_analyze_result(tab_id)` → `result.to_summary_dict()` | `{summary: dict \| null}` |
| `tab.get_post_analyze_params` | `get_tab_snapshot().post_analyze_params` → `dataclasses.asdict()` | `{post_analyze_params: dict \| null}` |
| `post_analyze.start` | `ctrl.start_post_analyze(tab_id, updated_params)` fire-and-forget（第二層分析，**FIT-only worker**，無 INTERACTIVE）；gate 主 analyze result 不存在→PRECONDITION（`no_analyze_result`）| `{operation_id}` |
| `tab.get_cfg_summary` | `schema_to_raw()` → `_strip_cfg_tags()` | `{summary: clean dict}` |
| `save.set_paths` | `update_tab_save_paths(tab_id, data_path, image_path)` | `{}` |
| `save.data` / `save.image` / `save.result` | path optional — falls back to `get_tab_save_paths()` | 回寫入路徑（`{data_path}` / `{image_path}` / 兩者）；data 路徑 `.hdf5`+suffix 在 start 同步算好回傳，save 本身仍 async |
| `context.set_md_attr` / `del_md_attr` | 對應 ContextService MD mutation | `{}` |
| ~~`context.set_ml_module` / `set_ml_waveform`~~ | **移除（ADR-0011 / wire v4）**：ml entry 建/改一律走 editor session（`ml.create_from_role` + `editor.open(from_name)` → `editor.set_field`（含 EvalValue）→ `editor.commit`） | — |
| `context.del_ml_module` / `del_ml_waveform` | 對應 ContextService ML delete | `{}` |
| `context.rename_ml_module` / `rename_ml_waveform` | `(old, new)` → register(new)+delete(old)+emit ML_CHANGED 一次；clash/missing/empty→PRECONDITION。cfg 引用 old 經 ModuleRefLiveField self-heal 降級 inline Custom（不遷移名） | `{}` |
| `adapter.list` | `get_adapter_names()` | `{adapters}` |
| `adapter.cfg_spec` / `analyze_spec` | 靜態 cfg / analyze-params 規格（無 tab/context）；unknown adapter → invalid_params | `{paths}` / `{params}` |
| `adapter.guide` | `get_adapter_guide(name)` → `AdapterGuide.asdict`（五欄 behavior/expects_md/expects_ml/typical_writeback/recommended，**導覽散文非契約**，開跑前讀）；unknown → invalid_params。BaseAdapter 給「未撰寫」誠實預設 | `{guide}` |
| `editor.open` | `open_cfg_editor(item_kind, from_name)` — **modify-only**（編輯既有 entry）；憑空建走 `ml.create_from_role(role_id='<disc>:blank')` 再 open(from_name)。`discriminator` 已從 RPC 移除（內部 seam 保留） | `{editor_id, paths}` |
| `editor.set_field` | `cfg_editor_set_field(editor_id, path, value)`；tagged eval `{__kind:eval,expr}` 由 service decode 成 EvalValue。**run-guard**：若 session owner（`owner_of_editor`）是正在 run 的 tab → `precondition_failed`（與人靠 disabled form 同義，ADR-0013 F11） | `{paths(子樹), valid}` |
| `editor.get` | `cfg_editor_get(editor_id)` | `{paths}` |
| `editor.commit` | `commit_cfg_editor(editor_id, name)`；lowering（eval→concrete）後落地 ml | `{}` |
| `editor.discard` | `discard_cfg_editor(editor_id)` | `{}` |
| `editor.subscribe` / `editor.unsubscribe` | **sentinel**（state-owning,改 `_ClientCtx.subscribed_editors`,不經 dispatch）| `{subscribed_editors}` |
| `notify.open` | `open_notify_prompt(message, timeout)`（主線程 mint NotifyHandles token + 開 non-modal `NotifyUserDialog`，dialog 是 timeout SSOT）| `{token}` |
| `notify.await` | `await_notify(token, timeout)`（**off_main_thread**：阻塞 IO worker 等 NotifyChannel settle）| `{reason, reply?}`，`reason ∈ {reply, dismiss, timeout}`。兩者皆 `_NON_GENERATED_METHODS`——agent 只見手寫 `gui_notify_user`（mcp 端 serial compose 此 two-RPC，timeout/dismiss 不 raise），不曝露 raw notify.*（ADR-0025）|
| `auth` | token check（sentinel）| `{}` |

⚠️ **`tab.update_cfg` 語義（既非 full-replace 也非 merge-over-current）**：`raw_to_schema(base, raw)` 只取 `base` 的 **spec**，value 從 `make_default_value(spec)`（spec 預設）起、再以 raw 提供的 key **覆蓋**。所以**缺漏的 key 重置為 spec 預設值，不保留現值**。等同「以 spec 預設為底、raw 覆蓋」。要保留現值請先 `tab.get_cfg` 拿完整 raw、改動後整份送回；單欄位編輯用 `editor.set_field`（on tab 的 editor_id，走 session live LiveModel，真正只改一個欄位）。

**Run progress 架構（Phase 111，GUI v7；progress 查詢 Phase 129 收斂）**：progress 統一在 Qt-free `ProgressService`（`services/progress.py`），不再由 view/RunService 各造 `ProgressModel`。`RunService.start_run` 在 acquire 後 `progress.make_factory(token, owner_id=tab_id)` 鑄 worker factory。container 本就 keyed by operation_id；**agent 查詢走 `operation.progress(operation_id)` → `bars_for_operation(id)`**（直查 container，run + device setup 通用），取代舊 owner-keyed `run.progress`/`device.setup_progress`（已移除）。GUI widget 仍經 `bars_for_owner`/`attach_by_owner` 走 owner 路徑（不變）。worker→主線程 marshal 走 `ProgressTransport` port（Qt 實作 `QtProgressTransport` 在 `gui/adapters/`，queued-connection），ProgressService 因此無鎖。詳見 `gui/app/main/README.md` 的「Progress 子系統」。

**Run 生命週期事件**：`RUN_STARTED{tab_id}` 與 `RUN_FINISHED{tab_id, outcome, error_message}`（outcome ∈ {finished, failed, cancelled}、失敗帶 error_message）—— 一事件一語義（wire v7 拆自舊 `RUN_LOCK_CHANGED`，後者用欄位有無區分 start/end 對 agent 不友善）。MainWindow 兩者都訂：start→`refresh_run_lock(tab_id)`、finish→`refresh_run_lock(None)`，鎖刷新需求由兩 handler 共同滿足，不重複 emit。cancel 判讀在 `RunService`（持 stop_event 者，ADR-0019）：`BackgroundService` 只回 done/failed，`_on_bg_done`/`_on_bg_error` 看 `stop_event.is_set()` 映 finished vs cancelled（+ 可能的部分結果）→ `_on_run_cancelled`；`cancel_run` 走 `gate.cancel(token)` 設 stop_event（異步通知，不阻塞）。progress bar 帶 `percent`（0–100，total 未知時 None）+ raw n/total（經 `operation.progress`，折進 `gui_run_poll` running 回傳）。**Phase 129：run 一開始 `RunService.start_run` 即 `State.clear_tab_results`（清 run/analyze/figure/writeback）**——run **進行中/失敗**時 tab 無 result，analyze/save fail-fast `no_run_result`（非誤導的 no analyze params / busy）。**cancel 例外（Phase 130 釐清）**：worker 在 stop_event 前已產出的 partial result 由 `_on_run_cancelled` 保留（`has_run_result=true`、可 analyze），有意為之；完全沒產出才無 result。

**Device setup 生命週期事件（與 run 對齊，wire v10）**：`DEVICE_SETUP_STARTED{name}` + `DEVICE_SETUP_FINISHED{name, outcome, error_message}`，完全鏡像 run_started/run_finished（一事件一語義 + outcome）。進度走 `operation.progress(operation_id)`（同 run，按 id 查的 live 拉，不嵌 snapshot、不經 event；折進 `gui_device_poll` running 回傳）。progress 在 `ProgressService`（非 DeviceService、非 dialog）：container 生死綁 setup operation（`make_factory`→終局 `discard_operation` 在 `_finish_operation`）→ 關 dialog 不銷毀進度（container 仍活，重開 re-attach 即見）。

**`AnalyzeResultBase`** mixin：adapter result dataclass 繼承後自動取得 `to_summary_dict()`（反射過濾非 JSON-safe 欄位）。`NoAnalysisResult` 也繼承。

**save path 設計**：`save.data` / `save.image` / `save.result` 的路徑參數現為 optional，省略時 fallback 到 `get_tab_save_paths()`（override 優先，無 override 則 adapter 預設）。新增 `save.set_paths` 讓 agent 設定 path override，等同 UI 的 path 欄位。

**Dialog EventBus 訂閱狀態**：

- `InspectDialog`：已訂閱 `MD_CHANGED`、`ML_CHANGED`、`CONTEXT_SWITCHED` — 不需改動
- `DeviceDialog`：訂閱 `DEVICE_CHANGED`、`DEVICE_SETUP_STARTED`、`DEVICE_SETUP_FINISHED`；進度經 `ctrl.attach_progress(device_name, listener)`（active setup 時 attach、無 active / close 時 detach），listener 查 `ctrl.progress_bars(device_name)` 渲染 ProgressStack（owner-attach，Phase 111）
- `SetupDialog`：已加 `CONTEXT_SWITCHED` 訂閱（context list 即時更新）
- `PredictorDialog`：已加 `PREDICTOR_CHANGED` 訂閱（load/clear 後 dialog 即時更新）

Device / context mutation RPC 是 fire-and-forget command surface；狀態追蹤走 event subscription + requery 或對應 snapshot/query RPC。**ml entry 建/改沒有 raw-dict RPC**（ADR-0011 移除 `context.set_ml_*`）——agent 走 editor session（create_from_role + editor.*），與 user 經 inspect dialog 的 editor.commit 同一條寫入路徑，且支援 EvalValue。`device.setup` 不接受 client 建構的 `BaseDeviceInfo` subtype，避免把 live Python object 形狀暴露到 wire。

### editor.set_field / tab.list_paths path 文法

`path_resolver.resolve_and_set(root, path, value)` 走 tab 的 live `SectionLiveField`，用既有 mutation surface（不直寫 State；auto-commit 經 form `schema_changed`）：

- Scalar：`section.sub.field` → `ScalarLiveField.set_value`
- Sweep：`...path.start|stop|expts|step`（`sweep` 限定詞可選）；`expts` 限 int、其餘限 number；走 `SweepEditor.update_*`
- Multi-sweep：`...path.<axis>.start|stop|expts|step`
- ModuleRef：`...path.ref`（key）/ `...path.<sub>...`（**直接遞迴 sub_field，無 `value` 包裝段** —— wire v8 移除；殘留 `value` 段 fast-fail 報遷移訊息）
- DeviceRef：`...path.device` 或直接 leaf → `set_chosen_name`
- Literal → reject；未知 path / 型別不符 → `invalid_params`
- **`editor.set_field` 回 `{paths, removed, added, valid}`**：`removed`/`added` 是 ModuleRef key 切換對整棵 draft 的 settable-path diff（切 ref 重建 sub-tree → agent 不必重新 list 整 tab 即知哪些 path 沒了/出現）。

### CfgEditor sessions（widget + agent 共享的 cfg draft SSOT）

`CfgEditorService`(掛 Controller,`editor_id` 索引)**永遠持有**有狀態 LiveModel 編輯 session,被多個 client 共享 —— widget 與 agent 都只持 `editor_id`。`CfgFormWidget` 是可插拔 viewer,`attach(model)`/`detach()`(detach 不 teardown)。詳見 `gui/CONTEXT.md` 的 CfgEditor session 與 **`docs/adr/0010`**(supersede 0003 delegated;0003/0002 其餘仍生效)。共用一組 Qt-free 機制:`SectionLiveField` / `resolve_and_set` / `list_settable_paths` / `CfgSchema.to_raw_dict`(唯一 lowering 入口,Phase 120a)。生命週期由 `gc` 標誌歸屬:

- **gc=True**(`editor.open`):agent 編 ml entry,無 owner。`commit`(lowering→ml)/`discard` 自動回收;斷線經 `_ClientCtx.editor_ids` 回收(`_track_editor_lifecycle` 在 open 加、commit/discard 移除;`_drop_client` marshal 到 main / `stop` 直呼 → `_reclaim_editors` → `ctrl.discard_cfg_editors`);另有**數量上限 LRU**(`_MAX_HEADLESS_EDITORS`)防孤兒。commit 失敗不銷毀,讓 agent 重試。
- **gc=False**(`open_seeded(seed, owner_key)`,無 item_kind→teardown-only、拒 commit):tab/inspect/writeback 的 UI-owned 草稿。owner 顯式 `teardown(editor_id)`,**不**計入 LRU、不隨斷線回收。tab 用 tab_id 當 owner key,editor_id 經 `tab.snapshot` 暴露;writeback 每 module/waveform item 一棵(owner_key `writeback:{tab_id}:{session_id}`,item 持 `editor_id`,reanalyze teardown);inspect 切型別時 teardown 舊+開新。`CfgFormWidget` 經 `get_root(editor_id)` 取 model 後 attach。external refresh(md/ml 變刷 EvalValue)歸 service:訂 MD/ML/CONTEXT/DEVICE_CHANGED → 對每棵 owned model `refresh_external`(ADR-0007 Reaction),widget 經 model on_change 免費重畫,**不**碰 EventBus。

**漸進編輯**:ModuleRef/WaveformRef 切 key 會 partial re-bind 子樹,client 無法預知切換後結構。`set_field` 回**以被改 path 為根的子樹**(scalar 回自己;切 `<path>.ref` 回新浮現子樹)。

**eval-on-commit**:scalar 可填 `{"__kind":"eval","expr":"r_f - 0.1"}`,decode 成 `EvalValue`;`set_field` 當下即經 `ScalarLiveField.set_value` 對 md 求值 fill `resolved` snapshot,故 `commit` 經 `CfgSchema.to_raw_dict` lower 成 concrete(ml 只存 concrete)時靠 snapshot,不靠 md;unresolved → fail-fast。**commit drift 檢查**:`cfg_editor.lower` 經 `ModuleLibraryWritePort.get_current_md()` 把 md 一併傳 lowering(純為檢查,非解析依賴)——snapshot 與 md 現值不一致(field set 之後 md 變了)時 `logger.warning`,snapshot 仍勝(語義不變)。`editor.commit` 是 **upsert**(create-or-modify)語義,撞名靜默覆寫(與 create-only 的 `ml.create_from_role` fail-fast 區分)。

**變更流(editor 專屬,GUI 內部)**:`CfgEditorService` 對每 session 的 `root.on_change` 掛 callback,經 `set_change_listener`(RCS 在 start 注入 `_on_editor_event`)產生 `editor_changed`/`editor_closed`。**RPC/GUI 端機制完整保留**(供 GUI 內部 + 未來用),但 **Phase 120c-2 起 agent 不再曝露**:`gui_editor_subscribe/unsubscribe` 工具移除,editor 變更/失效 agent 改為「下次 `editor.set_field` 撞 `unknown editor session`」被動得知（樂觀模型，ADR-0003/0010 主動通知退化）。editor 草稿變動仍經 `Controller.bump_editor_version` bump `editor:<id>` 版本(供 editor.commit guard)。**懸空防護**:所有移除路徑唯一收口 `_remove`,必先 `on_change.disconnect(change_cb)` 再做其餘;gc=False 的 `teardown` 在 widget `detach` 之後。

### 資源版本表 + 版本 guard（Phase 94,防 user 偷改 agent 不自知）

agent 是 turn-based、工具間不自發 poll,對 GUI 上 user 的並行操作預設盲目。Phase 94 用**資源版本表 + optimistic-concurrency guard** 取代 Phase 92/93 的 change buffer + origin。三層分工:RPC=mechanism、mcp=簿記+翻譯、agent 只收語義(從不見版本號)。見 `docs/adr/0005`。

- **版本表(RPC,`State.version` = `VersionTable`)**:中粒度資源 key(`context`/`soc`/`device:<name>`/`tab:<id>:cfg`·`:result`·`:save_path`/`tab:<id>`/`editor:<id>`),per-resource 單調遞增。bump 由**資源 owner service 在主線**做(state mutator / 各 service terminal slot / `State` 的 device mutator〔Phase 97 後 device 狀態下放 State,`device:<name>` bump 在 `put_device`/`set_device_*`,**非** `DeviceService`;但 `refresh_device_info_cache` 不 bump〕 / `Controller.bump_editor_version`)—— 是 [[State main-thread invariant]] 的推論。`resources.versions` RPC 回全表快照(`_NON_GENERATED_METHODS`,**不**生成 agent tool)。
- **版本 guard(`_guard_versions`,main thread,run/save/commit handler 前)**:讀 params 的 optional `expected_versions`(mcp 填),原子比對每個 key 對 `ctrl.resources_versions()`;不符(含 key 不存在=資源已 drop=讀 0)→ raise `PRECONDITION_FAILED(reason=stale_version)`。沒帶=不檢查(同普通 RPC)。比對在 `_dispatch._run()` 主線單一同步序列內 → 對任何 GUI 寫原子(真人操作也上主線)→ 無 TOCTOU。guard 不懂依賴語義(mechanism)。
- **`expected_versions` 是 wire-only(MCP-hidden)**:`ParamSpec.mcp_hidden=True`(`_expected_versions()`)→ 驗證+達 handler,但 `build_input_schema` 跳過 → 不進 agent schema。版本號只在 RPC↔mcp 流動。
- **mcp 簿記(policy)**:`mcp_server` 持 `_LAST_SEEN`、`_GUARD_DEPS`(依賴對應表:run→cfg/tab/soc/context/device:*/devices:__set__;save→result/save_path;commit→editor/context)。`device:*` glob 守既有成員 mutation,`devices:__set__` 基數 key 守集合成員新增/移除(glob 對新成員失明,Phase 102)。`send_gui_rpc` 對 guarded method 由 `_build_expected_versions` 組 `expected_versions`;**`_refresh_versions()`(`resources.versions` 全表,經 `_BRIDGE.send_rpc_raw` 避遞迴)在每次成功 RPC 後 refresh `_LAST_SEEN`** —— agent 同步 round-trip = 觀察到當前。async terminal(run/device/connect 完成)的版本 bump 由 agent 的 **poll/wait**（`gui_*_poll`/`gui_*_wait` 內部即 `operation.await` round-trip）承接 resync（Phase 120c-1/2 取代舊「event poll 後 refresh」）。被拒 → refresh + **`data.stale` 翻成語義訊息**給 agent（Phase 120c-3：`_describe_stale_keys`，「the active context / this tab's cfg / device 'flux'」）。
- **通知面（Phase 120c-2）**:GUI EventBus 照常 push 全部 event 上 wire，但 mcp 端只放行 diagnostic（resource-change event 丟棄）。agent 不再「訂閱+poll 收什麼變了」；resource 變動靠 guard 撞牆告知（語義 stale），async 完成靠 poll/wait。

### Off-main blocking handler（Phase 93/95）

`MethodSpec.off_main_thread`(`BoundMethod` 同名 property 轉發)標記 blocking 等待型 handler 在 **IO worker thread** 直接跑,**不** marshal 上 main thread。兩個用途:`operation.await`（等 op settle）+ `notify.await`（等 user 回應 prompt，WIRE 30）。若上主線,blocking wait 會卡死 main event loop,而它等的 main-thread 事件(worker terminal signal / dialog 回應)正需 event loop 處理 → 死鎖到 timeout。

- `_dispatch_on_main`（**現在共用 base `RemoteControlServiceBase` 擁有**，三 app 同一份）:`spec.off_main_thread` 為真 → 直接在 IO thread 跑 handler(無 guard / `done.wait`)；為假 → marshal 上主線，`_run` 內先呼 `self._guard(params)` 再跑 handler。measure-gui 的 `_guard` seam 轉呼具名 `_guard_versions`、`_after_success` seam 轉呼 `_track_editor_lifecycle`（read-only app 兩者皆 no-op、且永不設 off_main_thread，故行為等同 bare marshal）。
- **嚴格契約**:off-main handler 只能做 thread-safe 等待,**不得**碰 main-thread-owned 狀態(version table / CfgEditor / `_snapshots`)、需要版本 guard。新增 off-main handler 必須守此契約。
- `operation.await`(Phase 95)只 `OperationHandles.await_outcome(operation_id, timeout)` 消費該 op 的 per-token `OperationChannel`(單一有序事件 FIFO Settled/Message/Stop，`Queue.get(timeout)` 入列即醒，取代舊 `threading.Event`+`FeedbackInbox` poll-loop；main thread `handles.settle`/`message`/`stop` 為 producer、off-main waiter 為 consumer，依到達序折成 AwaitResult；ADR-0025/0019)。取代 Phase 93 的 by-name `wait_setup_done`。
- `notify.await`(WIRE 30)消費 per-prompt `NotifyChannel`(獨立 event 詞彙 Reply/Dismiss/Timeout，鏡像 OperationChannel 四不變式但不混進 operation 事件集；dialog 是 timeout SSOT)。agent 經手寫 `gui_notify_user` serial compose `notify.open`(主線 mint+開 dialog)+ `notify.await`。**嚴格契約同上**:off-main 只做 thread-safe 等待，producer(dialog 回呼/QTimer)在主線。
- **mcp operation-handle 簿記**:start op 回的 `operation_id` 由 `send_gui_rpc` 經 `_OP_KEY_OF[method](params)` 捕捉進 `_OP_BY_KEY[semantic_key]`(latest wins)並從 result strip(raw id 不上 agent)。device connect/disconnect/setup 都 key 在 `device:<name>`;run→`tab:<id>`。**`soc.connect` 不在此列**——它是同步 RPC，無 operation handle、無 `_OP_KEY_OF` 項、`operation.await/poll` 也不再認 "soc" key。agent 用語義 wait tool 翻語義名→id→`operation.await`：`gui_device_wait_operation(name)`（涵蓋 connect/disconnect/setup）、`gui_run_wait(tab_id)`。
- **mcp short-wait degrade**(`_start_op_with_short_wait`，泛化):有快/慢雙模態的 start op 都套——`gui_device_connect/_disconnect/_setup`、`gui_run_start`。start 後等 `wait_seconds`(預設 1.0):settle→回 `{status:"finished", **product()}`(device→`device.snapshot`、run→`tab.snapshot`，各自的產物狀態);TIMEOUT→`{status:"pending"}`(agent 改對應語義 wait：`gui_device_wait_operation`/`gui_run_wait`，或 watch `*_finished`/`device_changed`);genuine failed/cancelled 仍上拋。便利封裝歸 mcp 層,RPC 維持 start+await 兩原語。**`gui_soc_connect` 不走此 degrade**——它直接同步呼叫 `soc.connect`（帶顯式 ~2s timeout，讓 board 端 1s COMMTIMEOUT 先觸發乾淨錯誤），回 `{status:"finished", soc:{description, is_mock}}`，無 `_wait`/`_poll`。

### Typed Request Coercion

所有 typed request dataclass（`ConnectRequest` union、`StartupProjectRequest`、`ConnectDeviceRequest`、`DisconnectDeviceRequest` …）在 `wire.py` 的 `coerce_*_request` helper 邊界轉換。**raw dict 永遠不流入 Controller**（ADR-0011 後 ml-entry raw RPC 已移除，無例外）。缺欄位或型別不符直接 `RemoteError(INVALID_PARAMS)`。

### `run.start` 合約

`run.start` RPC reply OK immediately（不等 run_finished）。理由：若 RPC 等到 finish 會佔住 socket thread 數分鐘，與 30s timeout 衝突。**mcp 端** `gui_run_start` 則加 short-wait degrade（快 run 直接回 tab snapshot、慢 run 回 handle）。慢 run 追蹤：`gui_run_wait(tab_id)`（blocking）或 `gui_run_poll(tab_id)`（非阻塞，running 時回傳自帶 live progress bars）。

## CLI Integration

`script/run_measure_gui.py` 改用 `argparse`：

```
python script/run_measure_gui.py --control-port <port> [--control-token <hex>] [--control-allow-external]
```

`--control-port 0` 拿 ephemeral free port；服務啟動後印 INFO log 到 stderr / file。

`gui/app/main/app.py:run_app(control_opts=ControlOptions(...))` 在 window 建立後啟動 service 並 stash 到 `window.remote_control_service`；teardown 走 `MainWindow._perform_close`（closeEvent 與 RPC shutdown 共用入口），在 `self._ctrl.persist_all()` + `set_shutting_down(True)` 後呼叫 `service.stop()`，最後 `super().closeEvent(a0)`。

## MCP Bridge Contract

`mcp/measure/server.py` 是 NDJSON RPC 的 agent-facing bridge，對目前 RemoteControlAdapter 的 public command/query surface 提供 explicit tool mirror。任何新增 RPC method 若希望 LLM / MCP host 直接使用，都需要同步新增 MCP tool、完整 `inputSchema.type`、wrapper mapping 測試，以及最小 smoke / integration 測試。

**兩類 tool**:**生成型**（`_generate_tools` 由 `METHOD_SPECS` 鑄，1 tool↔1 RPC，schema 來自 ParamSpec SSOT）與**手寫 override**（`_OVERRIDE_TOOLS`，生命週期 / fan-out / file-write / 型別 coercion 等生成器表達不了的）。兩集合**按名互斥**（`_assemble_tools` 撞名即 programming error）。新增 override 須同列入 `_OVERRIDE_NAMES`。

**Batch convenience 工具（mcp-side fan-out，不動 wire）**:`gui_editor_set_fields`（一個 editor 套多個 `{path,value}`）與 `gui_context_set_md_attrs`（多個 `{key,value}` MetaDict attr）是 `_OVERRIDE_TOOLS` 裡的迴圈封裝，內部逐筆呼叫**既有** `editor.set_field` / `context.set_md_attr` RPC——**不新增 wire method，故 WIRE_VERSION 不動**（與 piggyback 同類「純 mcp 端 convenience」）。語義刻意 **fail-fast 且非原子**：第一筆失敗即停、先前已套用的**不 rollback**、raise 點名失敗的 path/key 與已套用筆數，讓 agent 自行 reconcile（符合專案 Fast-Fail 風格；真正原子性需 RPC 層交易支援，目前不提供）。`set_fields` 成功回 `{applied, valid}`（套用筆數 + 結果 draft 是否合法），**刻意不 echo cfg 內容**（echo 會逼一次 lowering，eager 求值 EvalValue）；要看 cfg 走 `gui_tab_list_paths`。入參在任何 RPC 發出前先驗證 shape（`_coerce_pairs`，空 list / 缺 key 直接 `ValueError`），把失敗邊界鎖在「nothing applied」。

## Reuse 既有元件

- `SessionPersistenceService.schema_to_raw` / `raw_to_schema`（`gui/services/session_persistence.py`）—— cfg roundtrip 不重寫 serializer。
- Controller 既有 façade —— Phase 80 只多 `list_tab_ids` / `get_tab_adapter_name` / `get_tab_cfg_schema` 三個 read-only wrapper。
- typed request dataclasses（`ConnectMockRequest` / `ConnectRemoteRequest` / `StartupProjectRequest` / device request dataclasses）—— 不重複定義，只在 wire.py 包 coercion。
- `editor.set_field` path resolver 重用既有 `LiveField.set_value` / `SweepEditor.update_*` / `set_chosen_key` / `set_chosen_name`，不新增 mutation surface。

## EventBus 訂閱管理慣例

各 dialog 的 EventBus 訂閱一律集中在 `_cleanup_bus_subscriptions()` 方法，`__init__` 末尾同時連接 `self.finished.connect` + `self.destroyed.connect`，並用 `_bus_subs_active` flag 防止 double-unsubscribe。

- `setup_dialog.py`：`CONTEXT_SWITCHED` / `SOC_CHANGED` / `DEVICE_CHANGED`
- `inspect_dialog.py`：`CONTEXT_SWITCHED` / `MD_CHANGED` / `ML_CHANGED`
- `cfg_form.py`：例外，使用 `_bus_subs` 列表（動態回調需要）
- `main_window.py`：集中清理方法 `_cleanup_bus_subscriptions()`

## Device Setup Events（wire v10，progress 不再經 event）

Setup 只發**兩個** event：`DEVICE_SETUP_STARTED`（worker 啟動）+ `DEVICE_SETUP_FINISHED`（terminal，帶 outcome）。**進度不經 event**（走 `operation.progress(operation_id)` 拉 live model，折進 `gui_device_poll` running 回傳），所以沒有舊版「大量 progress step 塞爆 event queue」的問題——早期的 `_emit_setup_changed` 節流（每 20 步）已隨之移除（progress SSOT 重構 + device↔run 對齊）。

## `gui_stop` / pid file

`gui_launch` 在啟動後寫 `$TMPDIR/zcu_tools_gui.pid`；`gui_stop` 經 `McpBridge._pid_for_stop()`：子進程握把 `self._proc` 還活著就用它，否則 fallback 讀 pid file（支援 MCP session 重啟後找回 GUI 進程）。`main()` stdin EOF 時自動呼叫 `_cleanup_on_exit()` → `gui_stop`。

`gui_launch` 啟動的 GUI 子進程繼承 MCP bridge process 的環境。Linux desktop session 是否可見取決於啟動 MCP bridge 的 host 是否保留 `DISPLAY` / `WAYLAND_DISPLAY` / `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` / `QT_QPA_PLATFORM` 等 session env。Claude 的 project `.mcp.json` 路徑可在正常 desktop session 下直接繼承；Codex 使用 `.codex/config.toml`，優先用 `env_vars` 白名單繼承 parent env，只有 Codex parent process 本身缺 display/session env 時才需要 explicit `env` bridge。不要把 per-user Linux display socket 寫死到共享 `.mcp.json` 作為跨平台預設；Windows 應避免注入 X11/Wayland env，讓 Qt 使用 Windows platform plugin。

## 後續可能擴充（未排程）

- tab 表單經 `editor.set_field` 的 EvalValue 編輯（目前 tab 的 sweep edge / scalar 限 numeric / direct;ml 編輯的 eval 同樣由 `editor.set_field` 支援）。
