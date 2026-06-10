# 0013 — RemoteControlAdapter 作為與 MainWindow 平級的第二個 View

**狀態：** accepted（**已實作**，ADR-0005 M6 的展開）。實作分 6 commit：C1 typed ctrl（gui2:90835c02）、C1b F11 cfg 收斂+run-guard（73668379）、C3 正名+handler 收 adapter（f9379e36）、C2b-1 ViewProtocol 拆分+診斷 fan-out（9ccef9e3）、C2b-2 agent 收診斷+render 經 adapter.render_view+刪 ViewQueryService（d65cc55b）、C2a mcp piggyback+診斷分流+default-subscribe（ae0ca4b7）。WIRE_VERSION 6。
**關聯：** 展開 [[0005]] M6；承 [[0001]]（Permit/Lease 共用 guard 路徑）、[[0002]]（async emit / off-main）。

## 實作偏離規劃的關鍵調整（grill 後定）

- **C2a piggyback 重定位到 mcp 端**：RPC transport 本來就 live-stream events（per-client writer thread），不存在「buffer 等 reply 夾帶」。真正的 buffer 在 mcp 端（`_EVENT_QUEUE`）。故 piggyback = mcp 在任意 tool result 附第二個 content block（drain `_EVENT_QUEUE`+`_DIAGNOSTIC_QUEUE`）；RPC 端零改動。
- **default-subscribe 歸 mcp 端**：「哪些 event agent 不該漏」是**實驗語義**，由 mcp 在 connect 後自動 `events.subscribe`，不在 RPC 端預置（RPC=純機制）。RPC 的 `events.subscribe` 已是 dedup'd set-add。
- **診斷與 event 同走 socket、mcp 端分流**：diagnostic 仍是無條件 push（RPC 不 gate），但 mcp reader 分到 `_DIAGNOSTIC_QUEUE`；poll 與 piggyback 都帶兩者。
- **C2b set_field 的 run-guard（F11 延伸）**：`editor.set_field` 原無 run-guard；新增「session owner 是 running tab → precondition_failed」（`owner_of_editor` 反查），與人靠 disabled form 同義。

## 背景

`RemoteControlService`（`gui/services/remote/`）在 [[0005]] 判讀表早被認定為 **Driving Adapter**——與
`MainWindow` 平級的 primary adapter（user 可以是人，也可以是 another server / agent）。但現況有三個名實不符：

1. **名字仍叫 `Service`**（DDD 詞彙裡 Service 是被動編排的 App Service，不是 user-facing adapter）。
2. **`__init__(controller: object)`**——ctrl 無型別契約。後果是 `service.py` 6 處 `getattr(self._ctrl, "<m>", None)`
   + 靜默降級，以及 `dispatch.Handler = Callable[[object, ...]]` 全 handler 的 ctrl 不被 pyright 檢查（改
   façade 方法名不會被抓、執行期才炸）。此即 Phase 110 review **F3**。
3. **下行渠道未對齊 MainWindow**：MainWindow 有兩條下行——狀態（EventBus subscribe）+ 診斷
   （ctrl 直呼 `show_error_dialog`/`show_status_message`）。`RemoteControlService` 只接到狀態那條
   （經 outbound queue / poll），**診斷那條完全沒接**——agent 觸發的 run 失敗，error 彈去 MainWindow
   的對話框，agent 那條 socket 上什麼都沒有。

同時 `ViewProtocol` 累積了一堆「ctrl 純轉手給 View」的方法（screenshot×3 / snapshot / live_model_root /
dialog 組），它們只為 agent 經 ctrl 拉 View 資料而存在，污染了 ctrl 的 View 接口與 façade。

## 決策

把 `RemoteControlService` 擺正成 **`RemoteControlAdapter`**——與 `MainWindow` 平級、實作**同一組拆分後 View 接口**
的第二個 View。ctrl 子系統對外有「一組 View」，對 View 有兩類下行概念（狀態、診斷），而**「實際怎麼把通知送達」
是每個 View 內部的私事**。

### 1. 持有具體 `Controller`，非 Protocol

經典 MVC/MVP 中 View→Controller 直接持有具體 façade 是正統；`MainWindow` 已 `self._ctrl: Controller`。
`RemoteControlAdapter` 對齊之（`__init__(controller: Controller)`、`Handler = Callable[[Controller, Mapping], Mapping]`）。
Controller 只有一種實作，再包 Protocol 零收益（無多型、無替換需求），純增驚訝。
→ 消 F3 全部 6 處 getattr（pyright 知道方法存在，降級是死碼）。

**注意**：`controller.py` 已 `from .services.remote.dialogs import DialogName`。為避免 `remote → controller`
反向 import 成環，`Controller` 的 import 走 `TYPE_CHECKING` + 字串 annotation（型別期可見、執行期不 import）。

### 2. 診斷不上 EventBus

「報告系統故障的通道，本身就是可能故障的那個系統」——診斷必須獨立於它要診斷的業務匯流。故診斷走
**ctrl→View 直接呼叫 fan-out**，不經 EventBus。對齊現狀：warn 走 `logger.warning`、operation error 走 RPC
error envelope（同步回發起者）、UI 彈框走 `show_error_dialog`，皆不經 bus。

診斷接口統一為 `notify_diagnostic(severity: Literal["error", "info"], title: str, message: str)`，取代 ctrl
~20 處 `show_error_dialog`/`show_status_message`。**severity 只放 `error`/`info`，不放空殼 `warn`**（現無 warn
呼叫點，放了違反 Fast-Fail；有來源時再誠實增量）。各 View 自決呈現：MainWindow `error`→彈框、`info`→狀態列；
agent enqueue 一條 diagnostic wire 行。

### 3. View 接口三層切分（基於實證 caller 分析）

| 接口 | 基數 | 掛在 | 方法 | agent 實作？ |
|---|---|---|---|---|
| **`DiagnosticSink`** | 多個（ctrl 持 `list`，fan-out） | ctrl | `notify_diagnostic` | 是 |
| **`RenderHost`** | 單一（`Optional`） | ctrl | `make_live_container`、`current_left_panel_width` | 否——ctrl `start_run`/`analyze` 核心依賴，兩 client 都觸發，**必須留 ctrl** |
| **`RenderView`** | 單一 | **adapter 直接持有** | screenshot×3、`get_view_snapshot`、dialog 組 | 否——純讀，100% 只為 agent，連同對應透傳搬離 ctrl |

- `ctrl.set_view` → `add_view`：ctrl 持 `list[DiagnosticSink]`（診斷 fan-out）+ 單一 `Optional[RenderHost]`（渲染依賴）。
- `RemoteControlAdapter.__init__(controller, render_view)`：render 查詢經 `adapter.render_view` 拉，**不經 ctrl**。
- headless（無 MainWindow）：`RenderHost=None`，`RunService.start_run` 已容忍 `Optional` pbar/container。
- **`ViewQueryService` 整搬 adapter**：F11 移除 `set_field`/`get_tab_live_model_root` 的 View-model 用途後，
  `ViewQueryService` 變純讀（snapshot/screenshot）→ 整個移交 adapter 的 `RenderView`，ctrl 不再持有。

### 3b. F11：tab cfg 編輯收斂到 ADR-0008 editor session（與人同一棵 model）

**違規（review F11，與 F1/F2/F3 同病灶）**：tab cfg 的編輯/發現有**兩條並存的路**——
- 人（MainWindow）：`open_seeded_cfg_editor(owner=tab_id)` → `CfgEditorService` 持 session model，form `attach` 當 viewer（合 [[0008]]）。
- agent：`cfg.set_field` / `tab.list_paths` → `get_tab_live_model_root` → **戳 View 的 model**（繞過 CfgEditorService）。

兩 client 走**不同 model**，違反「兩 client 必經同一路徑」+ [[0008]] 第 24 行「LiveModel 永遠由 CfgEditorService 持有」。
而正路早已全通：tab snapshot **已暴露** `editor_id`（`editor_id_for_owner(tab_id)`），`editor.set_field` 已存在。

**修法（純收斂，零新機制，正路現成）**：
- 刪 `cfg.set_field` RPC + `gui_cfg_set_field` tool + `Controller.set_tab_field` + `ViewQueryService.set_field`。
  agent 改 tab cfg 改走已存在的 `editor.set_field(editor_id, ...)`（editor_id 由 tab snapshot 取）。
- `tab.list_paths`（方案 α）：**wire 形狀不變**（agent 仍 `gui_tab_list_paths(tab_id)`），內部數據源從 View model
  修正為該 tab 的 **editor session model**（經 `editor_id_for_owner`）。便利查詢定位保留（[[rpc-vs-mcp-layering]]），
  agent 無感、違規消除。
- 兩用途遷走後 `get_tab_live_model_root`（經 View）整刪 → View 那棵 model 無人讀寫 path。
- 結果：tab cfg 的讀(list_paths)/寫(set_field)/發現(editor_id) 全收斂到 CfgEditorService session，與人同一棵。

### 4. 狀態渠道：共享 EventBus 已對齊 + piggyback 第二出口

兩 View 都 `bus.subscribe` → 已用「同一接口（EventBus）」對齊。RemoteControlAdapter 內部 per-connection buffer
（即現有 outbound event 累積）為**一個來源、兩個 drain 出口**：

- 出口 A `events_poll`（顯式阻塞拉，給閒置等待長 run 完成）——保留。
- 出口 B **piggyback**（任意 RPC response 順路 drain 並夾帶）——新增。
- 兩出口 drain 同一 buffer，不重複不取代。`events_subscribe` 仍決定 buffer 收哪些 event。

**default-on 訂閱**（agent 不 subscribe 也收）= `RUN_LOCK_CHANGED` + `DEVICE_SETUP_CHANGED` + `SOC_CHANGED`
（agent 最可能在等的 background 完成/失敗狀態）。此為**狀態** default-on，非把診斷架在 bus 上（見決策 2）。

## 拒絕的替代方案

- **給 ctrl 包 typed Protocol（`RemoteControlHost`）**：ctrl 單一實作，包 Protocol 是 `MainWindow` 都沒有的多餘
  抽象，不對稱地過度工程化第二個 View。改用具體 `Controller`。
- **診斷進 per-connection buffer / EventBus**：違反「報告故障的通道不該是故障系統」——bus 故障時診斷發不出去。
- **狀態也經 ctrl→View 接口 fan-out**（翻掉 pub/sub）：39 處 service `bus.emit` 是 ctrl 子系統內部實作，
  pub/sub 讓 service 與 View 解耦是對的；強行收進 ctrl 會讓它變狀態中轉上帝對象。
- **F11 廢 `tab.list_paths` 並入 `editor.*`（方案 β）**：path 發現塞進 editor.* 是更大的 wire 重塑，
  收益（少一 RPC）不抵 churn。改方案 α：保留 wire 形狀、只修正內部數據源至 session model。
- **F11 只刪 `cfg.set_field` 寫路、留 `tab.list_paths` 繼續讀 View model**：讀寫分走兩棵 model，
  結構漂移時 path 對不上。否決——讀寫必須一起遷到 session model。
- **severity 含 `warn`**：無呼叫點產 warn，空殼枚舉違反 Fast-Fail。

## 後果

- F3 解除；全 handler ctrl 受 pyright 檢查。
- F11 解除：tab cfg 讀/寫/發現收斂到 CfgEditorService session，agent 與人同一棵 model。
- agent 與 MainWindow 拿到同一份 ctrl 主動報的診斷（error/info），不再靜默漏接。
- `ViewProtocol` 從 11 方法瘦成 `DiagnosticSink`(1) + `RenderHost`(2)；ctrl façade 刪純讀透傳；`ViewQueryService` 純讀整搬 adapter。
- wire 變更（diagnostic 行 + piggyback + 刪 `cfg.set_field`）：WIRE_VERSION 5→6。
- 五個 commit（一 phase）：C1 typed ctrl / **C1b F11 cfg 收斂** / C2b 診斷 fan-out + 接口拆分 / C2a 狀態 piggyback / C3 改名。
