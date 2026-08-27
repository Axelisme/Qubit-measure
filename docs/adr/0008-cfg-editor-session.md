---
status: accepted
---

# CfgEditor session —— service-owned headless CfgDraft + 可插拔 widget viewer

**狀態：** accepted（已實作）。
**關聯：** external-refresh 是 [[0004]] Reaction 標準模式；commit 的 ml/md 寫入歸 [[0006]] 單一權威；tab cfg 讀寫收斂見 [[0013]] F11；agent 樂觀模型見 [[0002]]。

## 脈絡

agent（MCP RPC）與 user（Qt View）都要編輯三類 cfg：tab 的 cfg、ModuleLibrary 的 module/waveform entry、analyze 算出的 writeback 草稿。三個耦合需求：

- **(a) agent 改 cfg 必對 user 即時可見（WYSIWYG）**——不能背著 user 改。
- **(b) widget 未開時 agent 也要能編輯**——writeback 草稿 / 未開的 tab，沒有 widget 但 agent 要能調。
- **(c) 表達 EvalValue 與 ref 切換**：欄位值可引用 MetaDict（`freq = r_f − 0.1`，commit 才 eval）；module/waveform 含 ModuleRef/WaveformRef，切換會**動態改變後續可填欄位**，agent 必須「切 ref → 看新欄位 → 再填」漸進進行（無法一次送整份 raw）——這是引入**有狀態 session** 的不可替代理由。

## 決策

**`CfgDraft` 永遠由 `CfgEditorService` 以 headless 模式持有；`CfgFormWidget` 是可插拔 viewer。** 直接操作一般cfg-editor session的agent與widget以`editor_id`作為平級client；writeback則由`WritebackService`把這個identity封裝在opaque draft內，不交給workflow、adapter或wire client。

- **attach/detach**：`CfgFormWidget.attach(draft)` 接一個 service-owned `CfgDraft`、從 `draft.root` build widget tree；`detach()` disconnect + `deleteLater`，**不 close draft**。widget 的 Qt 重畫經 draft/field `on_change` 取得。
- **gc 生命週期**：`open(..., gc)`。`gc=True`（agent 自開 ml-entry）受 LRU + 斷線回收；`gc=False`（UI-owned：tab / inspect / writeback）只由 owner 顯式 `teardown`。tab cfg / writeback 草稿種子用 `open_seeded`（無 item_kind → teardown-only、拒絕 commit）。
- **draft / committed**：session = draft，`State.cfg_schema` = committed（run/save/persist 讀的 SSOT）。tab session 改動經 auto-commit（widget `on_change` → `schema_changed` → `update_tab_cfg`）即時同步進 State；run/save 前一道**強制 commit = valid 驗證閘**（draft invalid → fail-fast）。
- **commit 只交 CfgSchema 快照**：`CfgEditorSession.commit` 不 lower、不 register，只交出**未-lower 的 `CfgSchema`**；`CfgEditorService.commit(editor_id, name)` 把它交給 ContextService 經 write port 落地（lowering + register 歸 ContextService，見 [[0006]]）。
- **external refresh 歸 service**（[[0004]] Reaction）：service 訂閱 `MD/ML/CONTEXT/DEVICE_CHANGED`，明確映射到每個 draft 的 `refresh_expressions()`、`refresh_references()` 與 `refresh_options(source_id)`。職責跟著 draft 所有權從 widget 移到 service。
- **eval value** 以 tagged 形式 `{"__kind":"eval","expr":...}` 上 wire；**ref 切換漸進**：`editor.set_field` 回「以被改 path 為根的子樹 paths」+ valid，讓 agent 探索切換後新浮現的結構；commit 失敗保留 session。
- **失效訪問**：任何原因消失的 editor_id（LRU / tab close / commit / discard / 斷線）一律回 `unknown editor session`（INVALID_PARAMS），**不帶 reason 區分**（修復動作都是重開）。
- **editor 專屬變更流**（`editor_changed{editor_id, paths}` / `editor_closed{editor_id, reason}`，**不走全域 EventBus**）：機制在 RPC/GUI 端完整保留（GUI 內部用）；但 **agent 不 subscribe**（[[0002]] Phase 120c）——agent 改為「下次 `editor.set_field` 撞 `unknown editor session` 才知 session 沒了」，與樂觀模型一致（撞牆→重開）。
- **tab cfg 讀/寫/發現全收斂到 session model**（[[0013]] F11）：`tab.list_paths`（讀）、`editor.set_field`（寫）、tab snapshot 暴露的 `editor_id`（發現）三者都對該 tab 的 editor session model，**agent 與人同一棵**。原 `cfg.set_field` RPC / `get_tab_live_model_root`（戳 View 的另一棵 model）已刪。

## writeback opaque draft（建立在 service-owned 上）

analysis workflow一次算出adapter proposal items，再交給`WritebackService.create_draft()`建立opaque、service-owned draft。每個module/waveform proposal在draft-private entry中配對一棵`gc=False` headless model（種子 = item的`edit_schema`）；adapter item、workflow與wire projection都不持有`editor_id`。Analysis與Post-Analysis pane各自保存自己的opaque draft；read model只投影該pane可見的writeback items。user點開Edit dialog時，Writeback以draft與item session id解析並回傳同一棵`CfgDraft`供widget attach（WYSIWYG）。

Draft建立具有strong exception guarantee：任何editor open失敗時，Writeback在拋出原始錯誤前清理該draft已開啟的所有sessions，不把未成功回傳的cleanup責任交給workflow。Teardown先封閉draft再逐一清理，因此可重複呼叫；單一session cleanup失敗不會造成其他session被略過或讓已封閉draft再次close。Apply snapshot各item model並只把selected items組成一個`ContextWrites`交給ContextService（見[[0006]]），不直接修改md/ml。

**agent編writeback草稿的入口統一在writeback面：** remote/MCP以required `(tab_id, subtab_id)`定址Analysis或Post-Analysis draft，再以item session id選取draft-private entry；Writeback透過`CfgEditorPort.set_fields`套用ordered、fail-fast edits。`editor_id`不上writeback wire，agent不先取得editor handle。metadict item仍走同一面的value facet。這個封裝只涵蓋writeback draft；`editor.*` RPC對ModuleLibrary ml-entry的直接編輯仍使用`editor_id`，不受影響。

`CfgEditorPort`包含open、teardown、draft read與field edit所需的窄介面，使`WritebackService`依賴port而非concrete service（[[0005]]）。底層仍是同一個service-owned `CfgDraft`，因此user dialog與agent共享同一份可見草稿。

## 安全保證（核心動機）

「agent 的修改必對 user 可見」是**結構保證**：agent 可定址的每一棵 model 都 service-owned 且**隨時可被對應 widget attach 重畫**（tab 存在即可 attach；writeback item 經 Edit 可 attach）。不存在 agent 能改、但任何 surface 永遠無法顯示的 model。

## 演化（被取代的設計，保留脈絡）

CfgEditor session 經兩次轉向才到現行形狀：

1. **headless-only（RPC 專用、不綁 tab/View）**：最初定為「不讓 GUI 操作走它、不加逾時」。**翻轉**——agent 用 `editor.*` 改 tab cfg 時 user 完全看不到過程（headless、無 widget 綁定，commit 後才可見）。於是升為「被多 client 共享的 cfg draft SSOT」，widget 與 agent 平級。
2. **delegated（委派型）**：widget `populate` 時自建 field tree、委派 service 換 id，「widget 先 populate 才有 session」。**翻轉**——tab/dialog 沒開就沒 widget、沒 draft，agent 無從編輯（writeback / 未開 tab 尤其卡）；且 external refresh 綁在 widget 的 EventBus 訂閱、無 widget 不刷，agent-only 編輯讀到 stale EvalValue。改為現行「`CfgDraft` 永遠 service-owned headless、widget 只 attach」。

兩次轉向都圍繞同一動機：把「agent 改必對 user 可見」從*希望*變成*結構保證*。

## 替代方案與否決理由

- **在 `SectionWidget` 上加 attach/detach**：須複製建構子邏輯、波及每個 field widget class，零收益。改由 `CfgFormWidget` new/deleteLater widget tree。
- **保留 widget EventBus 訂閱 + 另給 widget-less model 一條 service 刷新路徑**：同一職責兩條路徑、易 race。統一歸 service。
- **adapter writeback item 持model參照或editor_id**：會把cfg-editor session identity洩漏到adapter dataclass與wire projection。改由`WritebackService`在opaque draft private entry中保存關聯。
- **agent 變更通知走全域 EventBus**：稀釋 EventBus 的 changed-resource 語義、細粒度欄位變更量大。採 editor 專屬流。
- **headless 孤兒用閒置逾時回收**：引入時間語義 + 計時器。採無時間語義的數量上限 LRU。
