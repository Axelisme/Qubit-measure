# `zcu_tools.gui.app.measure` — measure-gui

**Last updated:** 2026-10-16，Artifact／Writeback owner、Opaque Run／Load、共用 Form／持久 Spec、ARB 唯讀、terminal Controller close 與 raw cfg 非 UI 讀取

`gui.app.measure` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。使用者套件的 adapter 擁有實驗領域知識；
framework 只看 `ExpAdapterProtocol`。共用 `BaseAdapter` 住在 `adapter/base.py`。它宣告 `Form`，分頁從固定 `FormEnv` 產生 fresh input。

Main 擁有 `AppPersistedState` codec/version、filename、originator、restore presentation 與
lifecycle-only triggers；disk mechanism 使用 `gui.session.persistence.SingleFileCaretaker`。

RunInputs 固定 admission 的 entry／point、alias、storage capability 與 adapter 的 persistence tag。
`RunService` 在 worker submission 前固定 UTC started_at 和 run ID。
Worker 在 acquisition 前捕捉 software provenance，完成後補 finished_at／completion。
Run 框架在 hardware lease 後、adapter.run 前讀取全部已連線 device 的 get_info，各一次。
任一讀取失敗拒絕 Run，不寫設定、不進 adapter/core、不以 State cache 補值，無 device 記 `{}`。
這份實驗前 snapshot 沿 RunMetadata.device_snapshot 保存，不被後續 sweep 改寫。
Native Save 寫入既有 `/device`，不查 live hardware；實際 sweep 值由 cfg 與 persisted axes 記錄。
Run handle 保持 adapter-owned opaque，不因 native evidence 要求 RunRecord 或讀 cfg／result 欄位。
`artifacts` 集中 tracker、SaveService、SaveControl、保存中心與 `ArtifactHandover` 契約。
State、controller、remote 和 Qt caller 使用該 package 公開入口；不保留舊位置 shim。
Adapter 的 `data_item(key, result, comment=...)` 與 `analysis_item(key, result, parameters, pane=...)`
只交固定 framework items。框架選路徑與格式、同步編碼及逐輸出隔離失敗。
Data 準備錯誤交給 DataItem.failed 或 from_run，producer 拋錯也轉成該 data outputs 的失敗；
同選集的圖與分析仍保存。Figure 取已提交 pane，不取未提交草稿。
`writeback` 集中三類 proposals、opaque draft、WritebackService、WritebackControl 與 widget。
Adapter 公開入口繼續再匯出 proposal 型別，其他舊位置不保留 shim。
resources.entry executor 與 remote handler／wire types 留在各自 owner。
Artifact 與 Writeback 不直接互相 import，由 app composition 協調。
`measure-artifacts-no-writeback` 與 `measure-writeback-no-artifacts` contracts 約束兩個方向，
允許共用 State／events／context。
可提交的 native 結果先由 State 一次發布結果與 `RunPaneState.metadata`，
再在 terminal 追加 acquired。Origin 沿發起命令的 GUI／MCP，沒有 Save 也追加事件，
但不建立 run 資料夾。未綁定或沒有 storage 時不造事件，terminal 明確記錄診斷。
Ledger 失敗使 operation failed 並記錄 ERROR，但保留已發布的結果供 Save。
不重試、回滾或釋放已發布的結果資源。Save request 取用這份歷史
metadata，不從當前來源重抓，也不以存檔時間冒充量測時間。未宣告 native evidence
或沒有 saved origin 時 metadata 為 None；新 Run 清舊 evidence。Native Load 保留檔案的完整歷史 metadata，不製造 acquired 事件或 Save confirmation。

RunInputs 固定 admission 的 ParameterContext，RunPaneState 隨 result 保存同一引用。
晚開 FIT、interactive factory、Post 與 WB 讀這份 result source，不讀後來的 tab binding。
Native Load 保留檔案 metadata。在相同 opened entry，Load 一次讀回歷史 point，
固定檔案的 alias 名稱供分析與 WB。值是 Load 時的持久值，不重建未保存的原始參數版本。
跨 entry、EMPTY 或無 point 時，Load 不提供分析來源，也不回退全域 alias。
Context-free analysis 仍可用。Save storage 獨立固定在 Load request 時。
Load 不新增 Run token 或量測歷史。
純計算來源為 None，不回退到 session；新 Run 與 clear 清除舊來源。Analyze／Post-analyze 不攜帶 predictor。Predictor 是獨立查詢工具，不自動參與實驗。

## 局部設計與閱讀入口

此 app 的 View（Qt UI）與 `RemoteControlAdapter`（`remote/`）共用可用的
application commands，但 capability 驗證尚未覆蓋全部入口。`GuardService` 對
load、save、analyze、writeback 提供 `check_*`，成功返回 None，失敗 raise
`GuardError`。Run admission 檢查並捕捉 `RunInputs`，不把資料型別當 proof。
analyze 不檢查 adapter 的 analysis capability。操作期間的 busy／硬體互斥
由 owning service 或 app-local `OperationGate` 處理。Inputs 不需釋放，lease
須在終局釋放，handle 只追蹤 operation，不證明持有硬體。詳見
[Operation ADR](../../../../../docs/adr/0066-operation-lifecycle.md) 與
[capability draft](../../../../../docs/adr/draft/gui-adapter-capability-guards.md)。

Library entry 的開啟、提交與取代由 controller 交給 session 的
`LibraryEntryForm`：controller 開啟時捕捉 parameters 與 arb 名稱，表單由呼叫端
持有，寫入與捨棄不經 controller。`context.ml_edit` 經 controller 的
`edit_library_entry` 走同一個表單：整批寫入一次，只有 Valid 才寫 library。
Library 新建從 `Registry.templates` 讀 `TemplateDef`。Controller 的 `list_templates`
保留 role discovery 的欄位與順序，`create_from_template` 以固定 capture 建立
session 的 `new_library_entry`，驗證 Spec 後提交 put。Expr seeds 求成初值，
Slot 已綁定時保留 child library key。換範本建立新 State，不重設既有 State。
LibraryEntryForm 由呼叫端持有，關閉時丟棄，不另建 editor session。FormEditor 的
建立與卸載只管理 view，不取得 state 的提交權。
Tab cfg 由 `Session.cfg` 的 `TabFormResource` 擁有。Open／Reset 捕捉來源，
preview、result 和 Run 都使用這份來源。Input、來源與 guard 不交給 view 管理。
Writeback 的 module proposal 是持久 `ModuleSpec`；draft 為每個 module item
持有一個 session 的 `ModuleEntryForm`，固定在建立時的 capture 與 arb 名稱。
Preview 與 apply 都取同一個 entry form 的 Spec，不另建編輯樹。
共用編輯契約見 [Cfg ADR](../../../../../docs/adr/0065-cfg-editing.md)。
分頁 Form 契約見本頁 Config model。

`State` 保存可觀察的 app 資料，`ContextService` 寫入參數容器／library；services
依用途讀 owner 的 read contract、單向呼叫 command 或訂閱已提交 fact。
版本由資源 owner 發布，不以每次 emit 必然 bump 推導：
`SessionState.refresh_device_info_cache()` 在 driver info 與快取相同時不 bump，
不同時 bump device version，caller 再發布變更。這些 counters 描述來源與結果替換，
不授權寫入。Tab cfg edit／Run 不收 ref。Remote adapter 在 GUI owner thread 檢查
當下 holder，再呼叫 domain owner。明確 readonly methods 允許 observer；其他
methods 要求 holder。Authentication、domain guards 和 result provenance 仍適用。
MCP 不保存第二份 authority，不隱藏預讀或自動重試。Timeout 或斷線後，caller
先核對現況，再決定下一步，不能假定命令沒有副作用。
GUI 事件與 service 協作見 [GUI ADR](../../../../../docs/adr/0067-gui-application.md)，
wire guard 與 off-owner await 見 [Remote README](remote/README.md)。

`AppPersistedState` 是記住的 session preference 與 session 的選擇性投影
（磁碟 slice 沿用 `startup` 名稱），不是整個 State 的序列化。
`State.preferences` 保存下次預填值，面板寬度由 `Controller.get_left_panel_width()`
提供；restore
不自動連接 SoC 或套用 active context。`WorkspaceService` 處理 session
capture／apply，resource 還原兼容的 Form input；`SingleFileCaretaker` 只
處理單檔 I/O，不認識 State／cfg。關閉等待的 timeout 並不證明 cleanup
完成。保存邊界見 [Persistence ADR](../../../../../docs/adr/0063-persistence-ownership.md)，
關閉與取消見 [Operation ADR](../../../../../docs/adr/0066-operation-lifecycle.md)
及 [operation draft](../../../../../docs/adr/draft/operation-lifecycle-boundaries.md)。

## Qt control identity

`Controller.control_identity` 是 holder／lock／online 的唯一 owner。MainWindow 的指示燈
透過 `gui.widgets.control_indicator.ControlIndicator` 投影它，點擊直接委派 `toggle_local`。
Remote 時只停用寫入區，不停用主 root、tab 檢視與 canvas。每次 domain interaction refresh
同時套用 holder 和既有 busy/readiness 條件，不因背景事件自動收回。GUI 檢視 tab 不改 Remote
holder 的 application active tab，關窗先顯示含 holder label 的確認，再沿既有 shutdown 流程。

MainWindow 是唯一 Qt authority 訂閱者。`set_writable` 更新主窗與目前 tabs，
新 tab 建立時套用當下狀態。Tab 與 interactive frontend 只接收可寫狀態。
Local modal 期間，Qt remote adapter 拒絕 owner 取得，回 `local_busy` 與「Local 使用中」。
Chooser 使用非 native option。Dialog 不訂閱 control，也不在寫入前重查 holder。
Inspect、Devices 與 ARB Waveforms 的查看入口提供靜態 read-only mode。
Inspect 不開放 Add 或 library mutation，但仍可開啟唯讀 ARB Waveforms。
Inspect Delete 先固定 `ParameterContext`，
再用 `Controller.apply_library_change` 刪除該 target，不因確認期間 context switch 改寫新 point。
Add 與 RunLoad 的 finished 仍退休單次輸入，晚到提交不能重新寫入。
只有量測 Stop 和 device setup Stop 在 Remote 收回 Local locked 並請求 cancellation。
收回本身不取消 operation。
Mounted `InteractiveFrontend.set_read_only(read_only=...)` 必須停用 plugin command controls 和
pointer editing，但保留 figure viewing。FluxPickFrontend 同時保留 alignment busy gate。

## Package Boundaries

- `form_adapter` 從 core cfg 與作者樹推導 Form。`make_form(ctx)`、範本與實驗初值由使用者套件宣告，舊 builder、seeds 與角色 factory 表已退休。
- `flux_pick_plugin.py`、`flux_pick_frontend.py`、`interactive_flux_pick.py`：measure 選線的終態結果與 click-follow frontend。Caller 傳 FluxPickInputs 和初始數值，不帶具體實驗或參數容器。共用數值及 single-flight 機制仍由既有 kernel/plugin 擁有。
- `adapter/`：framework-facing contract、request/result/analyze params、
  adapter validation 與 protocol signature 所需的 session vocabulary。Writeback 型別由
  `writeback/` 擁有，adapter 公開入口再匯出。ModuleWriteback proposal 直接使用持久
  ModuleSpec，不帶 editor identity。
- `writeback/`：parameter／module／sample proposals、draft、service、control 與 Qt review widget。
- `services/`：app service layer。Service 依賴 ports，不直接 import sibling service
  implementation；package `__init__` 只做 lazy public re-export。
- `state.py`：tab/device/pane/path/version-table SSOT 與主線程 mutators；固定的
  Run、Analysis、Post-Analysis、Save pane 各自擁有自己的 resource。`running_tab_id`
  是唯一 run ownership 狀態，tab interaction 的 `is_running` 由它投影。MainWindow
  also projects that identity directly onto exactly one top-level tab as a compact blue
  `●` marker while Run is active; Run start/terminal EventBus reactions and tab insertion
  repaint from State, so the marker is never persisted or cached as a second busy state.
- `ui/`：Qt widgets、MainWindow top-level façade、capability-driven `ExpTabWidget`、
  feedback widgets；writeback view 由 `writeback/` 擁有，generic cfg form不屬於app package。
  `ExpTabWidget` owns capability-driven left subtab composition (fixed order Run |
  Analysis | Post-Analysis | Data | Guide, with optional Analysis/Post only when
  the adapter declares the capability) and per-pane `FigureContainer` routing
  (stable identity per (tab, pane) for the widget lifetime; refresh never replaces
  the container; busy tabs cannot close or rebuild, so the captured worker target
  outlives the operation without a lease). It receives tab actions through a narrow
  `TabActions` port with pane-qualified writeback (`apply_post_writeback`);
  `MainWindow` adapts those actions to top-level handlers. Run 使用共用 FormEditor，
  None／keys／Inline 合併下拉，Expression 在文字欄位右鍵切換，Advanced 預設折疊。
  Run action row is status-free with Reset 20% / Run 80% of available width；
  idle Reset與Run同高，Reset為一般按鈕、Run為blue primary，running時隱藏Reset並讓red Stop佔滿整列，
  settled後恢復20/80（A5）； Analysis uses an app-local single-column 13 px ledger with
  whole-header folding for `Analysis parameters` and `Writeback preview` and a
  full-width `Analyze` immediately after parameters and before Writeback preview
  (A6). Post-Analysis uses the same ledger, form font, full-width primary button
  (`Run Post-Analysis`, below its gate hint) and `Writeback preview` section, so
  both analysis steps present alike; a known parameter type without fields shows
  the form's no-parameters hint in either pane. Presentation
  Modules do not invoke operation services directly. Each `WritebackWidget`
  is pane-bound (analysis vs post_analysis) and edits/applies its own opaque
  draft via `Controller`/`WritebackControl` pane-qualified forwarding, while
  `WritebackService` remains stage-agnostic and fixes each draft's destination
  and role resolution at creation. Its preview returns per-item `WritebackValues`
  with the actual path, latest stored value and capture-evaluated proposal. Both
  panes obtain these facts through `writeback_control`; Qt does not resolve paths.
  Qt and remote preview also display the proposal's existing stderr. Qt rows use
  resources-provided static unit labels without conversion; ext and fields
  without static units have no label. The widget is a
  compact unified ledger: draft-owned unapplied items以粗體`target*`與`* = not applied`
  legend呈現，成功寫入的items改為一般字重`target`，retarget或內容修改後回到unapplied；
  apply/edit的app-owned fact會讓本地與remote入口都觸發同一writeback重投影；
  centered Current → Proposed columns on a shared-background continuous-boundary
  panel (white rows with bottom dividers), and equal 56×26 Edit/Copy actions。
  Scalar parameter 與 editable module items 使用 Edit；non-scalar parameter 的
  current/proposed headings使用一致的bounded summary（例如`3 × 3 matrix`）。small
  proposed matrix另外顯示read-only matrix view，current value維持summary-only；Copy會把
  complete proposed JSON放入clipboard。其它arbitrary long values只顯示bounded summary，
  因此ledger never widens. The widget owns a ~450 px breakpoint: wide rows stay
  single-line, narrow rows reflow to target/action above centered
  Current → Proposed. Apply Selected sits above the applied-state legend and
  bordered ledger, close to the analysis controls. The ledger hugs its rendered
  rows; long content grows naturally and delegates
  vertical scrolling to the existing Analysis pane rather than owning a nested
  scroll/cap lifecycle.
  `RenderHost` is pane-aware (run | analysis | post_analysis) and the worker
  captures its pane's container at start — switching the visible subtab never
  retargets the worker (ADR-0067). Run terminal reactions refresh canonical
  presentation without selecting a subtab; Analysis remains an explicit user
  selection. `ExpTabWidget` delegates the Data pane to an
  `ArtifactSaveCenter`。State 在 Run、Analysis、Post-analysis 的成功 terminal
  宣告目前可保存的 sections。每個 artifact 可以包含 data、具名 Figure 與窗格 JSON。
  沒有結果就沒有佔位 artifact；沒有圖的成功分析仍可以保存 JSON。
  Session 的 Qt-free `ArtifactTracker` 是保存狀態的唯一來源。
  TabService 與 Qt 讀取相同的 section/member/output 快照，不在 getter 計畫路徑。
  新 Run 清空全部 sections，Primary replacement 清除舊 Post。
  Save 按鈕提交整個 artifact ref，Save All 提交全部宣告，包含 Saved。
  每個 output 列顯示唯讀路徑、最新 status 與 error，刷新保留已有 widgets 與文字選取。
  `Open folder` 開啟該列最新路徑的父目錄，不建立目錄。
  GUI 不接受手動保存目的地。Comment 變更只把已有成功輸出的 run artifact 標為
  Unsaved changes。部分成功顯示 Partially saved，重存失敗不沿用前次成功狀態。
  關閉警告只看 measurement data 是否完整 Saved，不因圖片或 JSON 未存而阻擋。

  GUI 直接呼 SaveControlFacet。SaveService 讀 Run pane 的固定 source storage、
  歷史 RunMetadata、窗格結果與 result_params，再用一次 shared saver 保存全部選集。
  Data 寫 canonical `runs/<run_id>/data.h5`，每次保存另取新的 Labber filename。
  Data mapping、Figure 編碼、檔案 I/O 與一次 saved ledger event 都在 owner thread
  同步完成。shared saver 每個 member 編碼一次，隔離逐輸出失敗。
  Save 不持硬體 lease，不建立 worker、handle 或 save-busy phase。
  State 提交完整 report 後才發布同步 completion fact，再回傳同一份 SaveReport。
  Qt 顯示本次輸出的 path/status/error，包含實際 Labber path。
  Mapping／ledger 錯誤直接傳遞，不偽造 report 或回滾已寫文件。
  保存不產生 acquisition identity，也不重讀 live point。
  RPC／MCP 直接投影回傳報告。Native Load 由框架讀取 StoredRun 並比對 experiment tag，
  再交 adapter.restore(stored)。框架不檢查 record.cfg，歷史 metadata 直接取自 StoredRun。
  Load 在 adapter 呼叫前固定目前來源與不帶 alias 的 Save storage capability。
  Analyze/WB 另用檔案 point 與 resolved alias names，一次透過 context owner 讀取。
  歷史 point 的檔案、schema 或 kind 錯誤在發布結果前傳遞，舊結果不變。
  Save 在 mapping 或輸出 I/O 前比對 native 歷史 entry 與這份 result source storage。
  跨 entry 時顯示雙方來源並拒絕。同 entry 的不同 point 仍保存，沿用原 run_id
  與資料自己的 point，覆寫原 run 目錄，saved event 保留原 point。
  Tab birth capture 與後來的全域選取不參與路徑計算。Save 不要求 live ACTIVE context。
  Load Data 的 RunLoadDialog 由既有 SaveControlFacet 查詢 opened entry ledger，
  按 run_id 合併 acquired／saved，保留最近成功 native path。未存 native 的列不可 Load。
  Point／experiment／Recorded at 篩選只看 ledger，Recorded at 不冒充量測開始時間。
  清單不讀 data.h5 或檢查輸出是否仍存在；File 選檔與清單選定後都交既有 native Load。
  Ledger 或附檔錯誤明確回報，不偽裝空清單。
  Analysis/Post panes no longer own image-path/Save Image; Run's live figure
  remains view-only (display + screenshot, no canonical Save). Data's right pane
  is a `DataFigurePreviewGallery` Variant A responsive rail: capability-declared at
  construction (Run always, Analysis/Post only when supported), viewport-driven
  mosaic reflow — narrow single-column vertical vs wide three-card Run-left
  spanning two rows with Analysis/Post on the right, two-card side-by-side,
  and single full-width at the two-minimum-width-cards-plus-spacing breakpoint
  (gallery's own viewport, not window, no persisted toggle) — scrollable cards
  with named empty/unavailable states, raster-only presentation cache (pixmap/text)
  fed by an injected `Figure -> PNG bytes` adapter — production uses the Data
  Preview renderer with saved-image logical geometry and size restore — per-card render caches the
  original pixmap and aspect-fits with `KeepAspectRatio` and smooth transformation
  inside the image viewport without cropping; per-card failure is isolated and
  logged without blocking other cards or save controls, and no Figure/canvas
  ownership, timer, or per-draw subscription is introduced. `ExpTabWidget` remains the sole `FigureContainer` and
  current-figure authority: Data activation and Data-visible
  prepare/clear/show lifecycle refresh the gallery snapshot, while
  Data-invisible mutations postpone PNG rendering until the next activation;
  the gallery never attaches or reparents a canvas. Subtab routing keeps
  Run/Analysis/Post on their source stacks, Data on the gallery, and Guide on
  its placeholder. Top-level orchestration invokes behavior-oriented tab methods
  for result presentation, plot hosting, interactive-widget lifecycle, figure
  reads, and persisted panel geometry; the tab does not expose its Qt containers.
- `remote/`：與 `ui/` 平級的 GUI-process driving adapter，將 RPC 意圖交給
  application commands/queries。`remote.method_specs` public import path 不載入
  Qt-bound service code；MCP bridge 不在本 package。
- `driven/`：measure app-local Qt/liveplot driven adapters；與 `adapter/` 的 experiment
  framework contract 分開命名。

Shared layers:

- `zcu_tools.gui.form`：Qt-free input、FormState、schema、preview 與 result。
- `zcu_tools.gui.form_adapter`：core cfg／持久 Spec 推導、FieldMeta 呈現投影與 key-or-inline 轉換。
- `zcu_tools.gui.widgets.form`：FormEditor 與共用 value-widget helpers。
- `zcu_tools.gui.session`：context、SoC、device、project settings、predictor、operation
  handles、operation runner、progress/shutdown service、shared dialogs。
- `zcu_tools.gui.remote`：NDJSON RPC endpoint、framing、wire errors、router base。
- `zcu_tools.gui.plotting`：explicit figure host、container 與 export。
- `zcu_tools.mcp.measure`：agent-facing MCP policy layer and tool surface。

## Composition Root

`MeasureGuiBehavior` is the process-runtime behavior for the shared
`gui.runtime` launcher seam. It assembles `State`, `Controller`, `MainWindow`,
persistence caretaker, and the app-local `RemoteControlAdapter`
without owning process policy such as logging, rendering initialization,
`QApplication`, control option construction, or exit-code handling. The
standalone launcher is the process entrypoint; this module does not expose a
second `run_app` path. After the window is shown, `after_show` opens the same
Setup dialog through `MainWindow.open_dialog`, so a toolbar click focuses that
instance instead of building another.

The launcher still owns the experiment-adapter composition boundary by passing a
registry factory into `MeasureGuiBehavior`. The factory imports the injected
user catalog only after `gui.runtime` has configured logging and the
pre-Qt plotting policy.

`build_app_services()` constructs the app-local services and injects their driven
ports. `Controller` is a facade over the service bundle; UI and remote code use
the controller for app-specific workflow and the exposed session control facets
for setup/context/device/predictor/progress domains.

Composition 建立 runner／progress 後交給 `build_session_services()`。
Run、Analyze、PostAnalyze 從回傳的 `SessionServices.runner` 取用同一 lifecycle
機制。Operation control 與 app bundle 從 `SessionServices.progress` 取用同一
progress owner。這個入口不改 operation policy、hardware exclusion、取消或
owner-thread delivery，也不另建 infrastructure bundle。

Measure launcher 讀取使用者套件的具名 preset，以 `alias_presets` 經
`MeasureGuiBehavior`、`Controller` 與 `build_app_services` 注入 default rows。
Behavior 的公開 `alias_presets` 欄位保留完整名稱與有序表，再經 `MainWindow` 與
`MainDialogRegistry` 交給 Inspect 的 Load preset。
框架不宣告領域初值，也不讀 preset 檔。Caller 省略或傳 None 時建立空表。
顯式 mapping 必須含 default，否則 raise ValueError；default 的空 tuple 合法。
ContextService 保留 authored softlinks 與空列，在第一份非空元件 snapshot 以
role registry 的唯一候選填 primary 空列。每次開 entry 有 `aliases.yaml` 就載入。
無檔才重設 preset，重新啟用 primary 初始化。已保存的空表與空列不自填。切 point 不重讀檔或換表，明確
提交表後不再初始化，直到下次開 entry。Preset 的 flux_dev 空列不猜裝置，也不從 cache 填值。
參數捕捉經 `read_context(aliases=...)` 固定當次 point、library、storage 與有序表。
GUI adapter 不宣告角色，捕捉不從 GUI focus 或 adapter 的 via 推論 alias。
Alias 表在共用 Inspect 的 Aliases 頁（`session.ui.alias_panel.AliasPanel`）編輯，工具列沒有獨立 Aliases 按鈕。
Apply 先驗整表，再由 entry owner 原子保存，成功才安裝 memory 並發布 event。
EMPTY 時只改 memory。失敗保留舊表、舊檔與 draft，頁面顯錯可 retry。
關閉 Inspect 時未存 draft 走 Save／Discard／Cancel，Discard 不撤回已 Apply 的內容。`set_alias_rows` 仍只改 memory。
Tab 不顯示角色列，也不持有自己的 alias 表。

App-local driving-adapter facets mirror the shared session control pattern.
`TabControlPort` / `TabControlFacet` expose the tab resource surface (lifecycle,
active/running identity, tab read model, cfg resource lookup)
by composing `WorkspaceService`, `TabService`, `State`, and `EventBus`; remote
tab handlers use this facet instead of the giant `Controller` surface.
`RunAnalyzeControlPort` / `RunAnalyzeControlFacet` expose the run/load/analyze
operation surface (run start/cancel, result load, analyze/post-analyze start and
result reads) by composing the operation services, guards, tab read model, and a
render-host provider. Remote run/analyze handlers use this facet instead of the
giant `Controller` surface. `OperationControlPort` / `OperationControlFacet`
expose the op-agnostic handle/progress surface used by generic `operation.*`
handlers, including device setup handles. `SaveControlPort` / `SaveControlFacet`
以 `GuardService`、`SaveService` 與 `State` 組合 native Save 與 annotation commands。
Qt 與 remote 直接呼 facet；remote 投影同步 native Save report。 `WritebackControlPort` /
`WritebackControlFacet` expose persistent writeback draft read/edit/apply by
composing `GuardService`, `WritebackService`, `State`, and a resource-version
provider; remote writeback handlers use this facet instead of the giant
`Controller` surface. Qt and remote
writeback forwards are pane-qualified; no flat writeback forward remains.

Inside the Qt view, `MainWindow` remains the top-level View / `RenderHost` facade
while `MainWindowEventCoordinator` owns EventBus subscription and pane-specific
payload routing (ADR-0067). The coordinator speaks to `MainWindow` through a narrow
host protocol: it decides which refresh sequence a closed domain fact requires,
but the window keeps widget ownership and concrete rendering methods. Producers
emit only closed facts (run/analysis/post lifecycle or committed resources) — no
widget refresh flags — and the coordinator owns the ordered fact-to-reaction
matrix, fetching at most one `TabSnapshot` when a reaction needs it.
Operation start clears only the affected pane's presentation while retaining the
previous canonical pane for failure recovery; success shows the new pane's figure
and draft, failure/cancel restores the retained pane (primary failure restores
primary then post). Save/Guide show a placeholder and never borrow another pane's
figure. Local analyze/post parameter edits keep synchronous State commit timing.
Save 路徑由已提交的 artifact outputs 投影，不是 editor draft。Save 同步完成後的 domain facts
更新同一 Data pane 的 status 與 path，不替換整個 pane。
Analyze forms commit `QLineEdit` changes on `editingFinished` so partial text does
not trigger interaction refresh; choice, checkbox, and numeric controls retain
immediate value-change commits. The shared cfg widget layer owns this signal policy.
`MainWindowToolbar` owns the top toolbar widgets and slash-grouped new-tab menu;
it reports selected actions back through a narrow `MainWindowToolbarHost` surface
instead of reaching into `Controller` directly. `main_window_activity.py` projects
run activity marker text, color, and tooltip without owning Qt widgets.

Key ownership rules:

- Caller-correctable app/service failures由producer透過`ExpectedError`明列invalid-input或
  failed-precondition category；shared remote dispatch統一投影generic category。handler只保留
  request coercion與structured/domain-special wire policy，不擁有分類registry。
  ordinary/provider/persistence/invariant failures保持unexpected並保留controller traceback
  （ADR-0068）。
- `ContextService` owns GUI parameter and ModuleLibrary commands. It validates
  candidates and commits through the resource owners before publishing changed
  facts and bumping `context`; failures do not publish a candidate.
- `State` owns tab/device/pane/path resource state and resource versions. Pane swaps
  happen on the owner thread and return retired resources for post-commit cleanup.
- `GuardService` owns the implemented domain checks. Qt and remote controls
  call `check_save/load/analyze/writeback` before the service command on the
  same owner thread. Run uses `capture_run_inputs` once for the exact cfg and
  observed inputs. Analyze capability and post-analyze checks remain at their
  existing entry points (see the GUI capability draft).
- `OperationGate` is the app-local thin wrapper over the shared
  `RunBlocksHardwareGate` hardware exclusion policy。active lease另投影captured
  origin、domain note與duration；`state.hardware_gate`是read-only internal RPC，
  MCP 需要時透過 live catalog 讀取。
- `OperationHandles` owns async handles, cancellation hooks, and feedback/stop
  channel state.
- `OperationRunner` owns the generic operation lifecycle; each operation supplies
  an `OperationSpec` policy and narrow write ports. Terminal policy exceptions
  are contained in the shared runner so handles settle and exclusion leases release.

## Experiment reload

`ExperimentReloadService` 擁有 RAM-only snapshot 與重載／恢復狀態，透過注入的
`ExperimentCatalogLoader` 載入新 registry，透過 Workspace 的正常 close/apply seam
重建全部 tabs。Controller 只轉接，工具列與 MainWindow 負責整批 destructive confirmation
和 failure/recovery presentation。Registry 的持久範本、hardware context 與 framework 不重建。

`catalog_loader.py` 提供通用 `SourceExperimentCatalogLoader`。組合根用 `SourcePackage`
宣告 framework 與使用者 source 目錄，並指定 reload、preserved 與 catalog module。
Loader 不猜 namespace 或套件位置。Preserved prefix 優先於 reload prefix，其餘 source 也固定。
固定 source 的內容變更或刪除要求重啟。新增尚未 import 的固定檔案不阻擋重載，但重載中
import 新的固定 dependency 要求重啟。Loader 在所有來源保留固定 module identity，且每個
prepare plan 只能使用一次。它在重載失敗時清除部分 owned modules，不回滾任意 import side effect。

Bounded source inventory、prepare/load、fixed-source restart 與共同 token/error 型別由
`gui.catalog_loader` 擁有。Measure loader 必填 `current_registry`，新建 detached
candidate 時使用其 templates，不修改或自行發布 current registry。
Controller 的 recovery 與 UI 政策不變。

依賴重載是 best-effort，不全面保證 deferred/dynamic import 的一致性，也不禁止函式內 import；
可重載 code 的 import 不得操作硬體或啟動背景工作。此取捨不放寬資料丟棄確認或 lifecycle 保護。

Retry 的 prepare/load 皆保留 typed error disposition；要求 restart 後不再提供可 retry 狀態。
Shutdown 暫停 experiment entries；settle 後的未保存資料確認若取消，MainWindow 呼叫
`Controller.abort_shutdown()` 恢復入口，不自動重啟已取消的 operations，也不重新啟用失敗的
catalog。開始 shutdown 若拋錯，Controller 恢復原 gate 狀態並保留例外。

`Controller.close(timeout_ms=5000)` 是 Qt owner-thread 的 terminal 收尾入口。
它永久停止背景 admission，再 drain worker 與 queued owner deliveries，逾時 raise
`TimeoutError`，保留 owner 後可重試。Production 關窗與 Controller-owning fixture
都使用此入口。`begin_shutdown` 仍負責取消 operation，不代表硬體斷線或 RF safety。
MainWindow 在 close 成功前不 persist、不停止 remote、不釋放 View。逾時留窗詢問
「再等一下／強制關閉」，只有明確 force 才跳過 drain。再等或 dismiss 在下一個
Qt turn 重試，重複 GUI／RPC request 不代表 force。

Reload 在 owner thread 同步執行，不 pump Qt events；進行中 handle 與 tab busy flags
共同阻止 reload。Save 在同一 owner thread 同步完成，不建立 handle。共用 `ExperimentAccess` 阻止 local/remote
experiment driving facets 在切換時重入。確認等待期間 tab identity、resource versions 或 run
result 改變會使確認失效。重新建立的 tabs 使用新 id，舊 RPC locator 不可沿用。

Snapshot 只保留既有 `PersistedSession` 的名稱、cfg、順序與 active index，沒有 result、figure、
analysis params 或 path overrides。Reload 不呼叫 caretaker，不新增 checkpoint 檔案；正常
跨重啟 persistence policy 不變。Import cache 的一般 bytecode 清理不是 session persistence。
Catalog load 失敗時保留 snapshot、停用 experiment entry，受控失敗允許 Retry reload；fixed-state
integrity 無法確認時要求重啟。Partial restore 保留 skipped cfg，Retry skipped tabs 只附加仍未
成功的 entries、不重複還原或改變使用者選取；下一次完整 reload 需確認丟棄先前 skipped entries。
所有 RAM recovery 在 process 結束後遺失。

## Run / Analyze Workflow

1. A tab is created from a registered experiment adapter.
2. Tab 持有獨立於 widget lifetime 的 `TabFormResource`。
   Run 以 `FormEditor` 呈現同一個 FormState；Analysis
   renders its params through the app-local 13 px ledger with whole-header
   folding and a full-width `Analyze` immediately below parameters.

3. `Controller.start_run(tab_id)` accepts current Valid cfg without an expected ref.
   Qt and remote use the same admission path.
   `GuardService.capture_run_inputs` accepts fixed cfg and captures
   RunInputs with AcceptedConfig provenance, fixed ParameterContext and storage.
   Cfg.dev stores role-to-device names, not State-owned settings or setup patches.
   Cached observations used for cfg sources or remote guards are not the Run snapshot.
4. RunService fixes started_at and run_id before worker submission. After the hardware
   lease and before adapter.run, the Run framework reads get_info once from every
   connected device. Any read failure rejects Run without entering the adapter/core,
   writing settings or falling back to cache. No devices produce an empty snapshot.
   Run does not precheck whether cfg.dev names exist. Cfg format and required unit
   validation remain. RunService creates a fresh RunContext with borrowed drivers,
   operation plots and one StopSignal, without instrument setters.
   The adapter passes this context to the core. Core selects context.devices[cfg.dev[role]]
   when using a device; a missing name raises an error naming that device.
   Core uses experiment helpers; flux cfg requires flux_unit. Only progress remains ambient.
   Driver lookup does not refresh accepted cfg. RunMetadata.device_snapshot retains the
   pre-experiment settings even after a sweep. Native Save stores them at `/device`,
   not in tab preferences, and does not read hardware. A failed Run with no data
   publishes no metadata or ledger event. Usable stopped data retains its evidence.
5. `BackgroundRunner` executes blocking work off the Qt main thread and marshals
   terminal callbacks back to the main thread.
6. RunService uses State directly for busy checks and atomic result/evidence
   publication. AnalyzeService retains its narrow `AnalyzeStatePort`.
7. Writeback items are generated from analysis results. Module proposals are
   persistent Specs, edited through each draft's shared `ModuleEntryForm` and
   `FormState`. `WritebackService.create_draft` captures
   the destination ParameterContext and its fixed role names. Preview facts
   expose the actual destination, latest stored current value and complete proposal.
   Scalar parameter rows show values and stderr with static schema unit labels
   from resources; ext and fields without static units show no label. Labels never
   convert values. Draft-local retarget refreshes the destination facts and unit.
   Module items show bounded change summaries without units. Their `Edit`
   dialog mounts `FormEditor` directly on the draft's state and captured env;
   `Apply as:` retargets the same item. Invalid State input survives Close and
   refuses Apply. Unparseable local text also marks the row Invalid and locks
   Apply while open; Close drops only that local text and shows the State's
   result, without writing the library. Field writes go through the existing
   pane-qualified writeback guard. A refusal keeps the user's text in the open
   editor and displays the guard's reason without changing the shared state.
   A host refresh after module validation fails keeps Invalid rows and reads
   valid items through the service's single-item preview. MainWindow presents
   expected Apply refusals without discarding the draft; other failures propagate.
   The existing busy and Remote enable gates remain unchanged.
   Parameter and sample proposals retain their separate write contracts.
   Primary and post workflows own proposal timing;
   the Writeback service remains stage-free. Sample proposals name a concrete
   component and producer JSON fields. GUI adds dev_value/dev_unit from the
   Run-fixed $flux_dev binding and device snapshot. Missing binding, device,
   value or unit omits the sample with a public unavailable_reasons entry.
   Preview and write receipts read the latest entry JSONL row for that run.
   Only adapters whose Run does not change that device provide sample proposals.

   Primary FIT and interactive operations capture the source `RunMetadata.run_id`
   at start and pass it to the opaque draft. Post captures the Primary draft's
   identity at operation start and passes it to its new draft. A missing Primary
   identity stays None, even if live Run metadata or Save has an id. Accepted
   ledger events and library provenance use that identity whether or not Save
   occurred. The identity does not imply a saved file; consult saved events for
   persistence. Native Load retains the original identity, and analysis without
   native metadata uses `None`.

- `load_tab_result` 接受 RestoredRun 的 opaque record 與明示 cfg_issue。
  BaseAdapter.restore 委派 core.from_stored，不讀檔案或目前表單。cfg_issue 為 null 表示 typed cfg 可用，
  字串則解釋不可用原因；缺 cfg 仍可成功載入數據。
  原始 cfg 保存在 native 資料檔，透過公開 `datafile.load_run_data` 離線讀取為 `StoredRun`。
  GUI 不提供原始 cfg 檢視入口。
  WB 不從 Load-time calibration 補 pulse module、改槽或 pi_product。
  Load 保留表單原輸入與固定來源，不要求 draft 有效。
  Run/Load 先提交 primary result。Analysis preparation 失敗不撤回它，只報 diagnostic。
  Named figures 和各 pane 的 presentation lifecycle 分開管理。
The Guard and LoadService both enforce the adapter's import-validated
`capabilities.load_data` gate.

Run and Load retain committed results even when adapter analysis-parameter
preparation fails. `TabService.prepare_result_analysis` owns the capability check
and returns typed readiness or an error after logging the failure. Both workflows
publish their committed content fact once, with downstream panes cleared and
successful params already installed. Run sends a separate diagnostic to attached
views without changing its finished or cancelled outcome. Load returns
`analysis_error` and `has_analyze_params=false`; Qt warns after confirming the load,
and remote returns the same outcome. `tab.open_file` retains the new tab on this
partial success. A genuine load or run failure keeps its existing failure contract.

### Pane-owned lifecycle

`Session` is the aggregate root and its fixed pane carriers are the resource owners:
Run stores only the run result/source. Analysis and Post-Analysis each store params,
result, a named `Plots` collection and an opaque writeback draft (with S2 baseline
snapshot). Save stores the annotation; ArtifactTracker stores each member's latest
output paths/status/error. Run live figures remain view-only and
are not stored in State. Writeback baseline is a display-only draft-creation
snapshot；同一opaque draft另擁有per-item applied state，只有成功write包含的items才標記applied，
selection本身不改狀態，retarget或內容修改會重設；同kind items不得指向重複destination，
避免batch覆寫卻誤標applied。狀態不跨draft/process持久化，也不提供
concurrent-write detection或apply-conflict policy。

Analysis/Post 的 artifact 記錄目前 result／Figure 各 member 的最新保存輸出，
不追蹤 artist 或視圖的 dirty 變更。新結果從未保存開始；同步 Save 返回前
提交本次完整 report。重存失敗取代前次成功輸出狀態，不沿用舊 success 清單。
至少一個輸出成功才確認 run_id；failed resave 保留已確認的身分。
Comment 變更把已有成功輸出的 Run artifact 標為 Unsaved changes。

Analysis/Post result services prepare proposals, named plots and drafts before calling one
owner-thread State swap. The swap returns every retired pane resource; services tear
down retired drafts only after commit and never roll back a committed pane when cleanup
fails. A failed proposal/editor build leaves the previous canonical pane intact.
Primary analysis replacement invalidates Post-Analysis, Post replacement leaves
Analysis untouched, and a successful run/load clears both downstream panes.
Each successful analysis swap records its source operation token and a detached
copy of its actual input parameters in the pane. Editing the next analysis's
parameters does not change those result inputs. The read model exposes both;
operation-bound result reads return the inputs, while image exports can require
the same provenance. Replacement clears or updates both with the result.
Failed or cancelled analysis retains the previous pane and its provenance. Tokens are session-local;
this metadata is neither a result history nor persisted recovery state.

Run commits its operation token with the result, including usable partial data after
cancellation. Starting another Run clears both; Load publishes no Run token.
Raw saving and Primary analysis may require this source token. State owns the
identity check, and the control facets apply it before capturing inputs or changing
drafts. Remote analysis also checks before following the pane. Superseded sources
fail instead of consuming the latest result. Ordinary Qt operations still use the
current result, and remote observation guards remain independent.

Post analysis can require both its Primary analysis source and its Run source.
The remote handler checks them before following the Post pane. The control facet
checks them again before creating plots or capturing inputs, in the same owner
command. Omitting the source tokens retains ordinary current-result behavior.

State and `TabSnapshot` expose only the explicit Run, Analysis, Post-Analysis, Save
and path carriers; there are no flat tab result/writeback/path projections. Callers
name the pane they consume. Operation-start request/context inputs are captured and
reused by analysis and proposal hooks, without context-identity checks or terminal
active-context reads.

Data preview never owns pane resources: `ExpTabWidget` reads current figures from
the fixed `FigureContainer`s and pushes a transient `Figure` snapshot to
`DataFigurePreviewGallery` only on Data activation or while Data is visible;
the gallery renders to a 640×480 PNG using the same 12×9 inch logical canvas as
Save，restores the live size，holds only the raster cache (original pixmap) and isolates per-card failures with
aspect-fit scaling (`KeepAspectRatio`) that never exceeds the image viewport.
Viewport-driven mosaic reflow (gallery's own width vs two-minimum-width-cards
threshold) is presentation-only and never moves figure ownership, adds a second
canvas, or changes ADR-0067 reactions. No competing state owner is introduced.

## Tab lifecycle and ordering

Tab 不保存另一張 alias 表。Open／Reset 捕捉來源，下一次 Reset 前保持固定。
Reset 的單次 aliases 不保存；Run 使用顯示值並拒絕非空 aliases。
Run／Analyze／WB／artifact 保留原結果來源，不因目前 alias 改綁而重新解析。

New tabs are pure GUI configuration surfaces: creating one builds the adapter's
default cfg from the current context but does not start hardware work. The toolbar
therefore stays available while another tab is running; per-tab interaction state
and `OperationGate` still prevent starting a second run until the active run
finishes.

Top-level experiment tabs are movable. The visible order is synchronized back to
`State` through the controller/workspace lifecycle path, so `list_tab_ids()`,
remote tab views, and captured sessions all use the same tab order as the Qt tab
bar. Active and running tabs are identified by tab id, not visual index.

## Config model

`Session.cfg` owns one `TabFormResource`. The resource owns one `FormState`
and captures parameters, library and cached device infos on Open and Reset. Preview, result and Run use that capture until the next Reset.
Source events do not refresh existing forms. Device capture copies cached infos;
it never queries a live driver.

Adapters declare `Form(core_cfg, tree)` from the shared `zcu_tools.gui.form_adapter`. Core annotations define editable shape and
validation; the tree defines authored starting values, locks, hidden fields and
module defaults. The Registry binds the Form to its injected template catalog.
Form lowers ordinary input to core cfg, including optional extras through its named lower hook.
Ordinary strings remain literal; expression input uses an explicit envelope.
Nullable named device choices include None independently of whether a core field
has a default. Missing required input remains Invalid.

Frontends submit writes to the same resource. FormState applies a complete batch
atomically; rejection leaves no successful prefix. Reads do not poll providers.
Invalid input remains editable, including unfinished expressions. Run accepts only the current valid result and never uses an older
valid cfg. Busy, lifetime, context and SoC guards remain separate requirements.

`Reset(aliases=...)` captures one-shot named overrides and recomputes defaults.
The overrides change neither global rows nor the memento. Run rejects nonempty
aliases; None and an empty map use the displayed capture. Run metadata retains
that capture's point, roles and storage. Acquisition still takes its own live
device evidence after the hardware lease, as described above.

Notifications permit reads and reject reentrant writes, acceptance and lifetime
changes across tab resources. Subscriber faults log a traceback; the committed
input remains installed and other subscribers still receive the change. Close
revokes retained handles. Reopening creates a new resource and source capture.
Frontend attach and detach do not own resource lifetime.

The Qt Run panel mounts the shared `FormEditor` over `resource.editor_state`, the
same FormState lent for reading and subscribing, and passes `write=resource.write`
and `run_command=resource.run_command`. Editor input and right-click option
commands (such as a module key's Convert to inline) therefore take the resource
guards and publish to resource subscribers and the tab cfg version. Reset
installs a new FormState; the panel watches the resource and mounts a new
editor on that state. While the tab runs or analyzes, the panel only calls
`set_editing_enabled(enabled=False)`: inputs lock, but the form still scrolls
and its groups still fold. Detach unsubscribes
and detaches the editor before the tab closes.

Load replaces results, not Form input. Valid, invalid and unfinished drafts
keep their authored values and expression text, including extras. Load does not
validate the current draft or infer input from native cfg. The adapter owns
its record and cfg diagnostic. Raw cfg stays in the native data file and is
available offline as `StoredRun` through the public `datafile.load_run_data` API.
The GUI provides no raw cfg inspection entry point. Native Load's historical
analysis source and Save storage retain their separate Load-time ownership.

Workspace saves authored Form input JSON, including expression text. Restore
captures new sources and overlays compatible leaves through the resource,
reporting discarded paths. The app memento version changes with this input
format; earlier versions use the caretaker's default degradation without tab
migration.

GUI flux forms provide A, V and native units with initial A. This initial value
does not come from device info. Core and Notebook cfg require an explicit unit.

### Library、Inspect 與 writeback

Library／Inspect 使用 session 的 LibraryEntryForm，root 固定具體 Spec family。
Apply 直接交持久 Spec，保留 child key，屏蔽欄位由生成器預設補回。Name、未存提示、
Revert 與切換條目前的確認由共用 Inspect 管理。Parameters 頁仍使用 YamlValue／FieldSpec，
Modules 用 FormEditor，Aliases 是第三頁。Run／Remote 唯讀只鎖寫入，仍可複製、折疊與捲動。

Writeback module item 持有 ModuleEntryForm，Qt 與 remote 共用同一 state。
Form Invalid 或 view 有 local parse errors 時拒絕 Apply，關閉只丟棄未解析文字。
LibraryEntryInvalid 保留 row、selection、dialog 與 draft，由 presenter 顯錯；
非預期錯誤向外傳遞。Host 保留 busy、writable、result、context 與 holder 規則。
Selected Apply 先準備全部選中值，再由既有 owner append accepted 並依序提交，
首錯保留成功前綴，不提供跨檔交易。詳見 ADR-0008、ADR-0065。

## Operation Model

- Run, device setup, and SoC connect use hardware exclusion.
- GUI Save and Save All complete synchronously on the owner thread. They have
  no worker, operation handle or save-busy state. The existing SaveReport carries
  every attempted output, including failures; RPC returns its complete projection.
- Analyze and post-analyze use async handles but no hardware exclusion.
- `OperationChannel` is the ordered cross-thread channel for terminal state
  and Stop requests with optional cancellation reasons.
- Run reserves the existing tab busy state before cleanup and registration.
  Active-operation reads project domain-admitted handles. A startup reservation
  has no domain handle until submission succeeds; failed submission releases busy
  without publishing one. Reads remain valid during synchronous gate notifications.
- GUI domain owners project live run/analyze/device handles for `status`, including
  GUI-started work. Shared `OperationHandles` own the wait channel; unknown or
  evicted handles are errors to measure MCP, not finished operations. `wait`
  reports status, progress and feedback but no result payload. Figures and fit
  summaries are read through typed getters.

Cancellation is operation-specific through the registered cancel hook. Each run
owns a fresh RunContext and StopSignal. Adapters pass that context to the core;
Schedule roots receive its cancel_signal explicitly. Experiment set_flux helpers
receive the selected driver and the same cancel_signal, so YOKO ramps observe the
same stop request without ambient scopes or a global registry. Run terminal policy treats the cancel hook as
the source of user cancellation intent; `Schedule` may also set the same stop flag
for internal failed/interrupted outcomes, and those are surfaced as failed
operation outcomes instead of cancelled.

## Progress And Plotting

Progress is operation-scoped:

- Workers emit Qt-free `ProgressEvent` objects through a `ProgressTransport`; QICK accumulated acquisition的內部reps也經`progress_bar.make_pbar`進入同一ambient factory，因此GE每次g/e acquire共用operation-scoped transport而不另建progress model。
- `ProgressService` owns per-operation containers and owner-to-operation mapping.
- GUI widgets attach by owner id (`tab_id` or device name) through the relevant
  control facet; run tabs use `ProgressControlPort`, device panels use
  `DeviceControlPort`.
- Owner listener exceptions are logged and isolated by `ProgressService`; a broken
  progress view does not keep an operation pending.
- Agent polling reads by operation id.

Adapter 在每次 run/analyze 操作中接收明確的 `Plots`。Qt host 綁定該 pane 的
`FigureContainer`，普通 Figure 在完成後呈現，liveplot 的 artist 工作交給 owner thread。
關閉視窗不會將plot host設為全域shutdown；Qt runtime在`aboutToQuit`處理該狀態。
Figure export uses fixed
logical sizes so outputs do not depend on window size: saved images use a 12×9 inch
4:3 canvas at 150 DPI；Data Preview沿用同一logical canvas並以約53.33 DPI產生
640×480 WYSIWYG raster；agent screenshots維持6.4×4.8 inch at 100 DPI。
Analysis start leaves plot teardown to the render host. Terminal domain facts
restore retained figures only after failure, cancellation, or start rejection;
successful content commits attach new figures once. Run failure keeps the
placeholder because run start invalidates prior results, while loaded-result
commit explicitly clears canvases when the new State has no figure.

## Remote / MCP Boundary

`remote/` is the GUI-process driving adapter, peer to `ui/` rather than an
application service. It owns app-specific request coercion and wire projection:
method registry, event serialization, main-thread dispatch, holder admission, and diagnostics.
It projects internal tab interaction/content facts to the coarse
`{tab_id, requery:["tab.snapshot"]}` event envelope; GUI refresh masks and
pane figure details stay internal to `MainWindowEventCoordinator`. The wire event
name and payload are not the internal fact enum. Context/value/parameter/library RPC handlers use
the controller-exposed `ContextControlPort` facet; device RPC handlers use
`DeviceControlPort` for device lifecycle/query/progress; predictor RPC handlers
use `PredictorControlPort` for independent read-only predictor queries with explicit parameters. SoC handlers remain
on the app controller façade for connection policy. Entry list/info/new/open RPCs
use `SetupControlPort` and explicit entry names; there is no project identity mapping.

`zcu_tools.mcp.measure` is the agent-facing bridge: fixed tool declarations,
short waits and live catalog. The GUI remote adapter owns holder admission
and operation outcomes; MCP only maps opaque agent handles to GUI operation IDs.
New GUI RPC methods that should be agent-accessible need a live catalog policy and tests.

`set_frequency`、`set_frequency_sweep`、`set_flux_sweep` 與 `use_flux_device` 的
領域政策由 `zcu_lab.recipes.helpers` 擁有。`RecipeTab` 只提供通用 parameter／library／
cfg field／device 讀取與 set／set_fields／set_sweep，框架不推論 calibration 或 linewidth。
`zcu_lab.recipes.helpers.use_flux_device` 使用通用 `RecipeTab.read_device` 讀 cached
snapshot，caller 明填 unit 必須完全匹配，包括 native。驗證成功後才寫 cfg.flux_dev
與獨立 cfg.flux_unit，不換算、不連線或 setup，也不改 recipe policy 或 alias capture。

## Dialog Rules

Dialogs that can live across operations use `open()`, not `exec()`, and keep a
Python reference until they close. Blocking modal helpers are limited to short
direct user actions that do not wait on worker completion.

`MainWindow.open_dialog` / `close_dialog` is the public registry façade shared by
toolbar actions and remote screenshots. The named-dialog registry helper owns
lazy dialog construction, stable visible-name ordering, persistent predictor
caching, raise/show policy, and per-dialog screenshots; `MainWindow` remains the
`RenderView` façade. General named dialogs and transient dialogs outside the
remote named-dialog surface delegate reference retention and `finished` /
`destroyed` cleanup to the shared dialog lifecycle helper.

Inspect 是 session 的共用 `InspectDialog`。`MainDialogRegistry` 傳入它的全部必填協作者：
`context_control`、event bus、組合根給的 `registry.templates` 讀取函式（每次開 Inspect 讀一次，
reload 後的新範本在下次開啟生效）、`Controller.arb_waveforms`、window 的 DialogPresenter，
以及開 `DialogName.ARB_WAVEFORM` 的入口，讓 ARB 對話框保有具名管理與 remote screenshot，
加上組合根的具名 alias preset catalog。
Measure 不再有自己的 Inspect 子類別或 ARB 實作；行為與 Modules 分頁見
[session README](../../session/README.md)。共用 Inspect、Devices 與 ARB 都提供 `set_read_only`，
宿主在開啟時依控制權的 `writable` 設定唯讀 mode。從 Inspect 開的 ARB 沿用該 Inspect 的 `writable`。
Measure 不對這三個對話框加 run lock，也不追蹤開啟後的控制權變化。
Workflows 依自己的 run lock，從 Start 到 terminal delivery 同步設定這三個對話框的唯讀 mode。
`SetupDialog` receives `setup_control`, so project/context/SoC bootstrap UI no
longer depends on the concrete controller façade. The persistent measure
`PredictorDialog` receives both `predictor_control` and `device_control`, so the
shared dialog can refresh cached device values on every reopen without depending
on the concrete controller.

## Interactive analysis

Shared `gui.interactive` owns the Qt-free `Session` (detached committed snapshots,
atomic owner-loop commit, subscription and single-level undo), `PluginDefinition`,
typed `Action`, and validated command declarations. `AnalyzeService` opens and retains one session
and operation handle per tab; it captures run/context/params at start. Done
validates the committed state before closing input, then uses the existing
analysis-result/writeback terminal path. Cancellation, setup failure and result
failure retire the session and settle that same operation. Neither the service
nor generic remote dispatch interprets flux-line keys.

已遷移的INTERACTIVE adapters以明確的`plots=`建立plugin與frontend。
未遷移adapter不使用舊簽名fallback。
`RunAnalyzeControlFacet` starts the session before mounting; `MainWindow`
mounts/unmounts the plugin-owned `InteractiveFrontend` in the Analysis pane.
Failed finish validation keeps the widget editable; a valid finish unmounts it
before synchronous result events restore the canonical figure in that pane.
The frontend owns artists, pointer selection, preview and timers. For measure flux
picking, the first left click selects a line, pointer movement without a pressed
button previews locally, and the second valid left click commits against the
latest session snapshot. Release does not commit; external commits cancel preview.
Qt-free plugin不依賴widget執行command，Done從已提交的session建立數值結果及
具名圖。Frontend的preview不算結果圖。 `tab.interact` runs on the owner loop via the
same `RunAnalyzeControlFacet` and session: reads project committed state and
plugin-declared commands; writes validate each command's ParamSpec before its
typed action. The View supplies an optional live PNG and `preview_active` as
presentation metadata. `done` discards local preview and finishes the existing
analysis operation. Flux Auto Align uses one plugin-owned single-flight worker
policy for GUI and remote; terminal callbacks do not recommit. The fixed MCP
`tab_interact` tool forwards one request to the GUI `tab.interact` method. Reads do not change focus; validated commands follow the Analysis pane.
This best-effort method has no per-connection seen guard; a later committed
command wins. Cancel through `cancel(op)` / `operation.cancel`, not a domain
cancel tool. Notebook line pickers keep their existing interaction model.
A failed Action does not publish partial state; subscriber errors do not undo
a committed change. `done`
is reserved for terminal delivery. A local preview is not committed state, and
Esc, focus loss, hide, invalid placement or an external commit drop it.

### Arbitrary waveform remote contract

`Controller.arb_waveforms` 直接提供 session bundle 的 `ArbWaveformService`（`gui/session`），不再逐項轉接。
GUI dialog 只接收此 domain port；cfg source 僅依賴列出 keys 的 read port。
`arb_waveform.set` 成功只回傳 `success=true` 與 `status=created|overwritten`。
保存不請求 preview；需要 PNG 時另呼叫 `arb_waveform.preview`，回傳 recipe 與
preview figure。Preview 失敗不撤銷已保存的資產或 revision。GUI 自行繪圖，
remote handler 只將 owner 的 typed status／preview result 投影為 wire payload。
Invalid recipe、key collision、missing asset 等錯誤由 handler 轉為帶穩定 `reason` 的
`RemoteError`，走失敗的 RPC/tool call，不將 `success=false` 當一般 payload。
寫入受當下 holder admission 與資產 domain guards 約束；GUI 和 agent
共用同一資產 owner，遠端只作錯誤和 payload 投影。共同錯誤類別見
[Remote expected-error ADR](../../../../../docs/adr/0068-remote-transport.md)。

## Adapter-Facing Rules

- `ContextService` owns parameter/library commits, versions and change facts.
  The library editor supplies an unlowered schema and fixed ParameterContext.
  App-injected callbacks prepare the Spec before a single structure change.
- `ContextService.apply_writeback` owns fixed-target admission, resource binding
  and publication. It accepts only SavedPoint. Application admission requires a
  selected working point before append; selection does not replace the captured destination.
  `WritebackService` lowers selected proposals to concrete resource items in
  proposal order, not grouped by kind. The shared executor appends accepted,
  commits each item and stops at the first failure. The context publishes each
  successful prefix item. GUI marks that prefix applied and re-raises the original
  cause, preserving its existing wire category. No cross-file rollback or retry
  is provided. Persistence and crash durability remain separate, see ADR-0063.
- `ParameterWriteback` uses wire kind `parameter`, draft-local `param-<n>` IDs
  and parameter paths in `written.parameter` receipts.
- `ExpAdapterProtocol` 是 framework 呼叫 adapter 的契約。`AdapterCapabilities`
  宣告 SoC 需求、analysis、post-analysis、load 的支援範圍；`requires_soc` 不代表
  SoC 已連線。Run guard 檢查 context、cfg、SoC 與 preflight；operation owner 另
  檢查動態 gate。Load permit 和 `LoadService` 檢查 `load_data`，Qt tab 依 analysis／
  post-analysis 宣告建立控制項，remote writeback subtab params 也檢查對應宣告。
  但 analyze permit 未查 analysis，非 interactive 的分析會進 FIT 路徑；
  post-analyze 入口未查 post-analysis。不可將 UI／remote 的局部檢查視為
  所有 application 入口的保證；補齊目標見
  [GUI capability draft](../../../../../docs/adr/draft/gui-adapter-capability-guards.md)。
  `BaseAdapter` 的 import-time 條件 hook 驗證與 concrete adapter 義務見
  [framework BaseAdapter](adapter/base.py)；
  capability 判斷不用 method presence 推測。
- `AdapterGuide(markdown, resource_dir)` carries complete free-format Markdown
  and a `Path | None` for resolving relative image paths. `BaseAdapter` authors
  declare `guide_path = Path(__file__).with_name("guide.md")`, or an explicit
  sibling filename when adapters share a directory. The base reads UTF-8 on
  each `guide()` call. No declaration returns an unwritten-guide notice and
  `None`; a declared missing file raises. `guide_path` is a base-class authoring
  choice, not a member of `ExpAdapterProtocol`. TabService and Controller return
  the same named guide; only the remote adapter converts the directory to JSON.
  The framework does not parse sections. The Guide subtab (`ui/guide_view.py`)
  is the only reader of referenced images: a read-only view that parses with
  markdown-it-py (CommonMark plus tables, raw HTML shown as text) and the
  dollarmath plugin, then shows the HTML in a `QTextBrowser`. Clicked links go
  to the system browser. The only self-written rendering is the math token hook:
  inline `$...$` and `$$...$$` blocks become matplotlib mathtext images, while
  rejected TeX and `\$` stay literal. Code and link destinations keep their
  text because the library tokenizes them first. PNG/JPG files directly in
  `resource_dir` scale down to the pane width; network, SVG and
  other-directory images stay unloaded.
- Adapter `make_form(ctx)` builds a definition from captured source existence.
  Registry registers adapter classes without evaluating factories or templates.
  Open and Reset invoke the factory, bind the current Registry catalog and
  derive input against a fixed FormEnv. Reset replaces shape and defaults
  atomically; preparation failures retain the previous form and sources.
- `validate_run_request(req, cfg)` is a mandatory framework member that
  `GuardService` always calls before opening an async handle. `BaseAdapter`
  supplies the no-op default; overrides are pure, predictable preflight only and
  must not touch devices or mutate cfg/state.
- Adapter `run()` and preflight receive the exact validated Form output.
  The framework neither inspects its concrete type nor validates it again.
  BaseAdapter connects Form[T_Cfg] to run(T_Cfg); the heterogeneous Protocol
  treats only those cfg positions as opaque.
- `analyze()` / interactive analysis hooks must match `AdapterCapabilities`.
- `get_writeback_items()` and `get_post_writeback_items()` return domain writeback
  candidates for their owning analysis pane; the base post hook returns no candidates,
  and writeback commit is framework-owned.
- `WritebackService.create_draft()` accepts those candidates and returns an opaque,
  service-owned draft. Item-local cfg-editor sessions and their identities stay
  inside the service; draft creation cleans every session on failure, teardown is
  idempotent, and `apply_draft()` sends ordered concrete items through the
  context owner's shared writeback entry. `WritebackWidget` is pane-bound and the Qt-only
  `Controller`/`WritebackControl` pane-qualified forwarding (`*_for_pane` with
  `pane` in `analysis|post_analysis`) resolves the pane's opaque draft before
  calling the stage-agnostic service. Remote/MCP writeback operations use the same
  required pane locator; no tab-level draft adapter or wire editor identity exists.

User experiment adapters import `gui.app.measure` contracts. The framework
receives their catalogs by injection and never imports the user package.

## Maintenance Checks

- Cross-module design changes belong in `docs/adr/`.
- App/framework cheat-sheet changes belong here; session-core changes belong in
  `gui/session/README.md`; MCP bridge policy changes belong in
  `mcp/measure/README.md`.
- GUI tests that own `BackgroundRunner` call `quiesce()` before `deleteLater()` or
  process teardown.

Entry New／Open 不建立參數環境，也不自選既有 point。Run、WB 與目前 point 的 Inspect Set 要先選取或建立工作點；GUI 停用操作並提示，application 也拒絕未選取狀態。重新 Open 同 entry 清除 selection，WB 必須再次選取任一 point 才能提交，但目的地仍是候選捕捉的 SavedPoint，不跟隨畫面上的 point。Cfg/editor drafts 與 WB 候選 preview 保持各自的生命週期。
