# `zcu_tools.gui.app.measure` — measure-gui

**Last updated:** 2026-10-02 — 顯式 RunContext 與取消訊號

`gui.app.measure` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。實驗領域知識住在 `experiment/v2_gui/measure/` adapter；
framework 只看 `ExpAdapterProtocol`。

Main 擁有 `AppPersistedState` codec/version、filename、originator、restore presentation 與
lifecycle-only triggers；disk mechanism 使用 `gui.session.persistence.SingleFileCaretaker`。

## 局部設計與閱讀入口

此 app 的 View（Qt UI）與 `RemoteControlAdapter`（`remote/`）共用可用的
application commands，但 capability 驗證尚未覆蓋全部入口。`GuardService` 對
run、load、save、analyze、writeback 的既有檢查發放不同型別的 Permit；
analyze 不檢查 adapter 的 analysis capability。操作期間的 busy／硬體互斥
由 owning service 或 app-local `OperationGate` 處理。Permit、lease、handle
各有不同的生命週期，不以 handle 推斷取得硬體 lease。詳見
[Operation ADR](../../../../../docs/adr/0066-operation-lifecycle.md) 與
[capability draft](../../../../../docs/adr/draft/gui-adapter-capability-guards.md)。

`CfgEditorService` 按 `editor_id` 保存 cfg draft。可回收的 library-entry
session 受 LRU／disconnect 回收；由 UI owner 建立的 seeded session 由 owner
顯式 teardown。Widget attach／detach 不取得 draft 的銷毀權。
Tab cfg 由 `Session.cfg` 的 `CfgResource` 擁有，不建立 editor session 或
`State.cfg_schema` 鏡像。Qt 與 remote 都向同一資源提交指定 revision 的命令。
`WritebackService` 以 opaque draft 保存候選項及 item-local editor
session，不能把 preview 當成再次計算候選項的指令。`CfgDraft` 的共用
Spec／Value、`None`、locked literal、reference binding 與 lowering 契約見
[Cfg ADR](../../../../../docs/adr/0065-cfg-editing.md)、
[GUI cfg README](../../cfg/README.md) 與
[experiment cfg editing README](../../../experiment/cfg_editing/README.md)。
Tab cfg edit batch 在完整候選準備成功後一次發布，拒絕不改舊 publication。
獨立 library editor 的 draft batch 不具有這項原子保證。Source refresh 使用
已發布快照，不把 live provider 當成 observation 的第二個 owner。Measure tab、
Run、Qt 與 remote 的資源契約現況見 Cfg ADR；library conversion、selected Apply
與其他 app 的剩餘目標見 [cfg draft](../../../../../docs/adr/draft/cfg-editing-boundaries.md)。

`State` 保存可觀察的 app 資料，`ContextService` 寫入 md／ml；services
依用途讀 owner 的 read contract、單向呼叫 command 或訂閱已提交 fact。
版本由資源 owner 發布，不以每次 emit 必然 bump 推導：
`SessionState.refresh_device_info_cache()` 在 driver info 與快取相同時不 bump，
不同時 bump device version，caller 再發布變更。Tab cfg edit／Run 使用明示
`CfgRef`，不另要求每條連線的 cfg seen；它不取代 authentication 或其他 guards。
其他受護 RPC 由 remote adapter 在 GUI owner thread 比對該連線的 seen 與目前
版本，依賴與讀取揭露資源由 method entries 宣告。這些 key 未讀過時，即使
版本為 0 也不能寫入。MCP 原樣轉送 supplied cfg ref，不保存第二份 seen，
不傳 `expected_versions`，不隱藏預讀或自動重試。Stale 由 caller 明確重讀
对应完整 publication／snapshot，再決定是否重送。
GUI 事件與 service 協作見 [GUI ADR](../../../../../docs/adr/0067-gui-application.md)，
wire guard 與 off-owner await 見 [Remote README](remote/README.md)。

`AppPersistedState` 是記住的 session preference 與 session 的選擇性投影
（磁碟 slice 沿用 `startup` 名稱），不是整個 State 的序列化。
`State.preferences` 保存下次預填值，面板寬度由 `Controller.get_left_panel_width()`
提供；restore
不自動連接 SoC 或套用 active context。`WorkspaceService` 處理 session
capture／apply，shared cfg codec 轉換 cfg raw；`SingleFileCaretaker` 只
處理單檔 I/O，不認識 State／cfg。關閉等待的 timeout 並不證明 cleanup
完成。保存邊界見 [Persistence ADR](../../../../../docs/adr/0063-persistence-ownership.md)，
關閉與取消見 [Operation ADR](../../../../../docs/adr/0066-operation-lifecycle.md)
及 [operation draft](../../../../../docs/adr/draft/operation-lifecycle-boundaries.md)。

## Package Boundaries

- `adapter/`：framework-facing contract、measure-owned finished-cfg ports、analyze params、
  adapter validation與protocol signature需要的session vocabulary；不forward generic cfg API。
- `specs/`：`experiment.cfg_editing`的main policy adapter；只綁定Arb asset choices與readout
  cross-shape inheritance，不擁有program field/label清單。
- `cfg_schemas.py`：main raw/typed cfg normalization與policy facade；全七種module/六種waveform的
  spec walk、missing/nested/reference規則由`experiment.cfg_editing`materializer擁有。
- `services/`：app service layer。Service 依賴 ports，不直接 import sibling service
  implementation；package `__init__` 只做 lazy public re-export。
- `state.py`：tab/device/pane/path/version-table SSOT 與主線程 mutators；固定的
  Run、Analysis、Post-Analysis、Save pane 各自擁有自己的 resource。`running_tab_id`
  是唯一 run ownership 狀態，tab interaction 的 `is_running` 由它投影。MainWindow
  also projects that identity directly onto exactly one top-level tab as a compact blue
  `●` marker while Run is active; Run start/terminal EventBus reactions and tab insertion
  repaint from State, so the marker is never persisted or cached as a second busy state.
- `ui/`：Qt widgets、MainWindow top-level façade、capability-driven `ExpTabWidget`、
  writeback view、feedback/prompt widgets；generic cfg form不屬於app package。
  `ExpTabWidget` owns capability-driven left subtab composition (fixed order Run |
  Analysis | Post-Analysis | Data | Guide, with optional Analysis/Post only when
  the adapter declares the capability) and per-pane `FigureContainer` routing
  (stable identity per (tab, pane) for the widget lifetime; refresh never replaces
  the container; busy tabs cannot close or rebuild, so the captured worker target
  outlives the operation without a lease). It receives tab actions through a narrow
  `TabActions` port with pane-qualified writeback (`apply_post_writeback`);
  `MainWindow` adapts those actions to top-level handlers. Run uses the sole
  shared cfg tree (S1 corrected: 13 px, root at 0, descendants at 10 px with
  connectors, five depth colors `#5b8dc6`/`#6aae8a`/`#b8942f`/`#8a6bc9`/`#4fb3a8`
  cycle on guide lines stable under horizontal scroll, rows no longer use depth
  backgrounds, whole-row folding, reference shape elision, viewport follows
  available panel height with scrolling only when content exceeds viewport);
  Run action row is status-free with Reset 20% / Run 80% of available width；
  idle Reset與Run同高，Reset為green secondary、Run為blue primary，running時隱藏Reset並讓red Stop佔滿整列，
  settled後恢復20/80（A5）； Analysis uses an app-local single-column 13 px ledger with
  whole-header folding for `Analysis parameters` and `Writeback preview` and a
  full-width `Analyze` immediately after parameters and before Writeback preview
  (A6). Presentation
  Modules do not invoke operation services directly. Each `WritebackWidget`
  is pane-bound (analysis vs post_analysis) and edits/applies its own opaque
  draft via `Controller`/`WritebackControl` pane-qualified forwarding, while
  `WritebackService` remains stage-agnostic and owns the display-only baseline
  capture (S2): at draft creation it snapshots the destination `SessionEnv` and
  exposes per-item `current_summary` / `proposed_summary` to Qt. The widget is a
  compact unified ledger: draft-owned unapplied items以粗體`target*`與`* = not applied`
  legend呈現，成功寫入的items改為一般字重`target`，retarget或內容修改後回到unapplied；
  apply/edit的app-owned fact會讓本地與remote入口都觸發同一writeback重投影；
  centered Current → Proposed columns on a shared-background continuous-boundary
  panel (white rows with bottom dividers), and equal 56×26 Edit/Copy actions。
  Scalar MetaDict與editable module/waveform items使用Edit；non-scalar MetaDict的
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
  internal `ArtifactSaveCenter` which把capability-driven `Load Data` / `Save All`
  action row放在`Measurement data`card之前，並從`TabSnapshot`呈現DATA與目前每張
  具名圖的status、saveability與路徑草稿。每個`Session`的Qt-free `ArtifactTracker`是唯一狀態來源；
  `SaveService`於真實terminal成功後記錄實際路徑，失敗不清除先前成功的紀錄。
  Data save使用既有OperationRunner/Handles，不可取消、不持硬體lease；GUI與remote
  都取得同一SaveDataSubmission，包含operation ID與保留路徑，後者不代表成功。
  Save All 只選可保存且尚未保存的項目，也包含 DATA 的既有 path/comment 變更。
  同次請求的新 DATA draft 先參與選取預檢，驗證成功後才發布草稿。
  `ArtifactSnapshot.needs_save` 同時供 facade 選取與 Qt 按鈕 gating 使用；全數已保存時
  Save All 停用，明確單項匯出仍可再次保存。修改圖形目的地不取消已保存狀態。
  SaveService 依 analysis逐圖→post逐圖→data 順序，以單一 operation 執行並 Fast Fail，
  不回滾已完成的存檔；再次 Save All 省略成功項。Qt 按鈕不編排各項存檔。
  AppServices獨立注入OwnerScheduler，image export回owner thread，data I/O在worker。
  Batch completion在State/handle terminal之後發布，Controller沿既有diagnostic port呈現結果。
  GUI只警告尚未儲存的measurement data；`MainWindow`在使用者關閉tab/app前
  查詢其投影，並將app關閉的資料流失與active-operation風險合併確認。
  Programmatic RPC shutdown不彈互動式確認。
  Save All updates that center in place: terminal status updates do not replace the Data
  pane or its widgets, and the data-path editor retains focus, cursor and selection.
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

- `zcu_tools.gui.cfg`：Qt-free Spec/Value model、`CfgSchema` data carrier、inheritance、
  persistence codec、domain-free raw spec walk與generic finished-cfg validation/lowering ports。
- `zcu_tools.experiment.cfg_editing`：Qt-free closed program module/waveform shape catalog與fresh Spec
  factories，以及program missing/nested/reference/subset materialization policy；main以app-local policy
  啟用Arb choices、readout inheritance與完整7+6 materializable catalog。
- `zcu_tools.gui.widgets.cfg`：shared `CfgFormWidget`、field renderers、decoration contract與
  instance-owned frozen exact renderer registry。
- `zcu_tools.gui.session`：context、SoC、device、project settings、predictor、operation
  handles、operation runner、notify channel、progress/shutdown service、shared dialogs。
- `zcu_tools.gui.remote`：NDJSON RPC endpoint、framing、wire errors、router base。
- `zcu_tools.gui.plotting`：matplotlib backend、figure routing、host/container/export
  substrate。
- `zcu_tools.mcp.measure`：agent-facing MCP policy layer and tool surface。

## Composition Root

`MeasureGuiBehavior` is the process-runtime behavior for the shared
`gui.runtime` launcher seam. It assembles `State`, `Controller`, `MainWindow`,
persistence caretaker, and the app-local `RemoteControlAdapter`
without owning process policy such as logging, matplotlib backend selection,
`QApplication`, control option construction, or exit-code handling. The
standalone launcher is the process entrypoint; this module does not expose a
second `run_app` path. After the window is shown, `after_show` opens the same
Setup dialog through `MainWindow.open_dialog`, so a toolbar click focuses that
instance instead of building another.

The launcher still owns the experiment-adapter composition boundary by passing a
registry factory into `MeasureGuiBehavior`; the factory imports
`experiment.v2_gui` only after `gui.runtime` has configured logging and the
pre-Qt plotting policy.

`build_app_services()` constructs the app-local services and injects their driven
ports. `Controller` is a facade over the service bundle; UI and remote code use
the controller for app-specific workflow and the exposed session control facets
for setup/context/device/predictor/progress domains.

App-local driving-adapter facets mirror the shared session control pattern.
`TabControlPort` / `TabControlFacet` expose the tab resource surface (lifecycle,
active/running identity, tab read model, cfg resource lookup, save path overrides)
by composing `WorkspaceService`, `TabService`, `State`, and `EventBus`; remote
tab handlers use this facet instead of the giant `Controller` surface.
`RunAnalyzeControlPort` / `RunAnalyzeControlFacet` expose the run/load/analyze
operation surface (run start/cancel, result load, analyze/post-analyze start and
result reads) by composing the operation services, guards, tab read model, and a
render-host provider. Remote run/analyze handlers use this facet instead of the
giant `Controller` surface. `OperationControlPort` / `OperationControlFacet`
expose the op-agnostic handle/progress surface used by generic `operation.*`
handlers, including device setup handles. `SaveControlPort` / `SaveControlFacet`
expose save artifact creation and save-path mutation by composing `GuardService`,
`TabService`, `SaveService`, `State`, and `EventBus`; remote save handlers use
this facet instead of the giant `Controller` surface. `WritebackControlPort` /
`WritebackControlFacet` expose persistent writeback draft read/edit/apply by
composing `GuardService`, `WritebackService`, `State`, and a resource-version
provider; remote writeback handlers use this facet instead of the giant
`Controller` surface. Cfg-editor remains a separate domain. Qt and remote
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
figure. Local analyze/post/save-path edits keep synchronous State commit timing
but have no Qt reaction. Explicit image destinations committed by a save command
publish a separate save-draft fact. The coordinator projects those State paths
into open editors before export, including when export subsequently fails.
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
- `ContextService` is the only writer for live `MetaDict` / `ModuleLibrary`
  contents. Its ModuleLibrary schema-replacement interface validates names and
  lowers before mutation, then emits one `ML_CHANGED` fact and bumps `context`
  once; failures leave the live entry untouched.
- `State` owns tab/device/pane/path resource state and resource versions. Pane swaps
  happen on the owner thread and return retired resources for post-commit cleanup.
- `GuardService` checks the static preconditions it implements and returns typed
  permits for run/save/analyze/writeback. The analyze permit does not check adapter
  capability, and post-analyze has no capability permit; Qt and remote entry points
  keep their own checks (see the GUI capability draft).
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
和 failure/recovery presentation。Role catalog、hardware context 與 framework 不重建。
依賴重載是 best-effort，不全面保證 deferred/dynamic import 的一致性，也不禁止函式內 import；
確切限制見 `experiment/v2_gui/measure/README.md`。此取捨不放寬資料丟棄確認或 lifecycle 保護。

Retry 的 prepare/load 皆保留 typed error disposition；要求 restart 後不再提供可 retry 狀態。
Shutdown 暫停 experiment entries；settle 後的未保存資料確認若取消，MainWindow 呼叫
`Controller.abort_shutdown()` 恢復入口，不自動重啟已取消的 operations，也不重新啟用失敗的
catalog。開始 shutdown 若拋錯，Controller 恢復原 gate 狀態並保留例外。

Reload 在 owner thread 同步執行，不 pump Qt events；進行中 handle 與 tab busy flags
共同阻止 reload，包含 handle-backed data及artifact batch save。共用 `ExperimentAccess` 阻止 local/remote
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
2. The tab owns a persistent `CfgResource`, independent of widget lifetime.
   Run renders its publications through `ResourceCfgFormWidget`; Analysis
   renders its params through the app-local 13 px ledger with whole-header
   folding and a full-width `Analyze` immediately below parameters.

3. `Controller.start_run(tab_id, expected)` requires the observed `CfgRef`.
   Qt supplies the form's displayed ref; remote callers supply their observed ref.
   A different cfg identity or revision rejects Run without refresh or fallback.
   `GuardService` accepts that exact Valid revision and freezes a permit with
   `AcceptedConfig` provenance and detached State-owned device settings. Missing observed
   settings for a live device reject the permit without querying hardware.
   `RunRequest` carries only SoC handles and that device snapshot, not md/ml.
4. RunService captures borrowed drivers and creates a fresh RunContext with
   operation plots and one StopSignal. The adapter passes this context to the core;
   Schedule uses its StopSignal and device setup uses the same signal's event.
   Only progress remains ambient. Driver lookup does not refresh accepted cfg.
5. `BackgroundRunner` executes blocking work off the Qt main thread and marshals
   terminal callbacks back to the main thread.
6. Run/analyze services depend on narrow State ports (`RunStatePort` /
   `AnalyzeStatePort`) for busy checks, request-building reads, and result writes.
7. Writeback items are generated from analysis results and edited through the same
   cfg-editor machinery before commit; `WritebackService.create_draft` snapshots
   the destination `SessionEnv` at creation and the ledger shows
   `current_summary` → `proposed_summary` per item (S2). Scalar MetaDict items
   show concrete values; module/waveform items show bounded change summaries and
   keep full cfg editing in `Edit`. Primary and post workflows own proposal timing;
   the Writeback service remains stage-free.

- `load_tab_result` 載入 typed RunRecord，以 Record.cfg 作為來源快照。缺 cfg 時保留可用資料，不捏造設定。有效 cfg 透過同一個 CfgResource 原子 backfill，保留 resource identity 並推進 revision。Run/Load 先提交 primary result；analysis preparation 失敗不撤回它，只報 diagnostic。Named figures 和各 pane 的 presentation lifecycle 分開管理。
dynamic selectors must match live options and the new complete draft must be valid
before publication. A failed lookup or malformed option list rejects the entire
backfill rather than silently skipping that selector. Module/waveform references
become custom values rather than guessed library keys.
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
snapshot). Save stores the data-path override. Each image path override belongs to
one `(stage, figure_name)` key; the read model projects DATA and current named
images separately. Run live figures remain view-only and
are not stored in State. Writeback baseline is a display-only draft-creation
snapshot；同一opaque draft另擁有per-item applied state，只有成功write包含的items才標記applied，
selection本身不改狀態，retarget或內容修改會重設；同kind items不得指向重複destination，
避免batch覆寫卻誤標applied。狀態不跨draft/process持久化，也不提供
concurrent-write detection或apply-conflict policy。

Analysis/Post 逐圖記錄目前 result／Figure 產物是否曾成功保存，不追蹤
artist、視圖或目的地的 dirty 變更。新圖從未保存開始；舊圖晚到的保存完成
只更新提交時捕捉的record，不將同名新圖標成已保存。再次匯出失敗保留先前成功紀錄。DATA 仍追蹤原有
result、path 與 comment 的保存 signature。

Analysis/Post result services prepare proposals, named plots and drafts before calling one
owner-thread State swap. The swap returns every retired pane resource; services tear
down retired drafts only after commit and never roll back a committed pane when cleanup
fails. A failed proposal/editor build leaves the previous canonical pane intact.
Primary analysis replacement invalidates Post-Analysis, Post replacement leaves
Analysis untouched, and a successful run/load clears both downstream panes.

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

## Tab Lifecycle And Ordering

New tabs are pure GUI configuration surfaces: creating one builds the adapter's
default cfg from the current context but does not start hardware work. The toolbar
therefore stays available while another tab is running; per-tab interaction state
and `OperationGate` still prevent starting a second run until the active run
finishes.

Top-level experiment tabs are movable. The visible order is synchronized back to
`State` through the controller/workspace lifecycle path, so `list_tab_ids()`,
remote tab views, and captured sessions all use the same tab order as the Qt tab
bar. Active and running tabs are identified by tab id, not visual index.

## Config Model

Measure Config uses `ResourceCfgFormWidget` and a State-owned cfg resource.
The form keeps text input local and captures its first publication ref. Run
submits that input once, then starts only with a Valid returned publication ref.
Invalid, Stale, Unavailable or failed submission never runs the previous values.
Pending input can repair an Invalid publication, but cannot bypass busy,
context or SoC gates. External updates preserve local text, focus and selection.
Discard uses the latest delivered publication. Reapply asks for confirmation of
the differences and submits against the revision shown in that confirmation.
Library editors and writeback keep their existing `CfgDraft` binding behavior.

`Session.cfg` 只引用該 tab 的 `CfgResource`，不保存另一份 live schema。
CfgResource 擁有 input、resolution、revision、publication 與 acceptance。
Qt、remote、reset 和 load 共用這個 owner。Edit 完整 batch 原子發布，合法未完成輸入
發布 Invalid；拒絕不留下成功前綴。Observe、snapshot 和 Run acceptance 不重新讀取來源。
`TabSnapshot.cfg_schema` 是 detached input memento，只供 snapshot／workspace restore。
`CfgEditorService` 保留 library、inspect 和 writeback 的獨立 draft，不發布 tab cfg。

獨立 library／inspect／writeback 的 CfgEditor 在 app seam 解碼 `ValueRef`，
並依序操作 draft binding target。這個 draft batch 保留 fail-fast/non-atomic 行為，
每筆成功 edit 各自 bump version。它不適用於 tab cfg resource 的原子 batch。

Shared `CfgSchema` pairs two trees. The resource owns these inputs; the form
renders publications and keeps only unsubmitted input locally:

- Spec tree: static shape, labels, variants, literal locks, optional/ref rules.
- Value tree: authored inputs, including expressions and reference identities.

Resource acceptance rejects unresolved or invalid cfg. Legal optional `None`
remains valid. Run does not reread `MetaDict` or `ModuleLibrary`.
`schema_to_raw_dict(schema, md, ml)` remains the live lowering boundary for
non-Run draft consumers. `CfgSchema` stores shared spec/value data and `CfgDraft`
provides draft expression evaluation and cached validity. `ValueRef` resolves
once through the session `ValueLookup` and stores a direct scalar input.

generic model、spec walk、inheritance、codec、static/dynamic validation與lowering由
`zcu_tools.gui.cfg`擁有，consumer直接從shared owner匯入。measure adapter只把current
`MetaDict` expression evaluator、measure-owned module/waveform resolver與`SweepCfg`
factory組成三個窄ports；adapter package不import或forward shared cfg public names，也不保留
第二份algorithm或model/inheritance/codec implementation（ADR-0065）。

Module與waveform field在shared model都使用`ReferenceSpec(kind=...)` / `ReferenceValue`。
measure-owned pulse/waveform spec factory顯式設定`kind="module"`或`kind="waveform"`，
`services.tab_cfg.TabCfgResources` 管理 tab 到 cfg resource 的身份關聯。TabService 建立及關閉
資源，lookup 直接回傳 resource-bound editing，不轉送命令。建立失敗不留下關聯，retire
先撤銷舊 handle，再移除關聯。Widget detach 只停止觀察，不改資源 lifetime。

`MeasureCfgBindings.snapshot_from_state` 複製 metadata、library 與 options，並記錄 context、
device set、每個 device 及 value cache 的 source basis。Bare capture 使用同一 metadata。
Dotted capture 只讀 `ValueSourceBinder` 已發布的 detached cache，不查 live provider 或硬體。
Source 更新先準備及安裝所有 tab 的新 publication，再通知任何 subscriber。
來源故障發布 Unavailable，不保留舊 Valid。通知期間拒絕 cfg mutation、acceptance 和 tab lifetime mutation。

`MeasureCfgBindings`依`spec.kind`選擇精確的ModuleLibrary store/materializer facade，並提供expression、
dynamic scalar options與ValueRef resolution policy；widget只讀field API，shared cfg不認識這些
app-local policy。device selector是required string `ScalarSpec`，wire value維持`DirectValue(str)`。
measure composition在`ui/cfg_binding.py`以generic `QLineEdit -> keepalive object` enhancer seam安裝
`ValueSourceInputController`，保留eval input的completion與resolve-on-space；shared binding仍
不import ValueRef或session，shared widget只接收generic enhancer callable。

ModuleLibrary reference enumeration只以`experiment.cfg_editing.program_shape_for_input`讀root discriminator；
不normalize typed cfg或建立Spec/Value。resolve才呼main materializer façade一次。Experiment
composition把fresh canonical shape factory與eval-aware value factory一起註冊到immutable `RoleEntry`；catalog
registration只驗shape/kind，Controller create依序取得value與fresh shape並直接組`CfgSchema`，不從
value sniff discriminator。role wire metadata與既有blank role順序保持不變。

ModuleLibrary 新建唯一走 role catalog 的 `create_from_role`；`CfgEditorService.open`
只以 required `from_name` 開啟既有 module/waveform 的 modify session，不提供
discriminator blank seed。

Sweep-like fields keep their UI value model until this lowering boundary:
`SweepSpec` stores `start` / `stop` / `expts`, while `CenteredSweepSpec` stores
`center` / `span` / `expts` and lowers to a program sweep only when building the
raw experiment cfg. Centered sweep centers may be locked independently from the
span/expts controls, which lets callers expose generated centers while keeping
the search window editable. Sweep editors render as two balanced label+input
columns per row, so start/stop or center/span share the available width evenly
inside a full-width form row.

Linked module / waveform reference fields preserve their embedded value snapshot
when the library key is missing. The field stays library-keyed and invalid so
re-adding the same key relinks it, while persistence can still serialize the snapshot without consulting
`ModuleLibrary`. Restored overridden refs whose key is missing can instead
retain their overridden inline input without depending on the old key. Nested linked
references keep their own dependencies; explicit relink restores this layer's dependency.

Adapter cfg authoring lives in `experiment/v2_gui` as a context-free
`MeasureCfgDefinition`. A single `MeasureCfgBuilder` declaration fixes static shape,
field order, role, lock and deferred typed Seed; fresh `instantiate(ctx)` only resolves
value defaults, then `BaseAdapter.make_default_cfg` validates the finished schema.
The framework protocol does not expose a static spec query. Shared
`CfgSchemaAssembler` owns only paired-tree mechanics and has no measure domain/context
knowledge (ADR-0012、ADR-0065).

Measure tab 使用 shared `ResourceCfgFormWidget`，只 watch publication 並提交 typed input。
以下 `CfgFormWidget` binding 路徑供獨立 library／inspect／writeback draft 使用。
每個 `CfgFormWidget` 持有自己的 frozen exact registry；沒有顯式注入時，
`default_cfg_renderers()` 為五個 non-section exact field types
（`LiteralField`、`ScalarField`、`SweepField`、`CenteredSweepField`、`ReferenceField`）註冊固定
`FieldRenderer(field, context)`，`SectionField` 不在 registry 而由 sole tree
（`TreeCfgWidget`）直接建立 `QTreeWidgetItem` 結構。immutable `FieldRenderContext` 只攜帶 path、
top-level 標記、label width、decoration resolver、text enhancer 與同一 frozen registry；leaf 與
reference header 走 registry `render()`，而 section 結構由 tree 直接建立，沒有 consumer-side
constructor dispatch、global decorator registration 或 inheritance fallback。attach 在成功 build
tree root 後才訂閱 draft，detach 會解除 change/validity callbacks 但不 close draft。`CfgFormWidget`
accepts an optional field decoration provider keyed by full value tree path. The shared widget
owns only generic presentation metadata (`hidden`/`enabled`/tone/badge/tooltip/label suffix) and
computes the default decoration from the spec; app-specific policy such as generated fields stays in
the caller. `LiteralSpec` fields stay hidden by default, but a decoration provider can explicitly
reveal them as framed read-only values for generated or locked review fields. Decoration is a view
contract only: domain enforcement remains in the owning controller/runtime.

`CfgFormWidget.set_editing_enabled()` locks only the rendered form content, not
the widget shell or its `QScrollArea`. Busy/read-only hosts keep the cfg pane
scrollable while child editor controls are disabled, and the desired editing
state persists across `detach()` / `attach()` swaps of the service-owned draft.

Nested `CfgSectionSpec` fields render as tree items with whole-row folding
(`QTreeWidgetItem` at 10 px indentation, 13 px text, classic connectors); the section header
is the item label and does not create an additional parent-row label. This keeps grouped forms
such as autofluxdep Generation overrides from showing duplicated text like `Frequency recovery:`
next to a second `Frequency recovery` header, and keeps the single tree presentation consistent
across Run, autofluxdep, and module/waveform editors.

`ChoiceSectionSpec` is the shared selector-driven display contract for sections
whose fields depend on a local mode/strategy. The section still owns a complete
union `CfgSectionValue`; each `ChoiceBinding` names the selector field and the
fields rendered for each selector value. `CfgFormWidget` refreshes only the
affected section subtree (including reference-elided subtree owner, e.g.
`modules.qub_pulse` for `modules.qub_pulse.gain`) when a selector or decoration changes,
while hidden inactive fields keep their values in the model and lower/persist through the
normal section path. Decoration-provider changes follow the same section-local (section or
reference) refresh path instead of reattaching the full `CfgDraft`-backed form. Field widgets
expose a typed `refresh_section(path) -> bool` surface, and decoration state is consumed
through the shared `FieldDecoration` surface rather than ad-hoc attribute probing.
Unknown `ChoiceSectionSpec` selector values fast-fail instead of hiding all
controlled fields.

## Operation Model

- Run, device setup, and SoC connect use hardware exclusion.
- Analyze and post-analyze use async handles but no hardware exclusion.
- `OperationChannel` is the ordered cross-thread channel for terminal state,
  user messages, and Send & Stop.
- `NotifyChannel` mirrors the same pattern for the `notify.open` / `notify.await`
  RPC prompt. `notify.await` bounds the consumer wait so MCP transport stays alive.
- `FeedbackDockController` owns the docked feedback panel, target-tab
  resolution, and op-count plus agent-presence gate; `MainWindow` keeps the
  public render-view refresh façade.
- Run reserves the existing tab busy state before cleanup and registration.
  Active-operation reads project domain-admitted handles. A startup reservation
  has no domain handle until submission succeeds; failed submission releases busy
  without publishing one. Reads remain valid during synchronous gate notifications.
- GUI domain owners project live run/analyze/device handles for `status`, including
  GUI-started work. Shared `OperationHandles` own the wait channel; unknown or
  evicted handles are errors to measure MCP, not finished operations. `wait`
  reports status, progress and feedback but no result payload. Figures and fit
  summaries are read through typed getters.

Cancellation is operation-specific through the registered cancel hook. Run
cancellation sets the operation `stop_event`; worker thunks expose it to
Schedule-based experiments and executors through
`schedule_stop_scope(StopSignal(stop_event))`, so `ProgramBuilder`,
`Schedule.repeat/scan/batch`, and executor root schedules observe Stop without a
global task runner context. The same run-local `stop_event` is explicitly bridged
into `device_setup_cancel_scope(stop_event)`, so experiment-internal
`setup_devices(...)` calls can stop long device ramps without making the runner
module know about device policy. Run terminal policy treats the cancel hook as
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

已遷移的adapter在每次run/analyze操作中接收明確的`Plots`；Qt host綁定該pane的
`FigureContainer`，普通Figure在完成後呈現，liveplot的artist工作交給owner thread。
舊adapter仍可能使用`gui.plotting`的pyplot routing backend，直到全部實驗遷移完成。
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
method registry, event serialization, main-thread dispatch, resource-version
guard, editor lifecycle, and diagnostics.
It projects internal tab interaction/content facts to the coarse
`{tab_id, requery:["tab.snapshot"]}` event envelope; GUI refresh masks and
pane figure details stay internal to `MainWindowEventCoordinator`. The wire event
name and payload are not the internal fact enum. Context/value/md/ml RPC handlers use
the controller-exposed `ContextControlPort` facet; device RPC handlers use
`DeviceControlPort` for device lifecycle/query/progress; predictor RPC handlers
use `PredictorControlPort` for predictor load/query/compute. SoC and
`project.apply` handlers remain on the app controller façade because they span
project setup and connection policy rather than a single session-control domain.

`zcu_tools.mcp.measure` is the agent-facing bridge: fixed tool declarations,
short waits and live catalog. The GUI remote adapter owns per-connection seen
and operation outcomes; MCP only maps opaque agent handles to GUI operation IDs.
New GUI RPC methods that should be agent-accessible need a live catalog policy and tests.

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

`InspectDialog` adapts the measure controller into the shared
`InspectDialogBase` by passing `context_control`; the subclass keeps the concrete
controller only for measure-only CfgEditor and role-catalog actions. Measure owns
the dense two-column Parameters property grid and a separate Modules composition:
the Modules tree exposes only New/Delete collection actions, while the right pane
embeds the service-owned `CfgFormWidget` with Name, Saved/Unsaved, Revert, and
Apply on one row (there is no Raw pane or Modify/Rename action). Selecting an entry
opens one `gc=False` editor session; field and Name edits stay in that draft.
Apply calls the replacement write interface, which validates and lowers before the
single ContextService-owned name+cfg mutation, then reopens a fresh clean draft on
the resulting selection. Collision, invalid, or lowering failures leave the live
entry and draft intact. Each existing-entry editor retains its source
ModuleLibrary identity and fast-fails replacement after a context switch, leaving
that draft for explicit Revert/Discard. A pending refresh is consumed after a
dirty selection transaction so the tree reflects the resulting names. Revert
reloads live content, and dirty selection/close requires Apply, Discard, or Cancel.
New remains a retained non-blocking role catalog dialog; the Modules tree Delete
key is direct only at the tree focus boundary, while the button confirms.
Autofluxdep keeps the base presentation and its read-only wrapper.
`SetupDialog` receives `setup_control`, so project/context/SoC bootstrap UI no
longer depends on the concrete controller façade. The persistent measure
`PredictorDialog` receives both `predictor_control` and `device_control`, so the
shared dialog can refresh cached device values on every reopen without depending
on the concrete controller.

## Interactive analysis

`interactive/` owns the Qt-free `Session` (detached committed snapshots, atomic
owner-loop commit and subscription), `PluginDefinition`, typed `Action`, and
validated command declarations. `AnalyzeService` opens and retains one session
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

`Controller.arb_waveforms` 直接提供既有資產 owner 的窄 port，不再逐項轉接。
GUI dialog 只接收此 domain port；cfg source 僅依賴列出 keys 的 read port。
`arb_waveform.set` 成功只回傳 `success=true` 與 `status=created|overwritten`。
保存不請求 preview；需要 PNG 時另呼叫 `arb_waveform.preview`，回傳 recipe 與
preview figure。Preview 失敗不撤銷已保存的資產或 revision。GUI 自行繪圖，
remote handler 只將 owner 的 typed status／preview result 投影為 wire payload。
Invalid recipe、key collision、missing asset 等錯誤由 handler 轉為帶穩定 `reason` 的
`RemoteError`，走失敗的 RPC/tool call，不將 `success=false` 當一般 payload。
寫入受 `arb_waveforms` resource version 的 expected-version guard 約束；GUI 和 agent
共用同一資產 owner，遠端只作錯誤和 payload 投影。共同錯誤類別見
[Remote expected-error ADR](../../../../../docs/adr/0068-remote-transport.md)。

## Adapter-Facing Rules

- `ContextService` 擁有 md／ml 內容提交、`context` version bump 與變更事件。
  Measure 的 `ContextWritePort` 從 cfg schema 產生 app-side lowering callbacks，
  交給共用 service 在寫入時呼叫；`CfgEditorService` 只交未 lower 的 schema。
  Writeback 選出的 md／ml entries 交給一次 `apply_ml_writes()`，每批至多
  bump 一次、每種變更事件至多送一次。整批 lower／register 先在隔離候選
  完成，後項可讀前項候選，但不能改 live md／ml。準備失敗保留舊內容與版本。
  成功後一次套用內容、更新版本並通知，最後保存。保存失敗明確回報已套用
  但未保存，不 rollback 或自動重試。寫入與 crash durability 是不同責任，
  磁碟保存見 ADR-0063。
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
  [experiment adapter README](../../../experiment/v2_gui/measure/adapters/README.md)；
  capability 判斷不用 method presence 推測。
- Adapter `cfg_definition()` is context-free authoring; only fresh
  `make_default_cfg(ctx)` materializes deferred defaults and validates the schema.
- `validate_run_request(req, raw_cfg)` is a mandatory framework member that
  `GuardService` always calls before opening an async handle. `BaseAdapter`
  supplies the no-op default; overrides are pure, predictable preflight only and
  must not touch devices or mutate cfg/state.
- Adapter `run()` receives a concrete config and performs the experiment.
- `analyze()` / interactive analysis hooks must match `AdapterCapabilities`.
- `get_writeback_items()` and `get_post_writeback_items()` return domain writeback
  candidates for their owning analysis pane; the base post hook returns no candidates,
  and writeback commit is framework-owned.
- `WritebackService.create_draft()` accepts those candidates and returns an opaque,
  service-owned draft. Item-local cfg-editor sessions and their identities stay
  inside the service; draft creation cleans every session on failure, teardown is
  idempotent, and `apply_draft()` sends selected entries through one
  `ContextWritePort` batch. `WritebackWidget` is pane-bound and the Qt-only
  `Controller`/`WritebackControl` pane-qualified forwarding (`*_for_pane` with
  `pane` in `analysis|post_analysis`) resolves the pane's opaque draft before
  calling the stage-agnostic service. Remote/MCP writeback operations use the same
  required pane locator; no tab-level draft adapter or wire editor identity exists.

Import direction stays one-way: `experiment/v2_gui -> gui.app.measure`, never the
reverse.

## Maintenance Checks

- Cross-module design changes belong in `docs/adr/`.
- App/framework cheat-sheet changes belong here; session-core changes belong in
  `gui/session/README.md`; MCP bridge policy changes belong in
  `mcp/measure/README.md`.
- GUI tests that own `BackgroundRunner` call `quiesce()` before `deleteLater()` or
  process teardown.
