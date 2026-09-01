# `zcu_tools.gui.app.main` — measure-gui

**Last updated:** 2026-09-01 — progress, writeback applied state, Data and Run actions

`gui.app.main` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。實驗領域知識住在 `experiment/v2_gui/` adapter；
framework 只看 `ExpAdapterProtocol`。

Main 擁有 `AppPersistedState` codec/version、filename、originator、restore presentation 與
lifecycle-only triggers；disk mechanism 使用 `gui.session.persistence.SingleFileCaretaker`。

## Package Boundaries

- `adapter/`：framework-facing contract、measure-owned finished-cfg ports、analyze params、
  adapter validation與protocol signature需要的session vocabulary；不forward generic cfg API。
- `specs/`：`gui.measure_cfg`的main policy adapter；只綁定Arb asset choices與readout
  cross-shape inheritance，不擁有program field/label清單。
- `cfg_schemas.py`：main raw/typed cfg normalization與policy facade；全七種module/六種waveform的
  spec walk、missing/nested/reference規則由`gui.measure_cfg`materializer擁有。
- `services/`：app service layer。Service 依賴 ports，不直接 import sibling service
  implementation；package `__init__` 只做 lazy public re-export，讓
  `services.remote.method_specs` public import path 不載入 Qt-bound service code。
- `state.py`：tab/device/pane/path/version-table SSOT 與主線程 mutators；固定的
  Run、Analysis、Post-Analysis、Save pane 各自擁有自己的 resource。`running_tab_id`
  是唯一 run ownership 狀態，tab interaction 的 `is_running` 由它投影。
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
  capture (S2): at draft creation it snapshots the destination `ExpContext` and
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
  Current → Proposed, the bordered ledger hugs its rendered rows with Apply
  Selected immediately after it; long content grows naturally and delegates
  vertical scrolling to the existing Analysis pane rather than owning a nested
  scroll/cap lifecycle.
  `RenderHost` is pane-aware (run | analysis | post_analysis) and the worker
  captures its pane's container at start — switching the visible subtab never
  retargets the worker (ADR-0017). `ExpTabWidget` delegates the Data pane to an
  internal `ArtifactSaveCenter` which把capability-driven `Load Data` / `Save All`
  action row放在`Measurement data`card之前，同時擁有capability-driven artifact rows、
  high-contrast status rendering and the tab-local status lifecycle derived from
  result availability, path/comment edits and true terminal save outcomes (not
  persisted across process), with figure-gated save enablement while status still
  tracks result lifecycle. The center owns the saveability decision and the ordered
  Save All sequence (analysis→post→data with Fast Fail, never rolling back prior
  successes); tracker/invariant failures Fast Fail and operational failures are
  presented centrally, and async data completion is routed to the center.
  Analysis/Post panes no longer own image-path/Save Image; Run's live figure
  remains view-only (display + screenshot, no canonical Save). Data's right pane
  is a `DataFigurePreviewGallery` Variant A responsive rail: capability-declared at
  construction (Run always, Analysis/Post only when supported), viewport-driven
  mosaic reflow — narrow single-column vertical vs wide three-card Run-left
  spanning two rows with Analysis/Post on the right, two-card side-by-side,
  and single full-width at the two-minimum-width-cards-plus-spacing breakpoint
  (gallery's own viewport, not window, no persisted toggle) — scrollable cards
  with named empty/unavailable states, raster-only presentation cache (pixmap/text)
  fed by an injected `Figure -> PNG bytes` adapter — production uses the existing
  fixed-size figure renderer with size restore — per-card render caches the
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
  for result focus, plot hosting, interactive-widget lifecycle, figure reads,
  and persisted panel geometry; the tab does not expose its Qt containers.
- `services/remote/`：GUI process 內的 NDJSON RPC handler；MCP bridge 不在本 package。
- `driven/`：measure app-local Qt/liveplot driven adapters；與 `adapter/` 的 experiment
  framework contract 分開命名。

Shared layers:

- `zcu_tools.gui.cfg`：Qt-free Spec/Value model、`CfgSchema` data carrier、inheritance、
  persistence codec、domain-free raw spec walk與generic finished-cfg validation/lowering ports。
- `zcu_tools.gui.measure_cfg`：Qt-free closed program module/waveform shape catalog與fresh Spec
  factories，以及program missing/nested/reference/subset materialization policy；main以app-local policy
  啟用Arb choices、readout inheritance與完整7+6 materializable catalog。
- `zcu_tools.gui.widgets.cfg`：shared `CfgFormWidget`、field renderers、decoration contract與
  instance-owned frozen exact renderer registry。
- `zcu_tools.gui.session`：context、SoC、device、startup、predictor、operation
  handles、operation runner、notify channel、progress/shutdown service、shared dialogs。
- `zcu_tools.gui.remote`：NDJSON RPC endpoint、framing、wire errors、router base。
- `zcu_tools.gui.plotting`：matplotlib backend、figure routing、host/container/export
  substrate。
- `zcu_tools.mcp.measure`：agent-facing MCP policy layer and tool surface。

## Composition Root

`MeasureGuiBehavior` is the process-runtime behavior for the shared
`gui.runtime` launcher seam. It assembles `State`, `Controller`, `MainWindow`,
persistence caretaker, startup dialog, and the app-local `RemoteControlAdapter`
without owning process policy such as logging, matplotlib backend selection,
`QApplication`, control option construction, or exit-code handling. The
standalone launcher is the process entrypoint; this module does not expose a
second `run_app` path.

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
active/running identity, tab read model, cfg schema commits, save path overrides)
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
payload routing (ADR-0048). The coordinator speaks to `MainWindow` through a narrow
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
but have no Qt reaction.
Analyze forms commit `QLineEdit` changes on `editingFinished` so partial text does
not trigger interaction refresh; choice, checkbox, and numeric controls retain
immediate value-change commits. The shared cfg widget layer owns this signal policy.
`MainWindowToolbar` owns the top toolbar widgets and slash-grouped new-tab menu;
it reports selected actions back through a narrow `MainWindowToolbarHost` surface
instead of reaching into `Controller` directly.

Key ownership rules:

- Caller-correctable app/service failures由producer透過`ExpectedError`明列invalid-input或
  failed-precondition category；shared remote dispatch統一投影generic category。handler只保留
  request coercion與structured/domain-special wire policy，不擁有分類registry。
  ordinary/provider/persistence/invariant failures保持unexpected並保留controller traceback
  （ADR-0047）。
- `ContextService` is the only writer for live `MetaDict` / `ModuleLibrary`
  contents.
- `State` owns tab/device/pane/path resource state and resource versions. Pane swaps
  happen on the owner thread and return retired resources for post-commit cleanup.
- `GuardService` owns static preconditions and returns typed permits for
  run/save/analyze/writeback.
- `OperationGate` is the app-local thin wrapper over the shared
  `RunBlocksHardwareGate` hardware exclusion policy。active lease另投影captured
  origin、domain note與duration；`state.hardware_gate`是read-only internal RPC，
  `gui_overview.hardware_gate.active`提供MCP orientation snapshot。
- `OperationHandles` owns async handles, cancellation hooks, and feedback/stop
  channel state.
- `OperationRunner` owns the generic operation lifecycle; each operation supplies
  an `OperationSpec` policy and narrow write ports. Terminal policy exceptions
  are contained in the shared runner so handles settle and exclusion leases release.

## Run / Analyze Workflow

1. A tab is created from a registered experiment adapter.
2. The tab owns a service-managed cfg editor session backed by `CfgDraft`.
   Run renders that draft through the sole shared cfg tree (S1); Analysis
   renders its params through the app-local 13 px ledger with whole-header
   folding and a full-width `Analyze` immediately below parameters.

3. `GuardService` validates static preconditions and materializes a permit.
4. The operation policy builds worker thunks with the needed ambient scopes:
   plotting, progress, `Schedule` cancellation, and device setup cancellation.
5. `BackgroundRunner` executes blocking work off the Qt main thread and marshals
   terminal callbacks back to the main thread.
6. Run/analyze services depend on narrow State ports (`RunStatePort` /
   `AnalyzeStatePort`) for busy checks, request-building reads, and result writes.
7. Writeback items are generated from analysis results and edited through the same
   cfg-editor machinery before commit; `WritebackService.create_draft` snapshots
   the destination `ExpContext` at creation and the ledger shows
   `current_summary` → `proposed_summary` per item (S2). Scalar MetaDict items
   show concrete values; module/waveform items show bounded change summaries and
   keep full cfg editing in `Edit`. Primary and post workflows own proposal timing;
   the Writeback service remains stage-free.

`tab.load_data` is the analysis-only entry for canonical result files. It installs
the loaded result into an existing adapter tab, clears stale analysis/writeback
state, and does not backfill the Config tab. The Guard and LoadService both enforce
the adapter's import-validated `capabilities.load_data` gate.

### Pane-owned lifecycle

`Session` is the aggregate root and its fixed pane carriers are the resource owners:
Run stores only the run result/source, Analysis and Post-Analysis each store params,
result, canonical figure and an opaque writeback draft (with S2 baseline snapshot),
and Save stores the data-path override. Analysis and Post-Analysis image-path
overserides are independent resources; the read model projects data, analysis-image
and post-analysis-image paths separately. Run live figures remain view-only and
are not stored in State. Writeback baseline is a display-only draft-creation
snapshot；同一opaque draft另擁有per-item applied state，只有成功write包含的items才標記applied，
selection本身不改狀態，retarget或內容修改會重設；同kind items不得指向重複destination，
避免batch覆寫卻誤標applied。狀態不跨draft/process持久化，也不提供
concurrent-write detection或apply-conflict policy。

Analysis/Post result services prepare proposals, figures and drafts before calling one
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
the gallery renders to PNG via the fixed-size adapter with size restore, holds
only the raster cache (original pixmap) and isolates per-card failures with
aspect-fit scaling (`KeepAspectRatio`) that never exceeds the image viewport.
Viewport-driven mosaic reflow (gallery's own width vs two-minimum-width-cards
threshold) is presentation-only and never moves figure ownership, adds a second
canvas, or changes ADR-0048 reactions. No competing state owner is introduced.

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

CfgEditor在app seam解碼`ValueRef`，並以typed `CfgEdit` batch依序操作binding target。
Batch維持fail-fast/non-atomic；只有reference shape edit列出前後path set，成功回final net diff，
每筆成功edit仍各自bump version與觸發subscriber-aware lazy push。

The GUI uses a two-tree model:

- Spec tree: static shape, labels, variants, literal locks, optional/ref rules.
- Value tree: mutable draft data shown by the editor.

`adapter.lowering.schema_to_raw_dict(schema, md, ml)` is the finished-cfg lowering
boundary. `CfgSchema` 本身只保存 shared spec/value data。`EvalValue` resolves
against current `MetaDict` when a field is set or lowered. `ValueRef` is
resolve-once: it reads the session `ValueLookup` immediately and stores the
resolved direct scalar in the value tree.

generic model、spec walk、inheritance、codec、static/dynamic validation與lowering由
`zcu_tools.gui.cfg`擁有，consumer直接從shared owner匯入。measure adapter只把current
`MetaDict` expression evaluator、measure-owned module/waveform resolver與`SweepCfg`
factory組成三個窄ports；adapter package不import或forward shared cfg public names，也不保留
第二份algorithm或model/inheritance/codec implementation（ADR-0045、ADR-0046）。

Module與waveform field在shared model都使用`ReferenceSpec(kind=...)` / `ReferenceValue`。
measure-owned pulse/waveform spec factory顯式設定`kind="module"`或`kind="waveform"`，
`MeasureCfgBindings`依`spec.kind`選擇精確的ModuleLibrary store/materializer facade，並提供expression、
dynamic scalar options與ValueRef resolution policy；widget只讀field API，shared cfg不認識這些
app-local policy。device selector是required string `ScalarSpec`，wire value維持`DirectValue(str)`。
measure composition在`ui/cfg_binding.py`以generic `QLineEdit -> keepalive object` enhancer seam安裝
`ValueSourceInputController`，保留eval input的completion與resolve-on-space；shared binding仍
不import ValueRef或session，shared widget只接收generic enhancer callable。

ModuleLibrary reference enumeration只以`gui.measure_cfg.program_shape_for_input`讀root discriminator；
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
re-adding the same key relinks it, including restored overridden refs whose key
is absent at load time, while persistence can still serialize the snapshot
without consulting `ModuleLibrary`.

Adapter cfg authoring lives in `experiment/v2_gui` as a context-free
`MeasureCfgDefinition`. A single `MeasureCfgBuilder` declaration fixes static shape,
field order, role, lock and deferred typed Seed; fresh `instantiate(ctx)` only resolves
value defaults, then `BaseAdapter.make_default_cfg` validates the finished schema.
The framework protocol does not expose a static spec query. Shared
`CfgSchemaAssembler` owns only paired-tree mechanics and has no measure domain/context
knowledge (ADR-0012、ADR-0045).

`CfgFormWidget`由`zcu_tools.gui.widgets.cfg`擁有，measure UI直接import shared owner。
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
- `NotifyChannel` mirrors the same pattern for `gui_prompt_user`.
- `FeedbackDockController` owns the docked feedback panel, target-tab
  resolution, and op-count plus agent-presence gate; `MainWindow` keeps the
  public render-view refresh façade.
- Generic `operation.await` / `operation.poll` report only status and progress;
  products such as figures or fit summaries are read through typed getters.

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

Plotting uses the shared `gui.plotting` backend. Worker-created matplotlib
figures attach to the active `FigureContainer` through routing context; refresh,
activate, and close resolve through the figure registry. Figure export uses fixed
logical sizes so saved images and agent screenshots do not depend on window size.
Analysis start leaves plot teardown to the render host. Terminal domain facts
restore retained figures only after failure, cancellation, or start rejection;
successful content commits attach new figures once. Run failure keeps the
placeholder because run start invalidates prior results, while loaded-result
commit explicitly clears canvases when the new State has no figure.

## Remote / MCP Boundary

`services/remote` is GUI-process policy: method registry, event serialization,
main-thread dispatch, resource-version guard, editor lifecycle, and diagnostics.
It exposes the same behavior as the Qt UI. Context/value/md/ml RPC handlers use
the controller-exposed `ContextControlPort` facet; device RPC handlers use
`DeviceControlPort` for device lifecycle/query/progress; predictor RPC handlers
use `PredictorControlPort` for predictor load/query/compute. SoC/startup
handlers remain on the app controller façade because they span project setup and
connection policy rather than a single session-control domain.

`zcu_tools.mcp.measure` is the agent-facing bridge: tool declarations,
short-wait wrappers, diagnostics piggyback, operation-handle bookkeeping, stale
guard baseline, and generated/override tool mapping. New GUI RPC methods that
should be agent-accessible need MCP tool mapping and tests.

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
controller only for measure-only CfgEditor create/modify and role-catalog actions.
`SetupDialog` receives `setup_control`, so project/context/SoC bootstrap UI no
longer depends on the concrete controller façade. The persistent measure
`PredictorDialog` receives both `predictor_control` and `device_control`, so the
shared dialog can refresh cached device values on every reopen without depending
on the concrete controller.

## Interactive Analysis Seam

`adapter.types` owns the Qt-free closed control vocabulary
`InteractiveControl = ButtonControl | ToggleControl` (`ControlKey` as stable
identity, `label`, typed callback, and for toggles an exact `bool initial`)
and the `InteractiveSession` Protocol (`controls() -> tuple[InteractiveControl, ...]`,
pointer hooks, `info_text()` and `finish()`). Concrete sessions such as
`FluxPickSession` own the domain callback mapping (`Conjugate Line` toggle →
`TwoLinePicker.set_conjugate`, `Auto Align` → background alignment,
`Swap Lines` → swap+redraw) in declaration order; they cache the terminal
`finish()` result and ignore subsequent domain input and late background
completions. `InteractiveAnalysisWidget` is the generic host: it validates the
declaration and lowers it to Qt (`ButtonControl` → `QPushButton`,
`ToggleControl` → `QCheckBox`) without comparing domain keys, applies a
toggle's `initial` before connecting its signal so construction never fires the
callback, and reads the surface only once at bind.

Ordering: control surface is bound once; toggle `initial` is set before signal
connection. Errors: bind validates the whole surface before mounting — empty or
whitespace-only `key`/`label`, duplicate `key`, unsupported variant,
non-callable callback, or non-`bool` toggle `initial` Fast Fail with no partial
mount; construction-time `ButtonControl`/`ToggleControl` invariants also Fast
Fail; repeat `bind()` Fast Fails. Lifecycle: `Done` closes the input gate
first — it disables the checkbox, all buttons, the Done button itself and canvas
pointer forwarding, then invokes `on_done` exactly once; subsequent control or
pointer events are ignored, and a finished session's late background completion
does not mutate the picker or result. Variation is closed: new kinds are added
only for a real need, as a new `InteractiveControl` union member with an
exhaustive renderer; no generic widget factory, registry, dynamic surface or
cross-process/serialization representation exists.

## Adapter-Facing Rules

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

Import direction stays one-way: `experiment/v2_gui -> gui.app.main`, never the
reverse.

## Maintenance Checks

- Cross-module design changes belong in `docs/adr/`.
- App/framework cheat-sheet changes belong here; session-core changes belong in
  `gui/session/README.md`; MCP bridge policy changes belong in
  `mcp/measure/README.md`.
- GUI tests that own `BackgroundRunner` call `quiesce()` before `deleteLater()` or
  process teardown.
