# `zcu_tools.gui` — GUI framework cheat-sheet

**Last updated:** 2026-07-11（domain-free paired schema assembler）

High-level map of the shared GUI layer. App-specific detail lives in each app's
own README under `app/<name>/`; cross-cutting subpackages (`event_bus`,
`plotting`, `remote`, `session`, `widgets`) are shared by every app.

## Shared Config Core (`cfg/`)

`zcu_tools.gui.cfg` 擁有 Qt-free 的 Spec/Value tree、`CfgSchema` data carrier、
default/inheritance helpers、raw persistence codec，以及generic finished-cfg
validation/lowering。lowering只依賴expression/reference/range三個callable ports，維持
static → optional dynamic → lower、snapshot/relink與error contract（ADR-0046）。

`CfgSchemaAssembler`提供domain-free paired Spec/Value construction：同步declare dotted path、
Fast Fail duplicate/parent conflict與錯誤default carrier、建立choice binding、對齊locked literal，
並以one-shot deep-copy snapshot產生`CfgSchema`。它不知道role、Seed、ExpContext、MetaDict、
ModuleLibrary、logical key或generation policy；measure與autoflux各自保有domain builder，只共用這層
tree mechanics（ADR-0012、ADR-0045）。

generic public names由consumer直接從`zcu_tools.gui.cfg`匯入。measure adapter facade只暴露
framework contract、request/result/writeback/analyze params與protocol signature需要的session
vocabulary，不forward generic cfg names。`zcu_tools.gui.app.autofluxdep.cfg` package barrel只暴露
`NodeCfgSchema`、OverridePlan/policy、module reference spec helpers與其它autoflux-local API；module
conversion/spec functions仍由`cfg.module_adapter`擁有。

`gui.cfg.tree`提供三個existing-tree path operations：`resolve_spec_path`穿section與reference
allowed shapes並拒絕inconsistent leaf types；`read_value_path`/`replace_value_path`穿value section與
reference value且要求leaf已存在。這層不create、不wrap、不處理lock或domain policy；
`CfgSectionValue.with_field`只保留scalar wrapping後委派replace。

Reference節點統一使用`ReferenceSpec(kind=...)`與`ReferenceValue`；`kind`是shared core只
轉送的app-local opaque id。module/waveform factory、resolver與converter policy留在各app，
既有`module_ref`/`waveform_ref` persistence wire shape不變。

`zcu_tools.gui.cfg.binding`擁有Qt-free的`CfgDraft`、field tree與sweep editors。
`CfgDraft`集中snapshot、validity、refresh與close lifecycle；field只依賴expression evaluator、
opaque option provider與reference catalog三個窄ports，不持controller/environment aggregate。
scalar field在建構與修改時依宣告型別Fast Fail；section/reference的whole-value更新先完成純資料
key檢查與replacement build，再一次提交，不會留下partial tree或中間validity event。leaf edit只
沿ancestor傳遞dirty event，`CfgFormWidget`在UI boundary materialize一次`CfgSchema`。
close會遞迴關閉整棵field tree並清callbacks，close前cached的root/child之後讀寫或refresh都
Fast Fail。reference catalog以shape label與optional materialized value精確區分missing、
unsupported與corrupt。widget只attach draft並render `draft.root`，detach不會close
service-owned draft。

此package不import `gui.app.*`、`experiment.*`、Qt、`meta_tool`、`notebook`或`device`，
也沒有broad environment object或global resolver registry。scalar option source與reference
kind都是shared只轉送的opaque string；measure與autofluxdep各自提供
app-local ports與module shape policy；autofluxdep不經measure lowering/conversion。

## Process Runtime (`runtime.py`)

`gui.runtime` owns process-level startup mechanics for launchable GUI apps:
logging, matplotlib plot policy, `QApplication` creation, shared plot-host
lifecycle, remote-control option construction, adapter start/stop, and integer
exit-code handling. Apps expose a fixed `GuiRuntimeBehavior.spec` class variable
for static process contract and implement `assemble(control)` for app-local
controller/window/adapter wiring. Launch-time CLI values stay in
`GuiLaunchOptions` and behavior constructor arguments. App modules expose
behavior classes; standalone `script/run_*_gui.py` launchers are the process
entrypoints and call `launch_gui_runtime(...)` directly.

The runtime seam deliberately stops above remote method/domain/session policy:
`gui.remote` still owns transport, each app owns dispatch/domain handlers, and
`gui.session` owns measurement-session primitives. All four standalone GUI
launchers (`measure`, `autofluxdep`, `fluxdep`, `dispersive`) use this runtime
directly; measurement-session apps keep their domain composition in app behavior
classes and launcher-provided factories.

`gui.launcher` is the import-light CLI edge companion for runtime launchers. It
declares shared logging/control/project flags and converts parser output into
`GuiLaunchOptions` / `ProjectInfo`. It deliberately avoids importing Qt or
matplotlib or app modules at module import time, so scripts can import it before
runtime applies plot policy.

## Project / Result Scope

`ResultScopeManager` scans `result/**/params.json` under the project root and
treats each hit as a selectable result scope. Measurement-session setup dialogs
(measure/autofluxdep) use that scope to apply startup context; analysis dialogs
(fluxdep/dispersive) use the same discovery in `widgets.ProjectDialog` as a
dropdown picker only, leaving typed paths and Browse flows available.
Measurement-session discovery is snapshot-cached: ordinary dialog reopen/apply
paths reuse the last scan, while an explicit refresh requests a new scan. Applying
a previously selected scope validates against the snapshot first and only rescans
when the selected scope is absent.

## Dialogs — always non-blocking

Every app embeds a control socket whose RPC handler runs on the Qt event loop,
so **all dialogs (and message boxes) launch with `open()`, never `exec()`**.
A blocking `exec()` would freeze the event loop until the user dismisses the
dialog, stalling the control socket (and, for measure, deadlocking cross-thread
marshalling). Read a dialog's outcome from its `accepted` / `finished` signal
instead of `exec()`'s return value, set `WA_DeleteOnClose`, and hold an instance
reference so `open()`'s immediate return does not let it be garbage-collected.
The measure registry path (`MainWindow.open_dialog` / `close_dialog`) is detailed
in `app/main/services/remote/README.md`. The sole intentional `exec()` is the
global unhandled-exception hook in `app/main/utils/error_handler.py`, where the
process is already crashing and the message must block.

Short-lived modal confirmations, error reports, and text prompts are the
deliberate exception. A synchronous `QMessageBox.question` / `.warning` /
`.critical` or `QInputDialog.getText` may block: it is raised by a direct,
synchronous user action while no operation is in flight, carries no
long-running flow, and never marshals a worker result back to the main
thread. Its nested modal loop still services the control socket's queued
notifier, so it neither stalls startup nor deadlocks cross-thread
marshalling. The `open()` rule still governs every dialog that hosts a
long-running flow or can surface while an operation is running.

Widgets that need information messages, warnings, critical errors,
confirmations, or explicit destructive confirmations receive the shared
`DialogPresenter` port. Production uses `QtDialogPresenter`; when a
`DialogRefStore` is supplied, information, warning, and critical boxes stay
non-blocking and retained through the shared lifecycle helper. Close paths or
operation-running prompts use callback-style `confirm_async` /
`destructive_confirm`, so the Qt event loop keeps pumping while the decision is
pending. Tests inject a recording presenter instead of monkeypatching
`QMessageBox`, so dialog decisions are scripted at the object boundary.

Non-modal dialogs opened with `open()` use the shared dialog lifecycle helper,
or an equivalent named registry when remote screenshot/list semantics require
stable names. Either path keeps a Python reference until `finished` / `destroyed`
cleanup runs.

## Shared Qt Cfg Widgets (`widgets/cfg/`)

`zcu_tools.gui.widgets.cfg`擁有`CfgFormWidget`、field widgets與presentation-only
decoration contract。widget attach service-owned `CfgDraft`並render `draft.root`；detach會
unsubscribe並刪除Qt tree，但不會close draft。shared widget只import Qt、`gui.cfg`、
`gui.cfg.binding`與shared spinbox，不知道app/controller/session/EventBus/ModuleLibrary或
experiment policy。

每個`CfgFormWidget`持有一個`FrozenFieldRendererRegistry`。沒有顯式注入時，
`default_cfg_renderers()`會建立全新的builder、為六個exact field types註冊固定
`FieldRenderer(field, context)` factory再freeze。immutable `FieldRenderContext`只攜帶path、
top-level標記、label width、decoration resolver、text enhancer與同一frozen registry；不攜帶
controller/service/app/runtime資料。root、section child與reference subtree都走registry
`render()`，沒有consumer-side constructor分支、module-global mapping、decorator、string key或
inheritance fallback。registration會先驗factory call shape，render則驗QWidget與field-widget
protocol。

`CfgFormWidget.attach()`先成功建立完整root widget，再訂閱caller-owned draft；build失敗不留下
draft callbacks。detach以stable Python callback解除change/validity subscriptions並刪除Qt tree，
仍不close draft。

`CfgFormWidget` accepts an optional field decoration provider keyed by full dotted
cfg path. The shared renderer applies `hidden`, `enabled`, `tone`, `badge`, and
`tooltip` metadata without changing the live value model. App-specific semantics
stay outside the shared layer: for example, autofluxdep uses the same hook to mark
generated Default cfg fields from its `OverridePlan`。measure則以generic
`TextInputEnhancer` seam安裝app-local value-source completion，shared widget不import session。

Normal `LiteralSpec` rows remain hidden, but a decoration may explicitly unhide a
literal when an app needs to show a generated read-only value in the form.

## EventBus Lifecycle

`BaseEventBus` stays Qt-free and payload-type keyed. `subscribe()` returns an
idempotent cleanup handle, and long-lived windows / bridges group those handles
so close/stop tears down callbacks explicitly. Event delivery rules are
unchanged: callbacks for the concrete payload type run in order, and one
subscriber raising is logged without aborting later subscribers.
Remote bridges subscribe their EventBus push callbacks before opening the socket;
startup subscription failure is fatal and rolls back partial subscriptions, while
runtime subscriber exceptions remain isolated by the EventBus.

## Logging (`logging_setup.py`)

`logging_setup.setup_gui_logging` is the single place that decides *how* every
GUI entry point configures logging. All four `script/run_*_gui.py` launchers and
the measure MCP server (`mcp/measure/server.py:main`) call it instead of each
rolling their own handler set.

Key invariants:

- **The file handler is attached at the whole `zcu_tools.gui` namespace** (plus
  any `extra_namespaces` an entry point needs — measure adds
  `zcu_tools.experiment.v2_gui`; the MCP server adds `zcu_tools.mcp`). Attaching
  at the package root, not an app sub-namespace, is deliberate: cross-cutting
  subpackages (`event_bus`, `plotting`, `remote`, `session`) are siblings of the
  app namespace, and a handler scoped to one app would silently miss them. That
  missed-sibling gap is the bug this scheme exists to prevent; the regression
  test is `tests/gui/test_logging_setup.py`.
- **Per-session timestamped files** under `<repo>/logs/<group>/<app>/` (`group`
  = `gui` for launchers, `mcp` for the server). Each launch writes its own file,
  so a previous session's evidence is never overwritten. On startup the helper
  purges all but the newest `retain` (default 10) files in that directory.
- **Levels:** file handler at DEBUG, stderr handler at WARNING. High-frequency
  bookkeeping (operation create, background submit) logs at DEBUG; lifecycle
  events (operation settle, connect/device result, persistence flush/restore,
  writeback apply) at INFO; failures at WARNING/ERROR (worker exceptions carry
  `exc_info` so the real traceback survives the cross-thread marshal).
- A `--log-file` CLI override (when a launcher exposes one) wins over the
  per-session scheme: the explicit path is used verbatim and no purge runs.
- Repeated setup in the same process replaces handlers previously installed by
  this helper instead of stacking duplicate stderr/file handlers. User-installed
  handlers are left untouched.

The MCP server uses stdout for its JSON-RPC transport, so its logging never
touches stdout — the helper only adds a stderr handler and the DEBUG file
handler.

## Qt Typing Invariants

Qt override signatures must match the binding stubs exactly, including nullable
event parameters and stub parameter names. `closeEvent` implementations use
`a0: QCloseEvent | None`; `eventFilter` implementations use
`a0: QObject | None, a1: QEvent | None` and immediately delegate nullable inputs
to `super()`.

Qt table header accessors are treated as optional by the stubs. Code that
configures `horizontalHeader()` must assert the header is not `None` before
calling resize APIs; the assert records the widget invariant and keeps the
runtime failure explicit if a binding ever violates it.
