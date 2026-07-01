# `zcu_tools.gui.app.main` — measure-gui

**Last updated:** 2026-07-01

`gui.app.main` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。實驗領域知識住在 `experiment/v2_gui/` adapter；
framework 只看 `ExpAdapterProtocol`。

## Package Boundaries

- `adapter/`：framework-facing contract、Spec/Value type tree、lowering、analyze
  params、adapter validation。
- `specs/`：program module cfg 的 GUI spec factory。
- `services/`：app service layer。Service 依賴 ports，不直接 import sibling service
  implementation。
- `state.py`：tab/device/result/save-path/version-table SSOT 與主線程 mutators。
- `ui/`：Qt widgets、MainWindow、cfg form、writeback view、feedback/prompt widgets。
- `services/remote/`：GUI process 內的 NDJSON RPC handler；MCP bridge 不在本 package。
- `adapters/`：Qt/liveplot/shutdown 等 driven adapters。

Shared layers:

- `zcu_tools.gui.session`：context、SoC、device、startup、predictor、operation
  handles、operation runner、notify channel、progress service、shared dialogs。
- `zcu_tools.gui.remote`：NDJSON RPC endpoint、framing、wire errors、router base。
- `zcu_tools.gui.plotting`：matplotlib backend、figure routing、host/container/export
  substrate。
- `zcu_tools.mcp.measure`：agent-facing MCP policy layer and tool surface。

## Composition Root

`build_app_services()` constructs the app-local services and injects their driven
ports. `Controller` is a facade over the service bundle; UI and remote code talk
to the controller, not directly to service internals.

Key ownership rules:

- `ContextService` is the only writer for live `MetaDict` / `ModuleLibrary`
  contents.
- `State` owns tab/device/result/save-path resource state and resource versions.
- `GuardService` owns static preconditions and returns typed permits for
  run/save/analyze/writeback.
- `OperationGate` owns dynamic hardware exclusion.
- `OperationHandles` owns async handles, cancellation hooks, and feedback/stop
  channel state.
- `OperationRunner` owns the generic operation lifecycle; each operation supplies
  an `OperationSpec` policy and narrow write ports.

## Run / Analyze Workflow

1. A tab is created from a registered experiment adapter.
2. The tab owns a cfg editor draft backed by `LiveModel`.
3. `GuardService` validates static preconditions and materializes a permit.
4. The operation policy builds worker thunks with the needed ambient scopes:
   plotting, progress, and `ActiveTask` cancellation.
5. `BackgroundRunner` executes blocking work off the Qt main thread and marshals
   terminal callbacks back to the main thread.
6. State write ports record run/analyze/post-analyze results and bump versions.
7. Writeback items are generated from analysis results and edited through the same
   cfg-editor machinery before commit.

`tab.load_data` is the analysis-only entry for canonical result files. It installs
the loaded result into an existing adapter tab, clears stale analysis/writeback
state, and does not backfill the Config tab.

## Config Model

The GUI uses a two-tree model:

- Spec tree: static shape, labels, variants, literal locks, optional/ref rules.
- Value tree: mutable draft data shown by the editor.

`CfgSchema.to_raw_dict(md, ml)` is the lowering boundary. `EvalValue` resolves
against current `MetaDict` when a field is set or lowered. `ValueRef` is
resolve-once: it reads the session `ValueLookup` immediately and stores the
resolved direct scalar in the value tree.

Adapter defaults are assembled in `experiment/v2_gui` with `CfgBuilder` and the
role table. Locked literals are declared in `cfg_spec()`; adapter defaults do not
hand-write those values.

## Operation Model

- Run, device setup, and SoC connect use hardware exclusion.
- Analyze and post-analyze use async handles but no hardware exclusion.
- `OperationChannel` is the ordered cross-thread channel for terminal state,
  user messages, and Send & Stop.
- `NotifyChannel` mirrors the same pattern for `gui_prompt_user`.
- Generic `operation.await` / `operation.poll` report only status and progress;
  products such as figures or fit summaries are read through typed getters.

Cancellation is operation-specific through the registered cancel hook. Run
cancellation sets the `ActiveTask` stop event; experiments that call QICK through
the task runner pass `stop_checkers=[ctx.is_stop]` into `acquire()`.

## Progress And Plotting

Progress is operation-scoped:

- Workers emit Qt-free `ProgressEvent` objects through a `ProgressTransport`.
- `ProgressService` owns per-operation containers and owner-to-operation mapping.
- GUI widgets attach by owner id (`tab_id` or device name).
- Agent polling reads by operation id.

Plotting uses the shared `gui.plotting` backend. Worker-created matplotlib
figures attach to the active `FigureContainer` through routing context; refresh,
activate, and close resolve through the figure registry. Figure export uses fixed
logical sizes so saved images and agent screenshots do not depend on window size.

## Remote / MCP Boundary

`services/remote` is GUI-process policy: method registry, event serialization,
main-thread dispatch, resource-version guard, editor lifecycle, and diagnostics.
It exposes the same controller behavior as the Qt UI.

`zcu_tools.mcp.measure` is the agent-facing bridge: tool declarations,
short-wait wrappers, diagnostics piggyback, operation-handle bookkeeping, stale
guard baseline, and generated/override tool mapping. New GUI RPC methods that
should be agent-accessible need MCP tool mapping and tests.

## Dialog Rules

Dialogs that can live across operations use `open()`, not `exec()`, and keep a
Python reference until they close. Blocking modal helpers are limited to short
direct user actions that do not wait on worker completion.

`MainWindow.open_dialog` / `close_dialog` is the registry path shared by toolbar
actions and remote screenshots. Predictor dialog is persistent hide-on-close;
other dialogs are released when closed. Transient non-modal dialogs that are not
part of the remote named-dialog surface use the shared dialog lifecycle helper
for reference retention and `finished` / `destroyed` cleanup.

## Adapter-Facing Rules

- Adapter `cfg_spec()` and `make_default_value(ctx)` are pure GUI config
  construction.
- `validate_run_request(req, raw_cfg)` is the place for SoC-dependent preflight
  that can fail before starting an operation.
- Adapter `run()` receives a concrete config and performs the experiment.
- `analyze()` / interactive analysis hooks must match `AdapterCapabilities`.
- `get_writeback_items()` returns domain writeback candidates; writeback commit is
  framework-owned.

Import direction stays one-way: `experiment/v2_gui -> gui.app.main`, never the
reverse.

## Maintenance Checks

- Cross-module design changes belong in `docs/adr/`.
- App/framework cheat-sheet changes belong here; session-core changes belong in
  `gui/session/README.md`; MCP bridge policy changes belong in
  `mcp/measure/README.md`.
- GUI tests that own `BackgroundRunner` call `quiesce()` before `deleteLater()` or
  process teardown.
