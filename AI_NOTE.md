**Last updated:** 2026-06-08 | **Commit:** 0bbf0c79

# GUI Planning Notes


- `zcu_tools/gui` is a framework layer; experiment-specific knowledge belongs in `experiment/v2_gui`.
- Controller acts as a GUI façade over services including `WorkspaceService` and `StartupService`; it presents typed outcomes but does not coordinate persistence transactions.
- GUI refactors are evaluated against four global gates: Fast Fail, clear ownership, least surprise, and strong typing.
- The per-tab `Session` uses explicit fields: `cfg_schema`, `run_result`, `analyze_result`, `figure`, and `writeback_items`.
- `RunService` exposes public lifecycle signals and Controller should not depend on private runner internals.
- `CfgFormWidget` is a pure viewer that renders a service-owned `SectionLiveField`; the canonical attachment contract is `attach(model: SectionLiveField)`.
- Shared GUI specs are produced by fresh factory functions in `gui/app/main/specs`; do not treat them as reusable singleton objects.
- Path-based schema customization belongs at the adapter level; direct in-place tree mutation inside individual adapters should be avoided.
- `FakeFreqAdapter` lives in `experiment/v2_gui/adapters/fake/freq.py`; the real one-tone frequency adapter `OneToneFreqAdapter` lives in `experiment/v2_gui/adapters/onetone/freq.py`.
- Cross-experiment readout and waveform writeback helpers belong in `experiment/v2_gui/adapters/shared/`, not inside a single adapter module.
- Default `ModuleRefValue` selection from `ModuleLibrary` belongs in `experiment/v2_gui/adapters/shared/`; adapters should describe preference policy instead of scanning candidates inline.
- Writeback editing should reuse the same `CfgSchema`/`CfgFormWidget` path as cfg editing; do not build a parallel editor stack for module or waveform writeback.
- Shared default module selection uses ordered `preferred_names`; when the module type is supported, fallback key/spec inference should come from the helper instead of being redefined in each adapter.
- Writeback persistence belongs in `WritebackService`; adapters only produce typed writeback proposals and the dedicated widget only edits or selects them.
- Writeback proposals are tracked per tab/analyze result: items already applied in the current analysis cycle are treated as completed, start unchecked, and are excluded from the pending-count.
- Adapter refactors use an abstract-minimal-core pattern: the base adapter owns the common `run` and save-path templates, while concrete adapters mainly provide `make_default_cfg`, `build_exp_cfg`, `analyze`, `writeback`, and a filename stem.
- In `build_exp_cfg()`, adapters should mostly reshape flat GUI raw config into the nested experiment config and then delegate validation plus `modules` / `sweep` conversion to `req.ml.make_cfg(...)`; avoid duplicating `_require_*` checks or ad-hoc `from_raw()` fallback in the adapter.
- `Controller/State` is the live SSOT for GUI execution state; `ExpContext` should not be treated as a global frozen snapshot of everything. When consistency is required, prefer small operation-specific request objects over freezing `predictor`, `soc`, or similar live capabilities too early.
- Analysis parameters use adapter-owned dataclass instances plus `Annotated[..., ParamMeta(...)]` UI metadata; they stay separate from `CfgSchema`, and the GUI edits them through a lightweight analysis form rather than the cfg form or LiveModel path.
- `SectionWidget` hides discriminator literals such as `type` and `style` at row-construction time, not just by hiding the child widget.
- `LiveModelEnv` is the single source of truth bridge for `MetaDict`, `ModuleLibrary`, and `EventBus` access inside reactive form fields.
- `ModuleRefWidget` and waveform reference widgets place the expand/collapse control to the left of the selector and own the visibility of their nested config section.
- The left config-panel uses a single small trapezoid boundary handle in both states; the outer shape stays fixed while the arrow conveys expand/collapse direction, and it stays outside `_left_tabs`.
- Form rows in cfg panels use `DontWrapRows` so Module/Waveform selectors stay on one row with their labels.
- The left panel handle is implemented as a small custom-painted edge control whose position follows the splitter boundary after layout settles, not just the raw pane size.
- `tests/gui/conftest.py` owns the GUI test startup policy and forces `QT_QPA_PLATFORM=offscreen` for deterministic headless execution.
- Eval-mode cfg fields keep `ScalarSpec` as the physical-type declaration and move direct-vs-expression behavior into scalar value objects; context-driven channel/eval refresh is owned by `CfgEditorService.refresh_all()`, not by nested fields subscribing directly to global events.
- Channel fields such as `ch` and `ro_ch` use `ScalarSpec(type=int)`; dynamic MetaDict-backed channels are represented as `EvalValue` expressions rather than a separate channel spec/value/widget stack.
- Writeback interface is inline to the Analysis panel rather than a popup dialog. It uses `WritebackWidget` and dynamic collapsible sections.
- `SectionWidget` supports hiding its collapsible header through `no_header` parameter, allowing `ModuleRefWidget` to merge the inner collapsible button with its outer tool button.
- GUI adapter contracts are frozen around generic request objects: `AnalyzeRequest[T_Result, T_AnalyzeParams]`, `SaveDataRequest[T_Result]`, and `WritebackRequest[T_Result, T_AnalyzeResult]`.
- GUI event names should describe the changed resource, not a broad refresh intent: use `CONTEXT_SWITCHED`, `ML_CHANGED`, `SOC_CHANGED`, `PREDICTOR_CHANGED`, and `MD_CHANGED` rather than catch-all context or inspect broadcasts.
- GUI framework lowering and library reference resolution should fail fast. Unknown module-library references, unresolved eval values, unsupported config value types, and invalid widget/model state should raise at the boundary.
- Tab session codec uses module-level functions in `session_codec.py` (`raw_to_schema` / `schema_to_raw`); overall persistence (disk I/O, flush on close) is owned by `PersistenceCaretaker`. `WorkspaceService` coordinates tab create/close/session restore/persist and lifecycle events.
- Startup preferences are persisted via `PersistenceCaretaker` alongside all other app state in `gui_state_v1.json`; `StartupService` coordinates startup project/connection and remembered-device transactions from typed requests.
- Default cfg builders should only emit `EvalValue` when the required `MetaDict` attribute exists in the current context; otherwise they must fallback to explicit `DirectValue` defaults to keep startup behavior deterministic.
- `ExpContext.readiness` distinguishes `DRAFT` startup context from `ACTIVE` file-backed context; real run and save require `ACTIVE`.
- Session/startup caches use strict versioned repositories with typed user-visible failures; legacy payloads are rejected rather than migrated.
- `OperationGate` is the single conflict authority for run, SoC connect, and device mutations; option A permits only one active device mutation globally.
- `DeviceService` owns async device mutation workers and cached `DeviceSnapshot` render state; live device disconnection closes the driver before retaining a memory-only entry.
- `State.add_tab()` receives a `Session` object; `TabService` owns adapter/default cfg construction.
- Save-path suggestion is pure: adapter/tab queries do not create directories or cache suggested state; `SaveService` creates parents at actual save command boundaries.
- Sweep canonical arithmetic belongs to pure `SweepEditor` and `SweepLiveField`; widgets only dispatch intent and render model values.
- `TabService.get_snapshot()` builds immutable `TabSnapshot` render models; `MainWindow` renders snapshots instead of assembling individual tab getters.
- `require_soc_handles()` belongs to `gui.adapter` request validation; the GUI framework must not import `experiment/v2_gui`.
