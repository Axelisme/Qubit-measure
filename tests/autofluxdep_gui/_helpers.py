"""Shared test helpers — build ad-hoc providers without a real experiment.

The dependency-model tests need small providers with arbitrary declarations and
a scripted ``produce``. ``make_builder`` returns a ``Builder`` subclass instance
whose Node delegates to a supplied ``produce_fn(env, snapshot) -> Patch`` (or
returns an empty Patch). ``place`` wraps a Builder into a ``PlacedNode`` with
params. Together they replace the old ``NodeSpec`` + injected ``run_node``.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    NodeCfgSchema,
    OverridePlan,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import (
    Builder,
    Node,
    PlacedNode,
    RunEnv,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    ModuleRefValue,
    WaveformRefValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore, Notify

ProduceFn = Callable[[RunEnv, Snapshot], Patch]
NodeField = tuple[str, CfgNodeSpec, Any]


@dataclass(frozen=True)
class NodeFieldSpec:
    """One logical test knob mounted as a leaf inside a UI section."""

    logical_key: str
    section_key: str
    field_key: str
    spec: CfgNodeSpec
    default: Any

    @property
    def path(self) -> str:
        return f"{self.section_key}.{self.field_key}"


@dataclass(frozen=True)
class NodeFieldDecl:
    """One logical test knob before it is mounted into a UI section."""

    logical_key: str
    field_key: str
    spec: CfgNodeSpec
    default: Any


@dataclass(frozen=True)
class NodeSectionSpec:
    """A test-only UI section grouping one-or-more logical knobs."""

    key: str
    label: str
    fields: tuple[NodeFieldSpec, ...]


@dataclass(frozen=True)
class NodePathSpec:
    """One logical test knob mounted at an explicit cfg value-tree path."""

    logical_key: str
    path: str
    spec: CfgNodeSpec
    default: Any


def node_field(
    logical_key: str, field_key: str, spec: CfgNodeSpec, default: Any
) -> NodeFieldDecl:
    """Declare one ad-hoc test knob without repeating its section key."""
    return NodeFieldDecl(
        logical_key=logical_key,
        field_key=field_key,
        spec=spec,
        default=default,
    )


def node_path(
    logical_key: str, path: str, spec: CfgNodeSpec, default: Any
) -> NodePathSpec:
    """Declare one test knob at an explicit dotted value-tree path."""
    return NodePathSpec(
        logical_key=logical_key,
        path=path,
        spec=spec,
        default=default,
    )


def node_section(key: str, label: str, *fields: NodeFieldDecl) -> NodeSectionSpec:
    """Mount pending ad-hoc test knob declarations under one section key."""
    return NodeSectionSpec(
        key=key,
        label=label,
        fields=tuple(
            NodeFieldSpec(
                logical_key=field.logical_key,
                section_key=key,
                field_key=field.field_key,
                spec=field.spec,
                default=field.default,
            )
            for field in fields
        ),
    )


def flat_node_schema(fields: tuple[NodeField, ...]) -> CfgSchema:
    """Build a test-only flat cfg schema from ``(key, spec, default)`` tuples."""
    _ensure_unique("node field key", (key for key, _, _ in fields))
    return CfgSchema(
        spec=CfgSectionSpec(fields={key: node_spec for key, node_spec, _ in fields}),
        value=CfgSectionValue(
            fields={
                key: _default_value_for(node_spec, default)
                for key, node_spec, default in fields
            }
        ),
    )


def path_node_schema(
    fields: tuple[NodePathSpec, ...],
    *,
    section_labels: dict[str, str] | None = None,
) -> NodeCfgSchema:
    """Build a test-only node schema from explicit logical-to-cfg paths."""
    _ensure_unique("node logical key", (field.logical_key for field in fields))
    root_spec = CfgSectionSpec(fields={})
    root_value = CfgSectionValue(fields={})
    logical_paths: dict[str, str] = {}
    labels = section_labels or {}

    for field_spec in fields:
        _validate_node_path_spec(field_spec)
        _insert_path_field(root_spec, root_value, field_spec, labels)
        logical_paths[field_spec.logical_key] = field_spec.path

    return NodeCfgSchema(
        CfgSchema(spec=root_spec, value=root_value),
        logical_paths=logical_paths,
    )


def sectioned_node_schema(sections: tuple[NodeSectionSpec, ...]) -> NodeCfgSchema:
    """Build a test-only sectioned node schema with logical-key projection."""
    _ensure_unique("node section key", (section.key for section in sections))

    root_spec_fields: dict[str, CfgNodeSpec] = {}
    root_value_fields: dict[str, Any] = {}
    logical_paths: dict[str, str] = {}

    for section in sections:
        _validate_path_part("section key", section.key)
        _ensure_unique(
            f"field key in section {section.key!r}",
            (field_spec.field_key for field_spec in section.fields),
        )
        section_spec_fields: dict[str, CfgNodeSpec] = {}
        section_value_fields: dict[str, Any] = {}

        for field_spec in section.fields:
            _validate_node_field_spec(section.key, field_spec)
            if field_spec.logical_key in logical_paths:
                raise ValueError(
                    f"Duplicate node logical key {field_spec.logical_key!r}"
                )
            section_spec_fields[field_spec.field_key] = field_spec.spec
            section_value_fields[field_spec.field_key] = _default_value_for(
                field_spec.spec, field_spec.default
            )
            logical_paths[field_spec.logical_key] = field_spec.path

        root_spec_fields[section.key] = CfgSectionSpec(
            label=section.label,
            fields=section_spec_fields,
        )
        root_value_fields[section.key] = CfgSectionValue(fields=section_value_fields)

    return NodeCfgSchema(
        CfgSchema(
            spec=CfgSectionSpec(fields=root_spec_fields),
            value=CfgSectionValue(fields=root_value_fields),
        ),
        logical_paths=logical_paths,
    )


def read_value_tree(schema: NodeCfgSchema) -> dict[str, Any]:
    """Return a JSON-friendly value tree for white-box test assertions."""
    return _jsonify_value_tree(schema.schema.value)


def _default_value_for(spec: CfgNodeSpec, default: Any) -> Any:
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec default must be a SweepValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, ScalarSpec):
        if isinstance(default, (DirectValue, EvalValue)):
            return default
        return DirectValue(default)
    raise TypeError(f"Unsupported node field spec: {type(spec).__name__}")


def _validate_node_field_spec(section_key: str, field_spec: NodeFieldSpec) -> None:
    _validate_path_part("logical key", field_spec.logical_key)
    _validate_path_part("field key", field_spec.field_key)
    if field_spec.section_key != section_key:
        raise ValueError(
            f"Node field {field_spec.logical_key!r} declares section "
            f"{field_spec.section_key!r}, but is mounted under {section_key!r}"
        )
    if not isinstance(field_spec.spec, (ScalarSpec, SweepSpec)):
        raise TypeError(
            f"Unsupported node field spec for {field_spec.logical_key!r}: "
            f"{type(field_spec.spec).__name__}; only ScalarSpec and SweepSpec "
            "are supported"
        )


def _validate_node_path_spec(field_spec: NodePathSpec) -> None:
    _validate_path_part("logical key", field_spec.logical_key)
    for part in field_spec.path.split("."):
        _validate_path_part("path part", part)
    if not isinstance(field_spec.spec, (ScalarSpec, SweepSpec)):
        raise TypeError(
            f"Unsupported node field spec for {field_spec.logical_key!r}: "
            f"{type(field_spec.spec).__name__}; only ScalarSpec and SweepSpec "
            "are supported"
        )


def _insert_path_field(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    field_spec: NodePathSpec,
    labels: dict[str, str],
) -> None:
    spec_section = root_spec
    value_section = root_value
    parts = field_spec.path.split(".")
    prefix_parts: list[str] = []
    for part in parts[:-1]:
        prefix_parts.append(part)
        prefix = ".".join(prefix_parts)
        existing_spec = spec_section.fields.get(part)
        if existing_spec is None:
            existing_spec = CfgSectionSpec(label=labels.get(prefix, part), fields={})
            spec_section.fields[part] = existing_spec
        if not isinstance(existing_spec, CfgSectionSpec):
            raise ValueError(
                f"Node cfg path {field_spec.path!r} crosses non-section {prefix!r}"
            )
        existing_value = value_section.fields.get(part)
        if existing_value is None:
            existing_value = CfgSectionValue(fields={})
            value_section.fields[part] = existing_value
        if not isinstance(existing_value, CfgSectionValue):
            raise ValueError(
                f"Node cfg path {field_spec.path!r} crosses non-section value "
                f"{prefix!r}"
            )
        spec_section = existing_spec
        value_section = existing_value
    leaf = parts[-1]
    if leaf in spec_section.fields:
        raise ValueError(f"Duplicate node cfg path: {field_spec.path!r}")
    spec_section.fields[leaf] = field_spec.spec
    value_section.fields[leaf] = _default_value_for(field_spec.spec, field_spec.default)


def _validate_path_part(kind: str, value: str) -> None:
    if not value:
        raise ValueError(f"Node {kind} must not be empty")
    if "." in value:
        raise ValueError(f"Node {kind} must not contain '.': {value!r}")


def _ensure_unique(kind: str, values: Any) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Duplicate {kind}: {', '.join(sorted(duplicates))}")


def _jsonify_value_node(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, CfgSectionValue):
        return _jsonify_value_tree(value)
    if isinstance(value, ModuleRefValue):
        return {
            "__kind": "module_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": bool(value.is_overridden),
            "value": _jsonify_value_tree(value.value),
        }
    if isinstance(value, WaveformRefValue):
        return {
            "__kind": "waveform_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": bool(value.is_overridden),
            "value": _jsonify_value_tree(value.value),
        }
    if isinstance(value, SweepValue):
        return {
            "start": _knob_scalar_value(value.start),
            "stop": _knob_scalar_value(value.stop),
            "expts": int(value.expts),
        }
    if isinstance(value, DirectValue):
        return _knob_scalar_value(value.value)
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    raise TypeError(
        f"Unexpected node cfg value-tree leaf {type(value).__name__}; "
        "expected CfgSectionValue, DirectValue, EvalValue, or SweepValue"
    )


def _jsonify_value_tree(value: CfgSectionValue) -> dict[str, Any]:
    return {
        key: _jsonify_value_node(child)
        for key, child in value.fields.items()
        if child is not None
    }


def _knob_scalar_value(value: object) -> object:
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _knob_eval_value(value: EvalValue) -> dict[str, object]:
    data: dict[str, object] = {"__kind": "eval", "expr": value.expr}
    if value.resolved is not None:
        data["resolved"] = value.resolved
    if value.error is not None:
        data["error"] = value.error
    return data


def ensure_test_project(ctrl: Controller) -> ProjectInfo:
    """Seed a temporary project so production start_run can create artifacts."""
    if ctrl.state.project is not None:
        return ctrl.state.project
    result_dir = Path(tempfile.mkdtemp(prefix="autofluxdep-test-result-"))
    project = ProjectInfo(
        chip_name="test_chip",
        qub_name="test_qub",
        result_dir=str(result_dir),
        database_path=str(result_dir / "Database" / "test_chip" / "test_qub"),
        params_path=str(result_dir / "params.json"),
    )
    ctrl.state.project = project
    return project


def high_snr_simparams(snr: float = 5000.0, g: float = 0.08) -> Any:
    """A copy of DEFAULT_SIMPARAM with high readout contrast for acquire tests.

    The acquire roll-out tests average mock per-shot noise with large
    reps×rounds purely to clear fit-quality gates. DEFAULT_SIMPARAM deliberately
    uses weak mock readout contrast for GUI realism, so these tests opt into a
    stronger dispersive coupling and lower Gaussian noise without weakening what
    they prove: the full real-acquire + real-fit path still runs, and the
    sim-predictor provisioning derived from these params stays consistent.
    """
    from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

    return DEFAULT_SIMPARAM.model_copy(update={"snr": snr, "g": g})


def mock_flux_predictor(sim_params: Any) -> Any:
    """A FluxoniumPredictor aligned with a MockSoc SimParams instance."""
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

    return FluxoniumPredictor(
        params=(sim_params.EJ, sim_params.EC, sim_params.EL),
        flux_half=sim_params.flux_half,
        flux_period=sim_params.flux_period,
        flux_bias=sim_params.flux_bias,
    )


def calibrated_drive_pulse(
    ml: Any,
    name: str,
    freq: float,
    *,
    sim_params: Any,
    gain: float = 0.5,
    angle: float = 1.0,
) -> dict[str, Any]:
    """Build a concrete const pulse whose gain × length matches mock calibration."""
    if gain <= 0.0:
        raise ValueError("calibrated_drive_pulse gain must be positive")
    length = float(sim_params.pi_gain_len) * float(angle) / float(gain)
    ml.register_waveform(**{name: {"style": "const", "length": length}})
    return {
        "type": "pulse",
        "waveform": ml.get_waveform(name, {"length": length}),
        "ch": 1,
        "nqz": 1,
        "gain": float(gain),
        "freq": float(freq),
    }


def _default_test_simparams() -> Any:
    """DEFAULT_SIMPARAM with poll_latency=0.0 for all test connects.

    poll_latency is mock pacing (not physics); 0.0 skips the sleep entirely so
    tests that don't care about wall-time fidelity don't pay the sleep overhead.
    The simulated IQ values, noise model, and all physics are unchanged.
    """
    from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

    return DEFAULT_SIMPARAM.model_copy(update={"poll_latency": 0.0})


def connect_mock(ctrl: Controller, *, sim_params: Any = None) -> None:
    """Establish a mock SoC synchronously-enough for a headless test.

    The session ``ConnectionService`` settles a mock connect via
    ``QTimer.singleShot``, so we drive it through the controller's public connect
    API and pump a ``QEventLoop`` until the outcome signal fires (the same pattern
    measure-gui's tests use). The autouse ``qapp`` fixture has already created the
    QApplication. On return, ``ctrl.state.exp_context.soc`` is the MockSoc and
    ``has_setup`` is true.

    FLUX-AWARE-MOCK: a mock connect also fires the shared MockFluxProvisioner,
    which registers + ramps a ``fake_flux`` FakeDevice through the controller's
    BackgroundRunner (async). We pump until that settles so a flux-aware acquire
    sees the operating value and no background op is left running at teardown
    (an unquiesced worker QThread segfaults the process). Best-effort with a
    timeout — a test that does not exercise flux still returns promptly.
    """
    from qtpy.QtCore import QCoreApplication, QEventLoop
    from zcu_tools.gui.session.services.connection import ConnectMockRequest
    from zcu_tools.gui.session.services.mock_flux import (
        FAKE_FLUX_DEVICE_NAME,
        FAKE_FLUX_INITIAL_VALUE,
    )
    from zcu_tools.gui.session.state import DeviceStatus

    ensure_test_project(ctrl)
    loop = QEventLoop()
    ctrl.bind_connection_outcome(
        on_finished=loop.quit, on_failed=lambda _msg: loop.quit()
    )
    # Use poll_latency=0.0 by default in tests to skip mock sleep overhead.
    # The SimEngine physics (IQ values, noise, fits) is unaffected by this field.
    if sim_params is None:
        sim_params = _default_test_simparams()
    ctrl.start_connect(ConnectMockRequest(sim_params=sim_params))
    loop.exec()

    # Drive the async fake_flux provisioning (connect + initial-value ramp) to
    # completion before returning.
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        app.processEvents()
        dev = ctrl.state.get_device(FAKE_FLUX_DEVICE_NAME)
        if (
            dev is not None
            and dev.status is DeviceStatus.CONNECTED
            and dev.info is not None
            and getattr(dev.info, "value", None) == FAKE_FLUX_INITIAL_VALUE
        ):
            break
        time.sleep(0.005)


def pump_controller_until_idle(ctrl: Controller, *, timeout: float = 5.0) -> None:
    """Pump Qt events until the controller's async RUN terminal path settles."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    app = QApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if not ctrl.is_running:
            ctrl._background_svc.quiesce()
            return
        time.sleep(0.001)
    raise AssertionError("autofluxdep run did not settle before timeout")


def run_controller_to_completion(
    ctrl: Controller,
    *,
    notify: Notify | None = None,
    timeout: float = 5.0,
) -> InfoStore:
    """Start an async controller RUN and return its terminal InfoStore."""
    ensure_test_project(ctrl)
    ctrl.start_run(notify=notify)
    pump_controller_until_idle(ctrl, timeout=timeout)
    info = ctrl.last_run_info
    if info is None:
        raise AssertionError("autofluxdep run finished without an InfoStore")
    return info


class _FnNode(Node):
    def __init__(self, env: RunEnv, fn: ProduceFn | None) -> None:
        self._env = env
        self._fn = fn

    def produce(self, snapshot: Snapshot) -> Patch:
        if self._fn is None:
            return Patch()
        return self._fn(self._env, snapshot)


ResultFactory = Callable[[Any, Any], Any]


def make_builder(
    name: str,
    *,
    provides: tuple[str, ...] = (),
    requires: tuple[Dependency, ...] = (),
    optional: tuple[Dependency, ...] = (),
    requires_modules: tuple[ModuleDep, ...] = (),
    optional_modules: tuple[ModuleDep, ...] = (),
    provides_modules: tuple[str, ...] = (),
    schema_fields: tuple[NodeField, ...] = (),
    override_plan: OverridePlan | None = None,
    produce_fn: ProduceFn | None = None,
    result_factory: ResultFactory | None = None,
    plotter_factory: Callable[[Any], Any] | None = None,
) -> Builder:
    """A Builder whose declarations are the given tuples and whose Node's
    ``produce`` calls ``produce_fn(env, snapshot)`` (or returns an empty Patch).

    ``schema_fields`` (the ``(key, spec, default)`` tuples) declare the fake
    Builder's typed knobs — most dependency-model tests pass none (no knobs).
    ``result_factory(schema, flux) -> Result`` (optional) gives the fake Builder a
    sweep Result so a mechanics test that asserts ``run_results`` / per-instance
    containers can use it without a real measurement Builder.
    ``plotter_factory(figure) -> Plotter`` (optional) gives it a Plotter so a UI
    test that builds a liveplot canvas works without a real measurement Builder.
    """

    class _AdHocBuilder(Builder):
        def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
            del ctx
            return NodeCfgSchema(flat_node_schema(schema_fields))

        def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
            del schema
            return override_plan or OverridePlan()

        def build_node(self, env: RunEnv) -> Node:
            return _FnNode(env, produce_fn)

        def make_init_result(
            self, schema: NodeCfgSchema, flux: Any, md: Any = None
        ) -> Any:
            del md
            if result_factory is None:
                return None
            return result_factory(schema, flux)

        def make_plotter(self, figure: Any) -> Any:
            if plotter_factory is None:
                return None
            return plotter_factory(figure)

    b = _AdHocBuilder()
    b.name = name
    b.provides = provides
    b.requires = requires
    b.optional = optional
    b.requires_modules = requires_modules
    b.optional_modules = optional_modules
    b.provides_modules = provides_modules
    return b


def place(builder: Builder, **overrides: Any) -> PlacedNode:
    """Wrap ``builder`` into a PlacedNode, seeding its schema with ``overrides``."""
    return PlacedNode(builder=builder, overrides=overrides)


def node_schema(builder: Builder, params: Any = None) -> NodeCfgSchema:
    """The placement schema a Node lowers — defaults overridden by ``params``.

    The acquire / cfg tests build a Node off a bare Builder; this mirrors what a
    ``PlacedNode`` carries (``builder.make_default_schema()`` seeded with the
    test's knob overrides) so ``make_init_result`` / ``make_acquire_env`` get a
    schema, not a raw dict.
    """
    schema = builder.make_default_schema()
    if params:
        schema.with_overrides(params)
    return schema


class _TrivialPlotter:
    """A minimal Plotter the UI can build + call ``update`` on without matplotlib.

    The UI's liveplot path needs only a ``.update(result, idx)`` method per run
    point; this records the calls so a UI mechanics test can assert redraws fired
    without a real LivePlot-backed Plotter (those are physics-shaped and need a
    configured Result)."""

    def __init__(self, figure: Any) -> None:
        del figure
        self.updates: list[int] = []

    def update(self, result: Any, idx: int) -> None:
        del result
        self.updates.append(idx)


def make_measurement_builder(name: str) -> Builder:
    """A fake MEASUREMENT Builder for UI mechanics tests: a fillable 1-D Result + a
    trivial Plotter + a produce that fills this point's row.

    Lets a UI test drive a real run worker (lock → fill → unlock, build canvases,
    auto-follow) without a real experiment's acquire — the run path under test is
    the UI's, not the physics. Provides nothing (UI tests don't assert deps)."""
    import numpy as np
    from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult

    def _result_factory(schema: Any, flux: Any) -> Any:
        del schema
        return Sweep1DResult.allocate(
            np.asarray(flux, dtype=float), np.linspace(0.0, 1.0, 4), x_label="x"
        )

    def _produce(env: RunEnv, snapshot: Snapshot) -> Patch:
        del snapshot
        env.result.signal[env.flux_idx] = np.ones(env.result.n_x)
        if env.round_hook is not None:
            env.round_hook(env.flux_idx)
        return Patch()

    return make_builder(
        name,
        produce_fn=_produce,
        result_factory=_result_factory,
        plotter_factory=_TrivialPlotter,
    )


# --- RB-2 real-acquire integration helpers -----------------------------------
#
# Every real-acquire roll-out test (lenrabi / ro_optimize / t1 / t2* / mist)
# follows the same recipe as test_qubit_freq_acquire: connect a flux-aware
# MockSoc, pick fake_flux, then run a single Node directly over a few flux values
# and assert the physically-meaningful output varies with flux. These helpers
# factor out the boilerplate so each test only declares its modules + fitter.

# A readout module near the dressed resonator (~6 GHz under DEFAULT_SIMPARAM).
ACQUIRE_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 6000.0,
        "gain": 1.0,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_freq": 6000.0, "ro_length": 0.9, "trig_offset": 0.6},
}


def make_acquire_env(ctrl: Controller, *, flux: float, flux_idx: int, **kw: Any):
    """A ``RunEnv`` carrying the connected mock soc/soccfg + the fake_flux pick.

    Mirrors what ``Orchestrator._make_env`` curries for a real run, so a Node
    built off this env runs the same real-acquire path a full run would.
    Extra keyword args (schema / ml / result / tools) flow straight through —
    ``schema`` is the placement's ``NodeCfgSchema`` (build it via ``node_schema``).
    """
    from zcu_tools.gui.session.services.mock_flux import FAKE_FLUX_DEVICE_NAME

    ctx = ctrl.state.exp_context
    return RunEnv(
        flux=flux,
        flux_idx=flux_idx,
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        **kw,
    )
