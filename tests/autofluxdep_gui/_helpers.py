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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    NodeCfgSchema,
    OverridePlan,
    flat_node_schema,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeField
from zcu_tools.gui.app.autofluxdep.nodes.builder import (
    Builder,
    Node,
    PlacedNode,
    RunEnv,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore, Notify

ProduceFn = Callable[[RunEnv, Snapshot], Patch]


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


def high_snr_simparams(snr: float = 5000.0) -> Any:
    """A copy of DEFAULT_SIMPARAM with a high snr for fast acquire tests.

    The acquire roll-out tests average mock per-shot noise with large
    reps×rounds purely to clear the fit-quality gate at DEFAULT_SIMPARAM's
    snr=300. Raising the snr lets the same decay/fringe clear the gate at a
    fraction of the reps, cutting wall time without weakening what the test
    proves (it still drives the full real-acquire + real-fit path and asserts a
    finite, positive coherence time). snr only scales per-shot noise, so the
    sim-predictor provisioning derived from these params stays consistent.
    """
    from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

    return DEFAULT_SIMPARAM.model_copy(update={"snr": snr})


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
    "ro_cfg": {"ro_ch": 0, "ro_length": 0.9, "trig_offset": 0.6},
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
