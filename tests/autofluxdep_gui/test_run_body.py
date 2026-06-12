"""Run-body mechanics — the controller drives the orchestrator over providers.

These exercise the run *wiring* (predictor Service prepended, RunEnv threaded,
Patch merged into the InfoStore, the ModuleSource bridge), NOT experiment
physics: the per-Node real-acquire fit is covered against the flux-aware MockSoc
by the ``test_*_acquire.py`` integration tests. A fake measurement Node (a
``make_builder`` double whose ``produce`` returns a deterministic Patch) keeps the
mechanics fast and decoupled from any acquire.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency

from ._helpers import make_builder


def _consume_predict(env, snapshot):
    """A fake measurement Node: echo the Service's predict_freq into a Patch.

    Reads ``predict_freq`` (provided by the prepended predictor Service) and the
    flux from the env, and emits a deterministic ``measured`` key so the run wiring
    (snapshot projection → produce → Patch merge) is observable without physics."""
    predicted = float(snapshot["predict_freq"])
    patch = Patch()
    patch.set("measured", predicted + 0.5)  # a fixed, deterministic offset
    return patch


def _fake_consumer():
    return make_builder(
        "consumer",
        provides=("measured",),
        requires=(Dependency("predict_freq"),),
        produce_fn=_consume_predict,
    )


def test_controller_run_drives_predictor_service_then_consumer():
    # the controller prepends the predictor Service and runs the user node after it;
    # the consumer reads the Service's predict_freq and the final InfoStore carries
    # both keys — the full predictor-Service-then-Node run wiring.
    ctrl = build_core()
    ctrl.add_node(_fake_consumer())
    ctrl.set_flux_values([0.0, 1.0])
    info = ctrl.start_run()
    # the predictor Service produced predict_freq at the last point, and the
    # consumer produced its derived key off it
    assert "predict_freq" in info.point
    assert info.point["measured"] == info.point["predict_freq"] + 0.5


def test_run_threads_flux_into_env():
    # the run threads each flux point's value into the Node's RunEnv: the consumer
    # records env.flux per point, so the recorded sequence matches the sweep.
    seen: list[float] = []

    def record_flux(env, snapshot):
        del snapshot
        seen.append(env.flux)
        return Patch()

    ctrl = build_core()
    ctrl.add_node(
        make_builder(
            "recorder", requires=(Dependency("predict_freq"),), produce_fn=record_flux
        )
    )
    ctrl.set_flux_values([0.0, 0.5, 1.0])
    ctrl.start_run()
    assert seen == [0.0, 0.5, 1.0]


def test_produce_exception_fails_run_gracefully():
    # a Node whose produce raises (e.g. an unconfigured real acquire Fast-Failing)
    # must NOT propagate out of the run: the orchestrator catches it, the run ends
    # on RUN_FAILED (not RUN_FINISHED), the controller unlocks, and the error is
    # carried on the payload. This is what stops a GUI run worker QThread aborting.
    from zcu_tools.gui.app.autofluxdep.events.run import (
        RunFailedPayload,
        RunFinishedPayload,
    )

    def boom(env, snapshot):
        del env, snapshot
        raise RuntimeError("node not configured")

    ctrl = build_core()
    ctrl.add_node(
        make_builder("broken", requires=(Dependency("predict_freq"),), produce_fn=boom)
    )
    ctrl.set_flux_values([0.0, 1.0])

    events: list[str] = []
    ctrl.bus.subscribe(RunFailedPayload, lambda p: events.append(f"failed:{p.message}"))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append("finished"))

    ctrl.start_run()  # must not raise

    assert len(events) == 1 and events[0].startswith("failed:")
    assert "node not configured" in events[0]
    assert not ctrl.is_running  # the controller unlocked


def test_ml_module_source_returns_none_on_absent():
    # the orchestrator's ModuleSource contract is "None if absent", but
    # ModuleLibrary.get_module raises — the adapter start_run threads in must
    # bridge that so an absent module dep falls back instead of crashing the run.
    from zcu_tools.gui.app.autofluxdep.controller import _MlModuleSource
    from zcu_tools.meta_tool import ModuleLibrary

    source = _MlModuleSource(ModuleLibrary())
    assert source.get_module("not_a_module") is None  # must not raise
