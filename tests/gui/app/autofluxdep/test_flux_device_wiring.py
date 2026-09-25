"""RB-0b: the user's flux-source pick (state.flux_device_name) reaches each
RunEnv so a real-acquire Node can write this point's flux into the picked device.

These tests drive the real Controller + Orchestrator run path with an ad-hoc
capturing Node, asserting ``env.flux_device`` carries the picked device name (and
None when no source is picked). No real acquire here — only the wiring.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch

from ._helpers import connect_mock, make_builder, run_controller_to_completion


def _capture_builder(seen: list[str | None]):
    def produce_fn(env, _snapshot):
        seen.append(env.flux_device)
        return Patch()

    return make_builder("capture", produce_fn=produce_fn)


def test_flux_device_name_reaches_run_env():
    ctrl = build_core()
    seen: list[str | None] = []
    ctrl.add_node(_capture_builder(seen))
    connect_mock(ctrl)
    # The picker stores a connected device NAME; fake_flux is auto-provisioned by
    # the shared MockFluxProvisioner on the mock connect above.
    ctrl.set_flux_device("fake_flux")
    ctrl.set_flux_values([0.0, 0.2, 0.4])
    run_controller_to_completion(ctrl)

    assert seen == ["fake_flux", "fake_flux", "fake_flux"]


def test_flux_device_none_when_unset():
    ctrl = build_core()
    seen: list[str | None] = []
    ctrl.add_node(_capture_builder(seen))
    connect_mock(ctrl)
    # No flux source picked → env.flux_device is None (bare-number flux sweep).
    ctrl.set_flux_values([0.0, 0.5])
    run_controller_to_completion(ctrl)

    assert seen == [None, None]
