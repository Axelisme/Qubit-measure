"""RB-2: lenrabi end-to-end real acquire against the flux-aware MockSoc.

Runs the lenrabi Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, rabi_pulse, Readout).acquire -> fit_rabi) at a few flux
points and asserts a finite, positive pi length comes back -- i.e. the real
acquire produced a fittable Rabi oscillation (not noise / not a constant).

The drive is on resonance: the snapshot's ``qubit_freq`` is the predicted f01 at
each flux (FluxoniumPredictor matching DEFAULT_SIMPARAM), so the Rabi pulse drives
the qubit and the length sweep traces out an oscillation.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.program.v2 import ModuleCfgFactory, PulseCfg
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from ._helpers import (
    ACQUIRE_READOUT,
    connect_mock,
    make_acquire_env,
    node_schema,
)

_PARAMS = {
    "qub_waveform": "rabi_drive",
    "qub_ch": 1,
    "qub_nqz": 1,
    "qub_gain": 0.5,
    "qub_length": 1.0,
    # start above zero: a const waveform length-sweep needs >= a few FPGA cycles
    "sweep_range": SweepValue(start=0.05, stop=2.0, expts=41),
    "reps": 100,
    "rounds": 1,
    "relax_delay": 0.0,
}


def _ml(ctrl):
    ml = ctrl.state.exp_context.ml
    ml.register_waveform(rabi_drive={"style": "const", "length": 1.0})
    return ml


def test_lenrabi_acquire_fits_finite_pi_length():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = _ml(ctrl)
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = LenRabiBuilder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.06, 0.1]
    pis: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        f01 = predictor.predict_freq(flux)
        snap = Snapshot(
            {"qubit_freq": float(f01)}, modules={"opt_readout": ACQUIRE_READOUT}
        )
        patch = builder.build_node(env).produce(snap)
        if "pi_length" in patch.values():
            pi_length = float(patch.values()["pi_length"])
            pi2_length = float(patch.values()["pi2_length"])
            modules = patch.modules()
            pi_pulse = ModuleCfgFactory.from_raw(modules["pi_pulse"], ml=ml)
            pi2_pulse = ModuleCfgFactory.from_raw(modules["pi2_pulse"], ml=ml)
            assert isinstance(pi_pulse, PulseCfg)
            assert isinstance(pi2_pulse, PulseCfg)
            assert float(pi_pulse.waveform.length) == pi_length
            assert float(pi2_pulse.waveform.length) == pi2_length
            assert float(pi_pulse.freq) == float(f01)
            assert float(pi2_pulse.freq) == float(f01)
            pis.append(pi_length)

    # at least one flux point produced a finite, positive pi length from the real
    # acquire (a constant / noise signal would fail the fit-quality gate and omit it)
    assert pis, "no flux point fit a pi length from the real acquire"
    assert all(np.isfinite(p) and p > 0.0 for p in pis), (
        f"non-physical pi lengths: {pis}"
    )
