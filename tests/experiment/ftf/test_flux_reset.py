"""Reset physics through synthetic program templates and the public FTF/Engine seam.

These two-state checks validate existing flux wiring, not basis convergence or
nonadiabatic accuracy for a physical device. No production behavior is added.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.core.simulation import (
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
    TimelineTemplate,
)
from zcu_tools.experiment.ftf import FTFModelFacts, plan_template
from zcu_tools.resources.ftf import FTFSettings, parse_ftf
from zcu_tools.simulate.baths import H_OVER_K
from zcu_tools.simulate.evolution import Engine

from tests.core._simulation import single_point_template
from tests.experiment._ftf_templates import snapshot
from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


def reset_settings(*, delay_ns: float = 0.0) -> FTFSettings:
    """Isolate F1's two-state reset, with quasiparticle loss away from half flux."""
    base = parse_ftf(ftf_document()).model_dump(mode="json")
    return parse_ftf(
        {
            **base,
            "chip": {
                **base["chip"],
                "f1": {
                    "ej_ghz": 3.0,
                    "ec_ghz": 0.8,
                    "el_ghz": 0.6,
                    "q_cap": 1e6,
                    "xqp": 1e-3,
                },
                "g_f1_coupler_ghz": 0.0,
                "g_f2_coupler_ghz": 0.0,
                "g_f1_f2_ghz": 0.0,
                "temperature_k": 0.05,
            },
            "drift": None,
            "dc_bias": {
                **base["dc_bias"],
                "offset_phi0": {"f1": 0.5, "coupler": 0.2, "f2": 0.4},
            },
            "flux_lines": {
                "f1": {
                    "name": "reset",
                    "gen_ch": 4,
                    "flux_phi0_per_gain": 1.0,
                    "delay_ns": delay_ns,
                }
            },
            "numerics": {
                **base["numerics"],
                "truncation": {
                    "f1": {"levels": 2, "native_cutoff": 24},
                    "coupler": {"levels": 1, "charge_cutoff": 3},
                    "f2": {"levels": 1, "native_cutoff": 8},
                    "retained_levels": 2,
                },
            },
        }
    )


def reset_template(
    hold_ns: float, *, sudden: bool = False, settling_ns: float = 0.0
) -> TimelineTemplate:
    """Ramp for 8 ns, park at f1=0.4, return to 0.5, then settle before DC readout.

    The constant-envelope control instead jumps to park at t=0 and back at the
    same terminal time, so it has at least as much relaxation time at park.
    """
    ramp_ns = 8.0
    end_ns = 2 * ramp_ns + hold_ns
    times = np.unique(
        np.r_[
            np.arange(0, ramp_ns + 1, 2),
            ramp_ns + hold_ns,
            np.arange(ramp_ns + hold_ns, end_ns + 1, 2),
        ]
    )
    ramp_fraction = np.minimum(np.minimum(times, end_ns - times) / ramp_ns, 1)
    values = (1 - np.cos(np.pi * ramp_fraction)) / 2
    envelope = (
        None
        if sudden
        else PulseEnvelope(tuple(times / 1000), tuple(values.astype(complex)))
    )
    end_us = end_ns / 1000
    flux = PulseEvent(4, 0.0, end_us, 0.0, 0.0, -0.1, envelope)
    readout_us = end_us + settling_ns / 1000
    readout = PulseEvent(0, readout_us, 0.008, 6000.0, 0.0, 0.1)
    window = ReadoutWindow(0, 0, 4, 1000.0, (readout_us,), (6000.0,), (0.0,))
    return single_point_template(
        PointTimeline((), (flux, readout), (window,), readout_us + 0.008)
    )


def evolve_reset(
    settings: FTFSettings,
    facts: FTFModelFacts,
    hold_ns: float,
    *,
    sudden: bool = False,
) -> NDArray[np.complex128]:
    """Return the terminal density in the snapshot's DC energy basis."""
    plan = plan_template(
        settings,
        snapshot(),
        reset_template(hold_ns, sudden=sudden),
        facts=facts,
    )
    engine = Engine(1)
    try:
        return engine.evolve(plan.source, (plan.batch,)).densities[0]
    finally:
        engine.close()


def park_reference(facts: FTFModelFacts) -> tuple[NDArray[np.float64], float]:
    """Independent two-state Gibbs weights and population-relaxation time in ns."""
    flux = facts.joint.dc_flux.copy()
    flux[0] = 0.4
    energies, _ = facts.joint.eigensystem(flux)
    weights = np.exp(-H_OVER_K * (energies - energies[0]) / facts.joint.temperature_k)
    weights /= weights.sum()
    rates = facts.joint.rates(flux)
    return weights, float(1 / (rates[0, 1] + rates[1, 0]))


def test_cosine_reset_long_hold_approaches_park_gibbs_above_dc_equilibrium() -> None:
    settings = reset_settings()
    facts = FTFModelFacts(settings, snapshot())
    weights, park_t1_ns = park_reference(facts)
    dc_rates = facts.joint.rates(facts.joint.dc_flux)
    dc_t1_ns = 1 / (dc_rates[0, 1] + dc_rates[1, 0])
    assert park_t1_ns < dc_t1_ns / 100
    dc_ground = facts.joint.initial_state().density[0, 0].real
    # Round every hold to the public 2-ns flux grid, not an internal solver step.
    holds = 2 * np.rint(np.array([0.0, 1.0, 12.0]) * park_t1_ns / 2)
    densities = [evolve_reset(settings, facts, float(hold)) for hold in holds]
    ground = np.array([rho[0, 0].real for rho in densities])
    # For two populations, p0(t)=pG+(p0(0)-pG)*exp(-t/T1).
    expected = weights[0] + (dc_ground - weights[0]) * np.exp(-holds / park_t1_ns)
    # Absolute population tolerance includes ramp/bath discretization, not just
    # the 1e-3 default engine budget. This is not a large-excursion accuracy claim.
    np.testing.assert_allclose(ground, expected, atol=5e-3, rtol=0)
    assert np.all(np.diff(ground) > 0.05)
    assert ground[-1] > dc_ground + 0.2
    assert ground[-1] == pytest.approx(weights[0], abs=5e-3)
    for rho in densities:
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10, rtol=0)
        assert np.linalg.eigvalsh(rho).min() >= -1e-10


@pytest.mark.parametrize("delay_ns", [0.0, 1.0, 0.5, 2.0])
def test_delayed_cosine_reset_retains_constant_park_relaxation(delay_ns: float) -> None:
    settings = reset_settings(delay_ns=delay_ns)
    facts = FTFModelFacts(settings, snapshot())
    weights, park_t1_ns = park_reference(facts)
    hold_ns = float(2 * np.rint(12 * park_t1_ns / 2))
    plan = plan_template(
        settings,
        snapshot(),
        reset_template(hold_ns, settling_ns=8.0),
        facts=facts,
    )
    # The delayed return must finish before the DC readout observation.
    waveform = plan.batch.flux_waveform
    assert waveform is not None
    np.testing.assert_array_equal(waveform.flux[-1], facts.joint.dc_flux)
    assert waveform.time_ns[-1] < plan.batch.end_ns[0]
    engine = Engine(1)
    try:
        result = engine.evolve(plan.source, (plan.batch,))
    finally:
        engine.close()
    # A long constant park thermalizes even when its ramp ends between samples.
    assert result.densities[0, 0, 0].real == pytest.approx(weights[0], abs=5e-3)
    assert result.paths.segments_by_kernel["t4"] > 0


def test_sudden_reset_returns_rotated_park_gibbs_not_adiabatic_populations() -> None:
    settings = reset_settings()
    facts = FTFModelFacts(settings, snapshot())
    weights, park_t1_ns = park_reference(facts)
    hold_ns = float(2 * np.rint(12 * park_t1_ns / 2))
    ramped = evolve_reset(settings, facts, hold_ns)
    sudden = evolve_reset(settings, facts, hold_ns, sudden=True)
    flux = facts.joint.dc_flux.copy()
    flux[0] = 0.4
    _, vectors = facts.joint.eigensystem(flux)
    # Long sudden holds thermalize in the park basis. The instantaneous return
    # projects that density onto DC, rather than preserving energy-index labels.
    sudden_prediction = (vectors * weights) @ vectors.conj().T
    np.testing.assert_allclose(sudden, sudden_prediction, atol=2e-3, rtol=0)
    assert ramped[0, 0].real == pytest.approx(weights[0], abs=5e-3)
    assert ramped[0, 0].real > sudden[0, 0].real + 0.2
