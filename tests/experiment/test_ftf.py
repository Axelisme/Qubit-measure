"""Public FTF composition and acquisition, without GUI truth or lab recipes."""

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def test_point_composition_units_signed_drive_and_tail_freeze() -> None:
    from zcu_tools.experiment.ftf import plan_point

    settings = parse_ftf(
        ftf_document()
        | {
            "flux_sources": [
                {
                    "name": "flux",
                    "gen_ch": 1,
                    "coefficients_phi0_per_gain": [0.0, 0.1, 0.0],
                    "tau_ns": 100.0,
                }
            ]
        }
    )
    snapshot = EnvironmentSnapshot("ftf-v1", "test", (("bias", 0.2),))
    events = (
        PulseEvent(1, 0.0, 0.5, 0.0, 0.0, 0.4),
        PulseEvent(2, 0.5, 0.5, 20.0, 90.0, -0.2),
        PulseEvent(0, 2.0, 0.5, 6000.0, 90.0, 0.1),
    )
    window = ReadoutWindow(0, 0, 10, 100.0, (1.9,), (6000.0,), (45.0,), (0,))
    plan = plan_point(settings, snapshot, PointTimeline((), events, (window,), 2.5))
    assert plan.readout_start_ns == 2000.0
    assert plan.pulses[0].frequency_ghz == 6.0
    assert plan.pulses[0].phase_rad == pytest.approx(np.pi / 2, abs=1e-12, rel=0)
    assert plan.controls.pulses[1].start_ns == 500.0
    assert plan.controls.pulses[1].end_ns == 1000.0
    assert plan.controls.pulses[1].frequency_ghz == 0.02
    np.testing.assert_allclose(
        plan.grid.times_ns, 1900 + 10 * np.arange(10), atol=1e-12, rtol=0
    )
    expected_flux = 0.22 + 0.04 * (1 - np.exp(-5)) * np.exp(-15)
    assert plan.readout_flux[1] == pytest.approx(expected_flux, abs=1e-12, rel=0)
    assert plan.readout_flux[1] != 0.22
    source = next(s for s in plan.evolution_model.sources if s.name == "mw")
    expected = sum(
        (c * plan.joint.charge_operator(i) for i, c in enumerate([0.02, 0.0, -0.01])),
        np.zeros((8, 8), dtype=complex),
    )
    np.testing.assert_allclose(source.in_phase_ghz, expected, atol=1e-12, rtol=0)
    expected_q = sum(
        (-c * plan.joint.charge_operator(i) for i, c in enumerate([0.01, 0.0, 0.02])),
        np.zeros((8, 8), dtype=complex),
    )
    if source.quadrature_ghz is None:
        raise AssertionError("microwave source must retain its quadrature operator")
    np.testing.assert_allclose(source.quadrature_ghz, expected_q, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        plan.readout_model.frequencies_ghz,
        plan.joint.readout_frequencies_ghz(plan.readout_flux, plan.resonators),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("phase_deg", "expected_second"),
    [(0.0, -0.2), (180.0, 0.2)],
    ids=["same-phase", "opposite-phase"],
)
def test_common_carrier_origin_survives_pulse_start_quantization(
    phase_deg: float, expected_second: float
) -> None:
    from zcu_tools.experiment.ftf import plan_point

    settings = parse_ftf(ftf_document())
    snapshot = EnvironmentSnapshot("ftf-v1", "test", (("bias", 0.2),))
    window = ReadoutWindow(0, 0, 4, 100.0, (0.1,), (6000.0,), (0.0,), (0,))
    for second_start_us in (0.032, 0.033):
        events = (
            PulseEvent(2, 0.0, 0.016, 250.0, 0.0, 0.2),
            PulseEvent(2, second_start_us, 0.016, 250.0, phase_deg, -0.2),
            PulseEvent(0, 0.1, 0.04, 6000.0, 0.0, 0.1),
        )
        plan = plan_point(
            settings, snapshot, PointTimeline((), events, (window,), 0.14)
        )
        first, second = plan.controls.pulses
        # Both observations are whole carrier cycles from their common origin.
        assert plan.controls.amplitude(first, 8.0) == pytest.approx(
            0.2, abs=1e-12, rel=0
        )
        assert plan.controls.amplitude(second, 40.0) == pytest.approx(
            expected_second, abs=1e-12, rel=0
        )


def test_delayed_short_flux_pulse_survives_long_idle_with_large_max_step() -> None:
    from dataclasses import replace

    from zcu_tools.experiment.ftf import plan_point
    from zcu_tools.simulate.evolution import Evolution, SolverSettings

    settings = parse_ftf(ftf_document())
    # Low-frequency synthetic circuits permit adaptive steps longer than the pulse.
    scale = 1e-4
    model = settings.model
    settings = settings.model_copy(
        update={
            "model": model.model_copy(
                update={
                    "f1": model.f1.model_copy(
                        update={
                            "ej_ghz": 3.0 * scale,
                            "ec_ghz": 0.8 * scale,
                            "el_ghz": 0.6 * scale,
                        }
                    ),
                    "coupler": model.coupler.model_copy(
                        update={
                            "ejmax_ghz": 10.0 * scale,
                            "ec_ghz": 0.3 * scale,
                            "q_cap": None,
                        }
                    ),
                    "f2": model.f2.model_copy(
                        update={
                            "ej_ghz": 3.3 * scale,
                            "ec_ghz": 0.7 * scale,
                            "el_ghz": 0.55 * scale,
                        }
                    ),
                    "g1c_ghz": 0.03 * scale,
                    "g2c_ghz": 0.04 * scale,
                    "g12_ghz": -0.02 * scale,
                    "retained_levels": 2,
                }
            ),
            "flux_sources": (
                settings.flux_sources[0].model_copy(
                    update={
                        "coefficients_phi0_per_gain": (0.0, 0.0, 0.1),
                        "delay_ns": 400.0,
                        "tau_ns": 0.0,
                    }
                ),
            ),
            "solver": settings.solver.model_copy(
                update={
                    "max_step_ns": 10000.0,
                    "bath_segment_ns": 2000.0,
                }
            ),
        }
    )
    window = ReadoutWindow(0, 0, 4, 100.0, (1.0,), (6000.0,), (0.0,), (0,))
    events = (
        PulseEvent(1, 0.0, 0.002, 0.0, 0.0, 1.0),
        PulseEvent(0, 1.0, 0.04, 6000.0, 0.0, 0.1),
    )
    plan = plan_point(
        settings,
        EnvironmentSnapshot("ftf-v1", "test", (("bias", 0.2),)),
        PointTimeline((), events, (window,), 1.04),
    )
    initial = plan.joint.initial_state()
    solver_settings = SolverSettings(rtol=1e-8, atol=1e-10, max_step_ns=10000.0)
    # Independent event geometry: the actual flux pulse is [400,402) ns.
    reference_model = replace(plan.evolution_model, event_times_ns=(400.0, 402.0))
    expected = Evolution(reference_model, solver_settings).evolve(
        initial, plan.controls, end_ns=1000.0
    )
    assert np.max(np.abs(expected.density - initial.density)) > 1e-5
    actual = Evolution(plan.evolution_model, solver_settings).evolve(
        initial, plan.controls, end_ns=1000.0
    )
    np.testing.assert_allclose(actual.density, expected.density, atol=2e-7, rtol=0)


def _readout() -> PulseReadoutCfg:
    return PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.04),
        ro_length=0.06,
        trig_offset=-0.01,
    )


def test_ftf_public_acquire_tau_tail_round_dc_and_same_trace_raw() -> None:
    from zcu_tools.experiment.ftf import ftf

    settings = parse_ftf(
        ftf_document()
        | {
            "flux_sources": [
                {
                    "name": "flux",
                    "gen_ch": 1,
                    "coefficients_phi0_per_gain": [0.0, 0.1, 0.0],
                    "tau_ns": 8.0,
                }
            ]
        }
    )
    env = ftf(settings)
    try:
        pulse = PulseCfg(
            ch=1,
            nqz=1,
            freq=0.0,
            gain=0.3,
            waveform=ConstWaveformCfg(length=0.016),
            post_delay=0.016,
        )
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=2, rounds=2),
            modules=[pulse.build("flux"), _readout().build("ro")],
        )
        driver = env.devices.get_device("bias")
        from zcu_tools.device.fake import FakeDevice

        if not isinstance(driver, FakeDevice):
            raise TypeError("FTF bias must be a memory source")

        def change_bias(*_args, **_kwargs) -> None:
            driver.set_value(0.4)

        result = program.acquire_decimated(
            env.soc, progress=False, round_hook=change_bias
        )
        rounds = env.soc.binding.rounds
        assert rounds[0].snapshot.values == (("bias", 0.2),)
        assert rounds[1].snapshot.values == (("bias", 0.4),)
        assert driver.get_value() == 0.4
        assert np.isfinite(result[0]).all()
        assert np.any(np.asarray(result[0]) != 0)
        # Companion raw uses the last round trace, not a fresh realization.
        program2 = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=2, rounds=1),
            modules=[pulse.build("flux2"), _readout().build("ro2")],
        )
        trace = np.asarray(program2.acquire_decimated(env.soc, progress=False)[0])
        raw_channels = program2.get_raw()
        if raw_channels is None:
            raise AssertionError("decimated acquisition must retain companion raw")
        raw = np.asarray(raw_channels[0])
        np.testing.assert_allclose(
            raw.reshape(-1, 2),
            trace.reshape(-1, trace.shape[-2], 2).sum(axis=1),
            rtol=0,
            atol=(trace.shape[-2] + 1) / 2,
        )
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("kind", ["gate", "bias"])
def test_readout_rejects_active_controls_not_passive_tail(kind: str) -> None:
    from zcu_tools.experiment.ftf import plan_point

    settings = parse_ftf(ftf_document())
    readout = PulseEvent(0, 0.02, 0.04, 6000.0, 0.0, 0.1)
    control = PulseEvent(2 if kind == "gate" else 1, 0.03, 0.02, 0.0, 0.0, 0.1)
    window = ReadoutWindow(0, 0, 10, 100.0, (0.02,), (6000.0,), (0.0,), (0,))
    with pytest.raises(ValueError, match="readout.*control|control.*readout"):
        plan_point(
            settings,
            EnvironmentSnapshot("ftf-v1", "test", (("bias", 0.2),)),
            PointTimeline((), (readout, control), (window,), 0.07),
        )
