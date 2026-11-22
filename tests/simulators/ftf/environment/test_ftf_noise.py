"""Receiver PSD conversion and ADC sampling, independent of physical truth."""

import numpy as np
import pytest
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.boundary import (
    EnvironmentSnapshot,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
    SamplingIdentity,
)
from zcu_tools.simulators.ftf.config import parse_ftf

from tests.core._simulation import TIME_BASE, single_point_template, tick_tuple, ticks
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.ftf.config._data import ftf_document

pytestmark = pytest.mark.sim


def ftf_document_readout() -> dict[str, object]:
    """Return a detached synthetic readout record for ADC-only overrides."""
    readout = ftf_document()["readout"]
    if not isinstance(readout, dict):
        raise TypeError("synthetic readout fixture must be a mapping")
    return readout


def test_execution_grid_converts_spectral_density_to_sample_covariance() -> None:
    from zcu_tools.simulators.ftf.environment import plan_template

    settings = parse_ftf(
        ftf_document()
        | {
            "readout": {
                **ftf_document_readout(),
                "adc": {
                    "background": {"ripple_period_mhz": 1000.0, "pivot_ghz": 0.0},
                    "ro_ch": 0,
                    "gain": 0.0,
                    "noise_psd_adc2_ns": 10.0,
                },
            }
        }
    )
    window = ReadoutWindow(
        0,
        0,
        8,
        TIME_BASE.period_ticks(100.0),
        tick_tuple((0.1, 0.12)),
        (6000.0, 6100.0),
        (0.0, 90.0),
        (0, 0),
    )
    timeline = PointTimeline(
        (),
        (PulseEvent(0, ticks(0.1), ticks(0.08), 6000.0, 0.0, 0.1),),
        (window,),
        ticks(0.2),
        time_base=TIME_BASE,
    )
    plan = plan_template(
        settings,
        EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),)),
        single_point_template(timeline),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    np.testing.assert_allclose(
        plan.readouts[0].batch.model.receiver.noise_covariance,
        np.eye(2),
        atol=1e-12,
        rtol=0,
    )
    readout_event = plan.readouts[0].batch.events[0][0]
    np.testing.assert_allclose(
        readout_event.grids[1].times_ns,
        120 + 10 * np.arange(8),
        atol=1e-12,
        rtol=0,
    )
    assert readout_event.grids[1].demodulation_frequency_ghz == 6.1
    np.testing.assert_allclose(
        readout_event.grids[1].phase_rad, np.pi / 2, atol=1e-12, rtol=0
    )


def test_public_integrated_thermal_mixture_matches_conditional_field_mean() -> None:
    from zcu_tools.simulate.adc import ADC
    from zcu_tools.simulators.ftf.environment import ftf, plan_template

    document = ftf_document()
    model = document["chip"]
    if not isinstance(model, dict):
        raise TypeError("synthetic model fixture must be a mapping")
    model["temperature_k"] = 0.08
    settings = parse_ftf(document)
    env = ftf(settings, soccfg=profile_soccfg("ftf"))
    try:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000.0,
            gain=0.4,
            waveform=ConstWaveformCfg(length=0.08),
            ro_length=0.1,
        )
        program = compile_modules(
            env.soccfg,
            modules=[readout.build("ro")],
            reps=2048,
        )
        acquired = program.acquire(env.soc, progress=False)
        assert acquired is not None
        averaged = np.asarray(acquired[0]).reshape(-1, 2)[0]
        request = env.soc.binding.last_request
        if request is None:
            raise AssertionError("completed simulation retains its numeric request")
        plan = plan_template(
            settings,
            env.soc.binding.rounds[0].snapshot,
            request.templates[0],
            soccfg=profile_soccfg("ftf"),
            time_base=request.time_base,
        )
        readout_event = plan.readouts[0].batch.events[0][0]
        response = ADC(plan.readouts[0].batch.model, rtol=1e-8, atol=1e-10).response(
            readout_event.pulses, readout_event.grids[0]
        )
        energy = plan.source.model((0, 0, 0)).energies_ghz
        weights = np.exp(-0.04799243073366221 * (energy - energy[0]) / 0.08)
        weights /= weights.sum()
        conditional = response.traces.mean(axis=1)
        expected = weights @ conditional
        variance = weights @ (conditional - expected) ** 2
        assert variance.max() > 1.0
        np.testing.assert_allclose(
            averaged,
            expected,
            rtol=0,
            atol=6 * np.sqrt(variance.max() / 2048) + 1 / request.readouts[0].samples,
        )
    finally:
        env.devices.close_all_devices()


def test_scalar_receiver_psd_draws_isotropic_adc_samples() -> None:
    from zcu_tools.simulate.adc import ADC, PopulationInput
    from zcu_tools.simulators.ftf.environment import plan_template

    settings = parse_ftf(
        ftf_document()
        | {
            "readout": {
                **ftf_document_readout(),
                "adc": {
                    "background": {"ripple_period_mhz": 1000.0, "pivot_ghz": 0.0},
                    "ro_ch": 0,
                    "gain": 0.0,
                    "offset_adc": [10.0, -20.0],
                    "noise_psd_adc2_ns": 400.0,
                },
            }
        }
    )
    # At 100 MHz each sample spans 10 ns, so 400 ADC²·ns gives variance 40.
    window = ReadoutWindow(
        0, 0, 8, TIME_BASE.period_ticks(100.0), (0,), (6000.0,), (0.0,), (0,)
    )
    timeline = PointTimeline((), (), (window,), ticks(0.1), time_base=TIME_BASE)
    plan = plan_template(
        settings,
        EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),)),
        single_point_template(timeline),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    (readout,) = plan.readouts
    model = readout.batch.model
    (event,) = readout.batch.events[0]
    (grid,) = event.grids
    populations = np.zeros(len(model.state_labels))
    populations[0] = 1.0
    samples = ADC(model).sample(
        PopulationInput(model.state_labels, populations),
        event.pulses,
        grid,
        SamplingIdentity(123),
        shots=2048,
    )
    iq = samples.traces.reshape(-1, 2)
    np.testing.assert_allclose(iq.mean(axis=0), [10.0, -20.0], atol=0.2)
    np.testing.assert_allclose(iq.var(axis=0), [40.0, 40.0], rtol=0, atol=1.5)
    np.testing.assert_allclose(np.corrcoef(iq.T), np.eye(2), rtol=0, atol=0.03)
