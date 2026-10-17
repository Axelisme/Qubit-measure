"""Receiver PSD conversion and public rounded ADC noise, independent of truth."""

import numpy as np
from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.core._simulation import single_point_template
from tests.resources._ftf_data import ftf_document


def ftf_document_readout() -> dict[str, object]:
    """Return a detached synthetic readout record for ADC-only overrides."""
    readout = ftf_document()["readout"]
    if not isinstance(readout, dict):
        raise TypeError("synthetic readout fixture must be a mapping")
    return readout


def test_execution_grid_converts_spectral_density_to_sample_covariance() -> None:
    from zcu_tools.experiment.ftf import plan_template

    settings = parse_ftf(
        ftf_document()
        | {
            "readout": {
                **ftf_document_readout(),
                "adc": {
                    "ro_ch": 0,
                    "gain": 0.0,
                    "noise_psd_adc2_ns": [[10.0, 2.0], [2.0, 20.0]],
                },
            }
        }
    )
    window = ReadoutWindow(
        0, 0, 8, 100.0, (0.1, 0.12), (6000.0, 6100.0), (0.0, 90.0), (0, 0)
    )
    timeline = PointTimeline(
        (), (PulseEvent(0, 0.1, 0.08, 6000.0, 0.0, 0.1),), (window,), 0.2
    )
    plan = plan_template(
        settings,
        EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),)),
        single_point_template(timeline),
    )
    np.testing.assert_allclose(
        plan.readout.model.receiver.noise_covariance,
        [[1.0, 0.2], [0.2, 2.0]],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        plan.readout.grids[0][1].times_ns, 120 + 10 * np.arange(8), atol=1e-12, rtol=0
    )
    assert plan.readout.grids[0][1].demodulation_frequency_ghz == 6.1
    np.testing.assert_allclose(
        plan.readout.grids[0][1].phase_rad, np.pi / 2, atol=1e-12, rtol=0
    )


def test_public_integrated_thermal_mixture_matches_conditional_field_mean() -> None:
    from zcu_tools.experiment.ftf import ftf, plan_template
    from zcu_tools.simulate.adc import ADC

    document = ftf_document()
    model = document["chip"]
    if not isinstance(model, dict):
        raise TypeError("synthetic model fixture must be a mapping")
    model["temperature_k"] = 0.08
    settings = parse_ftf(document)
    env = ftf(settings)
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
        program = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=2048), modules=[readout.build("ro")]
        )
        averaged = np.asarray(program.acquire(env.soc, progress=False)[0]).reshape(
            -1, 2
        )[0]
        request = env.soc.binding.last_request
        if request is None:
            raise AssertionError("completed simulation retains its numeric request")
        plan = plan_template(
            settings, env.soc.binding.rounds[0].snapshot, request.templates[0]
        )
        response = ADC(plan.readout.model, rtol=1e-8, atol=1e-10).response(
            plan.readout.pulses[0], plan.readout.grids[0][0]
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


def test_public_decimated_receiver_noise_psd_and_offset() -> None:
    from zcu_tools.experiment.ftf import ftf

    psd = np.array([[400.0, 100.0], [100.0, 900.0]])
    settings = parse_ftf(
        ftf_document()
        | {
            "readout": {
                **ftf_document_readout(),
                "adc": {
                    "ro_ch": 0,
                    "gain": 0.0,
                    "offset_adc": [10.0, -20.0],
                    "noise_psd_adc2_ns": psd.tolist(),
                },
            }
        }
    )
    env = ftf(settings)
    try:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000.0,
            gain=0.0,
            waveform=ConstWaveformCfg(length=0.04),
            ro_length=0.06,
            trig_offset=-0.01,
        )
        program = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=256), modules=[readout.build("ro")]
        )
        values = np.asarray(
            program.acquire_decimated(env.soc, progress=False)[0]
        ).reshape(-1, 2)
        request = env.soc.binding.last_request
        if request is None:
            raise AssertionError("completed simulation retains its numeric request")
        rate = request.materialize_point(0).readouts[0].sample_rate_mhz
        assert len(values) >= 4096
        np.testing.assert_allclose(values.mean(axis=0), [10.0, -20.0], atol=0.7)
        # Integer ADC rounding contributes <=1/12 to each diagonal, negligible
        # compared with this fixture's covariance; it does not alter correlation.
        np.testing.assert_allclose(
            np.cov(values.T), psd * rate / 1000, rtol=0.08, atol=0.2
        )
    finally:
        env.devices.close_all_devices()
