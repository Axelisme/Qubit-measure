"""Receiver PSD conversion and public rounded ADC noise, independent of truth."""

import numpy as np
from zcu_tools.core.simulation import EnvironmentSnapshot, PointTimeline, PulseEvent, ReadoutWindow
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def test_execution_grid_converts_spectral_density_to_sample_covariance() -> None:
    from zcu_tools.experiment.ftf import plan_point

    settings = parse_ftf(ftf_document() | {"receiver": {"ro_ch": 0, "gain": 0., "noise_psd_adc2_ns": [[10., 2.], [2., 20.]]}})
    window = ReadoutWindow(0, 0, 8, 100., (.1, .12), (6000., 6100.), (0., 90.), (0, 0))
    timeline = PointTimeline((), (PulseEvent(0, .1, .08, 6000., 0., .1),), (window,), .2)
    plan = plan_point(settings, EnvironmentSnapshot("ftf-v1", "test", (("bias", .2),)), timeline)
    np.testing.assert_allclose(plan.readout_model.receiver.noise_covariance, [[1., .2], [.2, 2.]], atol=1e-12)
    np.testing.assert_allclose(plan.grids[1].times_ns, 120 + 10*np.arange(8), atol=1e-12)
    assert plan.grids[1].demodulation_frequency_ghz == 6.1
    np.testing.assert_allclose(plan.grids[1].phase_rad, np.pi/2, atol=1e-12)


def test_public_decimated_receiver_noise_psd_and_offset() -> None:
    from zcu_tools.experiment.ftf import ftf

    psd = np.array([[400., 100.], [100., 900.]])
    settings = parse_ftf(ftf_document() | {"receiver": {"ro_ch": 0, "gain": 0., "offset_adc": [10., -20.], "noise_psd_adc2_ns": psd.tolist()}})
    env = ftf(settings)
    try:
        readout = PulseReadoutCfg(gen_ch=0, ro_ch=0, nqz=1, freq=6000., gain=0., waveform=ConstWaveformCfg(length=.04), ro_length=.06, trig_offset=-.01)
        program = ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=256), modules=[readout.build("ro")])
        values = np.asarray(program.acquire_decimated(env.soc, progress=False)[0]).reshape(-1, 2)
        request = env.soc.binding.last_request
        if request is None:
            raise AssertionError("completed simulation retains its numeric request")
        rate = request.timelines[0].readouts[0].sample_rate_mhz
        assert len(values) >= 4096
        np.testing.assert_allclose(values.mean(axis=0), [10., -20.], atol=.7)
        # Integer ADC rounding contributes <=1/12 to each diagonal, negligible
        # compared with this fixture's covariance; it does not alter correlation.
        np.testing.assert_allclose(np.cov(values.T), psd*rate/1000, rtol=.08, atol=.2)
    finally:
        env.devices.close_all_devices()
