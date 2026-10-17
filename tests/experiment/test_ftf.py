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

from tests.core._simulation import single_point_template
from tests.resources._ftf_data import ftf_document


def test_template_composition_units_signed_drive_and_dc_readout() -> None:
    from zcu_tools.experiment.ftf import plan_template
    from zcu_tools.experiment.ftf.numeric import joint_parameters, resonators
    from zcu_tools.simulate.ftf import JointModel

    settings = parse_ftf(ftf_document())
    snapshot = EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),))
    events = (
        PulseEvent(2, 0.5, 0.5, 20.0, 90.0, -0.2),
        PulseEvent(0, 2.0, 0.5, 6000.0, 90.0, 0.1),
    )
    window = ReadoutWindow(0, 0, 10, 100.0, (1.9,), (6000.0,), (45.0,), (0,))
    template = single_point_template(PointTimeline((), events, (window,), 2.5))
    plan = plan_template(settings, snapshot, template)
    np.testing.assert_array_equal(plan.batch.end_ns, [2000.0])
    pulse = plan.readout.pulses[0][0]
    assert pulse.frequency_ghz == 6.0
    assert pulse.phase_rad == pytest.approx(np.pi / 2, abs=1e-12, rel=0)
    track = plan.batch.tracks[0]
    np.testing.assert_array_equal(track.start_ns, [500.0])
    np.testing.assert_array_equal(track.duration_ns, [500.0])
    np.testing.assert_array_equal(track.frequency_ghz, [0.02])
    np.testing.assert_array_equal(track.amplitude, [-0.2])
    np.testing.assert_allclose(
        plan.readout.grids[0][0].times_ns, 1900 + 10 * np.arange(10), atol=1e-12, rtol=0
    )
    # Native bias maps to DC coupler flux 0.2 + 0.1 * 0.2, independent of pulses.
    dc = np.array([0.3, 0.2 + 0.1 * 0.2, 0.4])
    joint = JointModel(joint_parameters(settings), dc, temperature_k=0.0)
    model = plan.source.model((0, 0, 0))
    source = next(s for s in model.drives if s.name == "mw")
    _, vectors = joint.eigensystem(dc)
    expected = sum(
        (c * joint.charge_operator(i) for i, c in enumerate([0.02, 0.0, -0.01])),
        np.zeros((8, 8), dtype=complex),
    )
    expected_q = sum(
        (-c * joint.charge_operator(i) for i, c in enumerate([0.01, 0.0, 0.02])),
        np.zeros((8, 8), dtype=complex),
    )
    np.testing.assert_allclose(
        source.in_phase_ghz, vectors.conj().T @ expected @ vectors, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        source.quadrature_ghz,
        vectors.conj().T @ expected_q @ vectors,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        plan.readout.model.frequencies_ghz,
        joint.readout_frequencies_ghz(dc, resonators(settings)),
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
    from zcu_tools.experiment.ftf import plan_template

    settings = parse_ftf(ftf_document())
    snapshot = EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),))
    window = ReadoutWindow(0, 0, 4, 100.0, (0.1,), (6000.0,), (0.0,), (0,))
    for second_start_us in (0.032, 0.033):
        events = (
            PulseEvent(2, 0.0, 0.016, 250.0, 0.0, 0.2),
            PulseEvent(2, second_start_us, 0.016, 250.0, phase_deg, -0.2),
            PulseEvent(0, 0.1, 0.04, 6000.0, 0.0, 0.1),
        )
        plan = plan_template(
            settings,
            snapshot,
            single_point_template(PointTimeline((), events, (window,), 0.14)),
        )
        first, second = plan.batch.tracks

        # Both observations are whole carrier cycles from their common origin.
        def amplitude(track, time):
            return track.amplitude[0] * np.exp(
                1j * (2 * np.pi * track.frequency_ghz[0] * time + track.phase_rad[0])
            )

        assert first.start_ns[0] == 0.0
        assert second.start_ns[0] == 32.0
        assert amplitude(first, 8.0) == pytest.approx(0.2, abs=1e-12, rel=0)
        assert amplitude(second, 40.0) == pytest.approx(
            expected_second, abs=1e-12, rel=0
        )


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


def test_ftf_public_charge_acquire_round_dc_and_same_trace_raw() -> None:
    from zcu_tools.experiment.ftf import ftf

    settings = parse_ftf(ftf_document())
    env = ftf(settings)
    try:
        pulse = PulseCfg(
            ch=2,
            nqz=1,
            freq=250.0,
            gain=0.3,
            waveform=ConstWaveformCfg(length=0.016),
            post_delay=0.016,
        )
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=2, rounds=2),
            modules=[pulse.build("charge"), _readout().build("ro")],
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
            modules=[pulse.build("charge2"), _readout().build("ro2")],
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
