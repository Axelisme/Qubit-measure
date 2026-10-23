"""Persisted mux settings drive the real program-to-FTF acquisition path."""

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.resources.ftf.schema import MuxReadoutSource

from tests.experiment._ftf_templates import settings
from tests.resources._ftf_data import mux_document

pytestmark = pytest.mark.sim


def jumping_settings(mux):
    """Two-level hot bath with identical noiseless receivers and visible ADC sums."""
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip
    chip = chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": chip.f1.model_copy(update={"q_cap": 10.0}),
            "f2": chip.f2.model_copy(update={"q_cap": 10.0}),
        }
    )
    receiver = numeric.readout.receivers[0].model_copy(update={"gain": 1e12})
    readout = numeric.readout.model_copy(
        update={
            "adc": receiver,
            "mux": (
                MuxReadoutSource(
                    gen_ch=1,
                    input_per_gain=1.0,
                    adc=tuple(
                        receiver.model_copy(update={"ro_ch": ch}) for ch in (4, 1)
                    ),
                )
                if mux
                else None
            ),
        }
    )
    return numeric.model_copy(update={"chip": chip, "readout": readout})


def jumping_program(env, mux, *, rounds=1):
    """Real QICK single/mux readout, 64 shots, identical carrier and ADC windows."""
    if mux:
        readout = MuxReadoutCfg(
            gen_ch=1,
            nqz=1,
            mixer_freq=6000.0,
            length=0.020,
            ro_length=0.026,
            tones=[MuxReadoutTone(ro_ch=ch, freq=6000.0, gain=0.1) for ch in (4, 1)],
        )
    else:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.020),
            ro_length=0.026,
        )
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=64, rounds=rounds),
        modules=[readout.build("readout")],
    )


def raw_shots(prog):
    """Require completed acquisition's public int64 shot buffers."""
    raw = prog.get_raw()
    if raw is None:
        raise AssertionError("completed FTF acquisition must retain raw shots")
    return raw


@pytest.mark.parametrize("mux", [False, True])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_public_acquisition_resolves_jump_bridges_and_shared_mux_paths(mux, mode):
    env = ftf(jumping_settings(mux), time_scale=0)
    repeat = ftf(jumping_settings(mux), time_scale=0)
    try:
        prog = jumping_program(env, mux)
        replay = jumping_program(repeat, mux)
        if mode == "integrated":
            result = prog.acquire(env.soc, progress=False)
            repeated = replay.acquire(repeat.soc, progress=False)
        else:
            result = prog.acquire_decimated(env.soc, progress=False)
            repeated = replay.acquire_decimated(repeat.soc, progress=False)
        raw = raw_shots(prog)
        for first, second in zip(raw, raw_shots(replay), strict=True):
            assert first.dtype == np.int64
            assert first.shape == (64, 1, 2)
            # Two static state centres cannot produce this bridge of values.
            assert np.unique(first[:, 0], axis=0).shape[0] > 10
            np.testing.assert_array_equal(first, second)
        for first, second in zip(result, repeated, strict=True):
            np.testing.assert_array_equal(first, second)
        if mux:
            assert tuple(prog.ro_chs) == (4, 1)
            np.testing.assert_array_equal(raw[0], raw[1])
            if mode == "decimated":
                np.testing.assert_array_equal(result[0], result[1])
        if mode == "decimated":
            for trace, sums in zip(result, raw, strict=True):
                np.testing.assert_array_equal(
                    sums[:, 0], np.rint(trace.sum(axis=-2)).astype(np.int64)
                )
    finally:
        env.devices.close_all_devices()
        repeat.devices.close_all_devices()


@pytest.mark.parametrize("mux", [False, True])
def test_deferred_getter_keeps_its_round_and_does_not_advance_mean_stream(mux):
    env = ftf(jumping_settings(mux), time_scale=0)
    repeat = ftf(jumping_settings(mux), time_scale=0)
    try:
        held = jumping_program(env, mux, rounds=2)
        reference = jumping_program(repeat, mux, rounds=2)
        first_mean = held.acquire(env.soc, progress=False)
        expected_mean = reference.acquire(repeat.soc, progress=False)
        expected_raw = [shots.copy() for shots in raw_shots(reference)]
        # Only the reference materializes raw before the next acquire.
        other = jumping_program(env, mux, rounds=2)
        other_reference = jumping_program(repeat, mux, rounds=2)
        next_mean = other.acquire(env.soc, progress=False)
        expected_next = other_reference.acquire(repeat.soc, progress=False)
        captured_rounds = env.soc.binding.rounds
        for actual, expected in zip(raw_shots(held), expected_raw, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert env.soc.binding.rounds == captured_rounds
        for actual, expected in zip(first_mean, expected_mean, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(next_mean, expected_next, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
            raw_shots(other), raw_shots(other_reference), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        assert any(
            not np.array_equal(first, second)
            for first, second in zip(expected_raw, raw_shots(other), strict=True)
        )
    finally:
        env.devices.close_all_devices()
        repeat.devices.close_all_devices()


@pytest.mark.parametrize("mux_gen, channels", [(1, (4, 1)), (7, (9, 8))])
def test_one_environment_acquires_mux_then_pulse_without_reconfiguration(
    mux_gen, channels
):
    numeric = settings().with_telemetry(enabled=False)
    configured = parse_ftf(mux_document(channels))
    assert configured.readout.mux is not None
    numeric = numeric.model_copy(
        update={
            "readout": configured.readout.model_copy(
                update={
                    "mux": configured.readout.mux.model_copy(update={"gen_ch": mux_gen})
                }
            )
        }
    )
    env = ftf(numeric, time_scale=0)
    try:
        mux = MuxReadoutCfg(
            gen_ch=mux_gen,
            nqz=1,
            mixer_freq=6000,
            length=0.020,
            ro_length=0.026,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=6000 + 80 * i, gain=0.1)
                for i, ch in enumerate(channels)
            ],
        )
        pulse = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.020),
            ro_length=0.026,
        )
        for module, expected_channels in ((mux, channels), (pulse, (0,))):
            prog = ModularProgramV2(
                env.soccfg, ProgramV2Cfg(reps=2), modules=[module.build("readout")]
            )
            result = prog.acquire(env.soc, progress=False)
            assert tuple(prog.ro_chs) == expected_channels
            assert len(result) == len(expected_channels)
            assert all(np.isfinite(iq).all() for iq in result)
            shots = raw_shots(prog)
            assert all(iq.shape == (2, 1, 2) for iq in shots)
            assert any(np.any(iq != 0) for iq in shots)
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("channels", [(4, 1), (1,)])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_yaml_mux_program_produces_each_tones_raw_shots(channels, mode):
    try:
        settings = parse_ftf(mux_document(channels))
    except ValueError as error:
        pytest.fail(f"mux settings must enable end-to-end FTF acquisition: {error}")
    env = ftf(settings)
    try:
        module = MuxReadoutCfg(
            gen_ch=1,
            nqz=1,
            mixer_freq=6000.0,
            length=0.020,
            ro_length=0.026,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=6000.0 + 80 * i, gain=0.1, phase=45 * i)
                for i, ch in enumerate(channels)
            ],
        )
        prog = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=7), modules=[module.build("joint")]
        )
        if mode == "integrated":
            prog.acquire(env.soc, progress=False)
        else:
            prog.acquire_decimated(env.soc, progress=False)
        request = env.soc.binding.last_request
        assert request is not None
        assert tuple(w.channel for w in request.readouts) == channels
        raw = prog.get_raw()
        assert raw is not None
        assert len(raw) == len(channels)
        for shots in raw:
            assert shots.dtype == np.int64
            assert shots.size == 14
            assert np.any(shots != 0)
        again = prog.get_raw()
        assert again is not None
        for first, second in zip(raw, again, strict=True):
            np.testing.assert_array_equal(first, second)
    finally:
        env.devices.close_all_devices()
