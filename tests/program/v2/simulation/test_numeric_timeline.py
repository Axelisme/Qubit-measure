"""Numeric timeline contracts independent of a physical solver."""

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.core.simulation import ChannelSignals, RoundSignals
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.control import Branch
from zcu_tools.program.v2.modules.delay import DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord
from zcu_tools.program.v2.modules.pulse import Pulse
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg, ConstWaveformCfg

from tests.program.v2.simulation.support import raw_of, request_of


def readout(name="readout", *, channel=0, samples=31, waveform=None):
    return PulseReadoutCfg(
        gen_ch=channel,
        ro_ch=channel,
        nqz=1,
        freq=100,
        gain=0.4,
        waveform=waveform or ConstWaveformCfg(length=0.1),
        ro_length=samples / 307.2,
    ).build(name)


@pytest.mark.parametrize("loader", [LoadValue, LoadWord])
def test_empty_table_is_a_semantic_noop_if_no_consumer_needs_its_register(loader):
    env = white_noise(std_adc=0)
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[
            loader("empty", [], "reps", "unused"),
            readout(),
        ],
    )
    try:
        prog.acquire(env.soc, progress=False)
    except ValueError as exc:
        pytest.fail(f"empty loader must preserve its hardware no-op semantics: {exc}")
    assert len(request_of(env).materialize_point(0).events) == 1


def test_arb_preserves_signed_complex_scale_sample_axis_and_detached_ownership(
    tmp_path,
):
    path = tmp_path / "signed.npz"
    np.savez(
        path, time=[0, 0.05, 0.1], idata=[-0.2, 0.1, -0.1], qdata=[0.1, -0.05, 0.05]
    )
    env = white_noise(std_adc=0)
    module = readout(waveform=ArbWaveformCfg(data="signed", path=str(path)))
    prog = ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=2), modules=[module])
    prog.acquire(env.soc, progress=False)
    event = request_of(env).materialize_point(0).events[0]
    envelope = event.envelope
    assert envelope is not None
    values = np.asarray(envelope.values, dtype=np.complex128)
    assert envelope.sample_us[0] == 0
    assert envelope.sample_us[1] == pytest.approx(1 / 6144)
    assert event.duration_us == pytest.approx(38 / 384)
    assert values.real.min() == pytest.approx(-0.2, abs=1 / 32766)
    assert np.max(np.abs(values)) < 0.23  # never normalize this to a unit peak
    assert values[0].imag == pytest.approx(0.1, abs=1 / 32766)
    assert np.interp(0.025, envelope.sample_us, values.real) == pytest.approx(
        -0.05, abs=0.003
    )
    assert event.gain == pytest.approx(0.4, abs=1 / 32766)
    # Changing the asset after preparation does not mutate the detached request.
    np.savez(path, time=[0, 0.05, 0.1], idata=[0, 0, 0], qdata=[0, 0, 0])
    assert envelope.values[0].real < 0


def test_adc_read_windows_retain_each_generator_association():
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    second = readout("second")
    second.cfg.gen_ch = 1
    # Build again so the owned Pulse and flat readout share the selected route.
    second = second.cfg.build("second")
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(), modules=[readout("first"), second]
    )
    prog.acquire(env.soc, progress=False)
    window = request_of(env).materialize_point(0).readouts[0]
    assert getattr(window, "generator_channels", ()) == (0, 1)


def test_two_adc_sample_counts_order_round_averaging_and_companion_raw():
    env = white_noise(std_adc=0, n_gens=2, n_readouts=2, routes=((1, 1), (0, 0)))
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=2, rounds=2),
        modules=[
            readout("one", channel=1, samples=7),
            readout("zero", channel=0, samples=3),
        ],
    )
    seen = []

    def execute(request):
        seen.append(request)
        return RoundSignals(
            tuple(
                ChannelSignals(
                    window.channel,
                    np.full(
                        (
                            np.prod(request.acquire.loop_dims),
                            window.reads,
                            window.samples,
                            2,
                        ),
                        10 * window.channel + request.index + 0.25,
                    ),
                )
                for window in reversed(request.acquire.readouts)
            )
        )

    env.soc.binding.execute = execute
    traces = prog.acquire_decimated(env.soc, progress=False)
    assert [trace.shape for trace in traces] == [(2, 7, 2), (2, 3, 2)]
    np.testing.assert_array_equal(traces[0], 10.75)
    np.testing.assert_array_equal(traces[1], 0.75)
    raw = raw_of(prog)
    np.testing.assert_array_equal(raw[0], 79)  # round(7 * 11.25), last round only
    np.testing.assert_array_equal(raw[1], 4)  # round(3 * 1.25)
    assert len(seen) == 2


def test_join_swept_branch_emission_order_is_fixed_even_when_times_cross():
    env = white_noise(std_adc=0, n_gens=2, n_readouts=2, routes=((0, 0), (1, 1)))
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[
            Join(
                [SoftDelay("swept", QickParam(0.1, {"axis": 0.3})), readout("a")],
                [SoftDelay("fixed", 0.2), readout("b", channel=1)],
            )
        ],
        sweep=[("axis", 2)],
    )
    prog.acquire(env.soc, progress=False)
    request = request_of(env)
    assert [
        [e.source for e in events]
        for events in (
            request.materialize_point(p).events
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == [
        ["a", "b"],
        ["a", "b"],
    ]
    assert (
        request.materialize_point(1).readouts[0].trigger_us[0]
        > request.materialize_point(1).readouts[1].trigger_us[0]
    )


def test_undefined_scalar_and_out_of_bounds_table_fail_before_signals():
    env = white_noise(std_adc=0)
    uninitialized = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[
            Branch(
                "choice",
                Pulse("none", None),
                LoadValue("not_taken", [0], "reps", "cycles"),
                compare_by="reps",
            ),
            DelayAuto("undefined", "cycles"),
            readout(),
        ],
    )
    with pytest.raises(
        ValueError, match="DelayAuto.*undefined.*reps.*undefined register.*cycles"
    ):
        uninitialized.acquire(env.soc, progress=False)
    out_of_bounds = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[
            LoadValue("short_table", [0], "axis", "value"),
            readout(),
        ],
        sweep=[("axis", 2)],
    )
    with pytest.raises(
        ValueError, match="LoadValue.*short_table.*axis.*out of bounds.*axis"
    ):
        out_of_bounds.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds
