"""Semantic loop/table scheduling through real program acquisition.

Remove the scalar diagnostic reset parameter when the Module-tree interpreter
is deleted. Keep the acquisition case on the public acquisition route.
"""

from fractions import Fraction
from math import ceil

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.delay import DelayAuto
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import GEN_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timeline import compile_modules
from tests.program._timing_expectations import (
    generator_duration,
    native_sweep,
    readout_duration,
    runtime_cycles,
    tproc_ceil,
)
from tests.program.v2.simulation.support import request_of
from tests.program.v2.simulation.time_observations import time_us, times_us


def drive(length=0.1, gain=0.2):
    return PulseCfg(
        ch=1, nqz=1, freq=80, gain=gain, waveform=ConstWaveformCfg(length=length)
    )


def receiver():
    return PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )


def acquire(modules, *, sweep=None, reps=1):
    env = white_noise(
        load_soccfg(n_gens=2), std_adc=0, routes=((0, 0), (1, 0)), seed=None
    )
    prog = compile_modules(
        env.soccfg,
        modules=modules,
        sweep=sweep,
        reps=reps,
    )
    try:
        prog.acquire(env.soc, progress=False)
    except ValueError as exc:
        pytest.fail(f"supported semantic control must acquire: {exc}")
    return request_of(env), prog, env


def test_repeat_count_and_delay_cycles_are_local_to_each_acquire():
    env = white_noise(
        load_soccfg(n_gens=2), std_adc=0, routes=((0, 0), (1, 0)), seed=None
    )
    builder = TimelineBuilder(env.soccfg)
    builder.add_loop("shot", 2)
    builder.emit(LoadValue("wait", [43, 86], idx_reg="shot", val_reg="cycles"))
    for count in [0, 1, 2]:
        with builder.repeat(f"repeat_{count}", count):
            builder.emit(drive().build("drive"))
    builder.delay_by_timeline()
    builder.emit(DelayAuto("wait_loaded", t="cycles"))
    builder.emit(receiver().build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    prog.acquire(env.soc, progress=False)
    request = request_of(env)
    assert request.point_dims == (2,)
    drive_end = tproc_ceil(env.soccfg, generator_duration(env.soccfg, 0.1, 1))
    for point, wait in enumerate((43, 86)):
        readout_start = (3 * drive_end + wait) / TPROC_MHZ
        timeline = request.materialize_point(point)
        assert [e.source for e in timeline.events] == [
            "drive",
            "drive",
            "drive",
            "readout",
        ]
        assert time_us(
            request.time_base, timeline.events[-1].start_ticks
        ) == pytest.approx(readout_start)
    prog.acquire(env.soc, progress=False)
    for point in range(2):
        assert request_of(env).materialize_point(point) == request.materialize_point(
            point
        )


@pytest.mark.parametrize(
    ("kind", "count"), [("single", 1), ("parallel", 2), ("bath", 3)]
)
def test_reset_request_events_keep_authored_wrapper_name(kind, count):
    cavity = drive()
    qubit = drive().model_copy(update={"ch": 0})
    if kind == "single":
        cfg = PulseResetCfg(pulse_cfg=cavity)
    elif kind == "parallel":
        cfg = TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit)
    else:
        cfg = BathResetCfg(
            cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=drive()
        )
    request, _, _ = acquire([cfg.build("authored_reset"), receiver().build("readout")])
    assert [event.source for event in request.materialize_point(0).events] == [
        *(["authored_reset"] * count),
        "readout",
    ]


def test_reset_wrappers_expand_only_actual_parallel_pulses():
    reset = TwoPulseResetCfg(
        pulse1_cfg=drive(0.1),
        pulse2_cfg=PulseCfg(
            ch=0, nqz=1, freq=60, gain=0.2, waveform=ConstWaveformCfg(length=0.2)
        ),
    ).build("reset")
    request, _, _ = acquire([None, reset, receiver().build("readout")])
    assert len(request.materialize_point(0).events) == 3
    assert [
        time_us(request.time_base, e.start_ticks)
        for e in request.materialize_point(0).events[:2]
    ] == [0, 0]
    assert time_us(
        request.time_base, request.materialize_point(0).events[-1].start_ticks
    ) == pytest.approx(
        tproc_ceil(load_soccfg(), generator_duration(load_soccfg(), 0.2)) / TPROC_MHZ
    )


def test_two_pulse_reset_swept_lengths_use_machine_blocking_end():
    env = white_noise(
        load_soccfg(n_gens=3), std_adc=0, routes=((0, 0), (1, 0), (2, 0)), seed=None
    )
    pulses = [
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=80,
            gain=0.2,
            pre_delay=0.01,
            post_delay=0.02,
            waveform=ConstWaveformCfg(length=QickParam(0.05, {"length": 0.1})),
        )
        for ch in (1, 2)
    ]
    reset = TwoPulseResetCfg(pulse1_cfg=pulses[0], pulse2_cfg=pulses[1])
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.2),
        pre_delay=0.01,
        post_delay=0.02,
        trig_offset=0.03,
        ro_length=0.1,
    )
    try:
        prog = compile_modules(
            env.soccfg,
            modules=[reset.build("reset"), readout.build("readout")],
            sweep=[("length", 3)],
        )
        prog.acquire(env.soc, progress=False)
        request = request_of(env)
        timelines = [request.materialize_point(point) for point in range(3)]
        # Each native point covers the actual waveform plus pre/post once.
        lengths = native_sweep(env.soccfg, 0.05, 0.15, 3, gen_ch=1)
        starts = [
            2 * tproc_ceil(env.soccfg, 0.01)
            + ceil(
                runtime_cycles(env.soccfg, n, 1)
                + Fraction("0.02") * Fraction(str(TPROC_MHZ))
            )
            for n in lengths
        ]
        triggers = [
            start - tproc_ceil(env.soccfg, 0.01) + int(env.soccfg.us2cycles(0.03))
            for start in starts
        ]
        readout_end = tproc_ceil(
            env.soccfg, generator_duration(env.soccfg, 0.2) + Fraction("0.02")
        )
        adc_end = tproc_ceil(env.soccfg, readout_duration(env.soccfg, 0.1))
        for point, timeline in enumerate(timelines):
            assert [event.source for event in timeline.events] == [
                "reset",
                "reset",
                "readout",
            ]
            assert [
                time_us(request.time_base, event.duration_ticks)
                for event in timeline.events[:2]
            ] == pytest.approx([lengths[point] / GEN_MHZ] * 2)
            assert time_us(
                request.time_base, timeline.events[-1].start_ticks
            ) == pytest.approx(starts[point] / TPROC_MHZ)
            assert times_us(
                request.time_base, timeline.readouts[0].trigger_ticks
            ) == pytest.approx((triggers[point] / TPROC_MHZ,))
            assert time_us(request.time_base, timeline.duration_ticks) == pytest.approx(
                max(starts[point] + readout_end, triggers[point] + adc_end) / TPROC_MHZ
            )
    finally:
        env.devices.close_all_devices()


def test_reps_counter_lookup_rejected_before_backend():
    env = white_noise(
        load_soccfg(n_gens=2), std_adc=0, routes=((0, 0), (1, 0)), seed=None
    )
    builder = TimelineBuilder(env.soccfg)
    builder.emit(LoadValue("writer", [0], "reps", "selection"))
    builder.emit(
        ComputedPulse("reader", val_reg="selection", pulses=[drive(), drive(0.2)])
    )
    builder.emit(receiver().build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    with pytest.raises(
        ValueError, match=r"MetaInst.*preopt.*(?:reps shell|cross-shot.*s12)"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_reps_counter_time_lookup_rejected_before_backend():
    modules = [
        LoadValue("time", [1], "reps", "cycles"),
        DelayAuto("invalid_wait", t="cycles"),
        receiver().build("readout"),
    ]
    env = white_noise(load_soccfg(), std_adc=0, seed=None, routes=((0, 0),))
    prog = compile_modules(env.soccfg, modules=modules)
    with pytest.raises(
        ValueError, match=r"MetaInst.*preopt.*(?:reps shell|cross-shot.*s12)"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds
