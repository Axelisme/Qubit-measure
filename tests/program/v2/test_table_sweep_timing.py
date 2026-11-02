"""Table sweeps preserve scalar and native-linear linked output and reference time."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.program.v2 import SweepCfg, TimelineBuilder, TimelineProgramV2, ir
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program._profiles import load_soccfg

Field = Literal["freq", "phase", "gain", "length", "delay", "readout_freq"]
Mode = Literal["scalar", "linear", "table"]
RAMP_US = 0.04
POINTS = 3


def compile_scan(
    field: Field,
    values: list[float],
    *,
    mode: Mode,
    flat: bool,
    branch: bool,
    reps_innermost: bool,
) -> tuple[TimelineProgramV2, str]:
    """Compile the same nested scan/branch topology for every kind of parameter."""
    soccfg = load_soccfg()
    # Two live branch arms plus nested counters and runtime length consumers
    # need more registers than the minimal compile fixture reserves.
    soccfg["tprocs"][0]["dreg_qty"] *= 2
    builder = TimelineBuilder(
        soccfg,
        reps=2,
        initial_delay=0,
        relax_delay=0.1,
        reps_innermost=reps_innermost,
    )
    builder.add_loop("outer", 2)
    parameter: float | QickParam
    if mode == "linear":
        parameter = builder.add_sweep(
            "point",
            SweepCfg(
                start=values[0],
                stop=values[-1],
                expts=POINTS,
                step=(values[-1] - values[0]) / (POINTS - 1),
            ),
        )
    elif mode == "table":
        parameter = builder.add_table_sweep("point", values)
    else:
        builder.add_loop("point", POINTS)
        parameter = values[0]

    waveform = (
        FlatTopWaveformCfg(
            length=0.2,
            raise_waveform=GaussWaveformCfg(length=RAMP_US, sigma=0.01),
        )
        if flat
        else ConstWaveformCfg(length=0.1)
    )
    cfg = PulseCfg(ch=1, nqz=1, freq=80, gain=0.2, waveform=waveform)
    if field not in ("delay", "readout_freq"):
        cfg.set_param(field, parameter)
    drive = Pulse("drive", cfg)
    receiver = PulseReadout(
        "readout",
        PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=waveform,
            ro_length=0.1,
        ),
        freq=parameter if field == "readout_freq" else None,
    )
    if field == "delay":
        builder.delay(parameter, tag="hold")
    if branch:
        with builder.branch(by="point", align_timeline=True) as cases:
            with cases.case(0):
                builder.emit(drive)
            # Point 1 deliberately does not play the swept pulse. The next
            # emission must still use point 2, including after outer/reps wrap.
            with cases.case(2):
                builder.emit(drive)
    else:
        builder.emit(drive)
    builder.emit(
        PulseCfg(
            ch=1, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        ).build("follower")
    )
    builder.emit(receiver)
    builder.delay_by_timeline()
    prog = builder.build()
    pulse = receiver.pulse if field == "readout_freq" else drive
    return prog, pulse.pulse_name(prog)


def analyze(prog: TimelineProgramV2) -> ir.TimingAnalysis:
    """Observe optimized linked assembly using owner-produced layout and memories."""
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list,
            prog.labels,
            prog.timing_layout,
            dmem=prog.compile_datamem(),
            wmem=ir.compiled_wmem(prog),
        )
    )


def relative_shot(shot: ir.ShotTiming) -> ir.ShotTiming:
    """Discard only linked addresses and the accumulated preceding-shot reference."""
    return replace(
        shot,
        start_cycle=0,
        body_end_cycle=shot.body_end_cycle - shot.start_cycle,
        end_cycle=shot.end_cycle - shot.start_cycle,
        events=tuple(
            replace(event, address=0, cycle=event.cycle - shot.start_cycle)
            for event in shot.events
        ),
    )


def assert_scalar_oracle(
    prog: TimelineProgramV2,
    values: list[float],
    field: Field,
    *,
    flat: bool,
    branch: bool,
    reps_innermost: bool,
) -> ir.TimingAnalysis:
    """Check every shot against a scalar build with identical control flow."""
    observed = analyze(prog)
    scalars = [
        analyze(
            compile_scan(
                field,
                [value],
                mode="scalar",
                flat=flat,
                branch=branch,
                reps_innermost=reps_innermost,
            )[0]
        )
        for value in values
    ]
    point_axis = [loop.name for loop in prog.timing_layout.loops].index("point")
    expected_reference = scalars[0].shots[0].start_cycle
    for shot in observed.shots:
        scalar = scalars[shot.loop_indices[point_axis]].shots[shot.index]
        assert relative_shot(shot) == relative_shot(scalar)
        assert shot.start_cycle == expected_reference
        expected_reference += scalar.end_cycle - scalar.start_cycle
        assert shot.end_cycle == expected_reference
        assert all(e.wave_words is not None for e in shot.events if e.kind == "wave")
    assert len(observed.shots) == len(scalars[0].shots)
    assert observed.end_cycle == expected_reference
    return observed


@pytest.mark.parametrize(
    "field", ["freq", "phase", "gain", "length", "delay", "readout_freq"]
)
@pytest.mark.parametrize("flat", [False, True], ids=["const", "flat_top"])
@pytest.mark.parametrize("branch", [False, True], ids=["played", "unplayed_branch"])
@pytest.mark.parametrize(
    "reps_innermost", [False, True], ids=["reps_outer", "reps_inner"]
)
def test_linear_actual_points_preserve_linked_outputs_and_reference(
    field: Field, flat: bool, branch: bool, reps_innermost: bool
) -> None:
    # Gain's exact round-trip interval is intentional; arbitrary gains have a
    # scalar-word contract, not a linear round-trip contract (tested below).
    bounds = {
        "freq": [76.75, 91.5],
        "phase": [-30.25, 80.75],
        "gain": [0.0, 0.5],
        "length": [0.125, 0.375],
        "delay": [0.015625, 0.046875],
        "readout_freq": [83.125, 101.5],
    }
    linear, name = compile_scan(
        field,
        bounds[field],
        mode="linear",
        flat=flat,
        branch=branch,
        reps_innermost=reps_innermost,
    )
    if field == "delay":
        parameter = QickSweep1D("point", bounds[field][0], bounds[field][-1])
        linear.us2cycles(parameter)
        actual = (
            np.asarray(
                parameter.get_rounded({"point": POINTS}).to_array({"point": POINTS})
            )
            .flatten()
            .tolist()
        )
    else:
        parameter_name = "freq" if field == "readout_freq" else field
        actual = (
            np.asarray(linear.get_pulse_param(name, parameter_name, as_array=True))
            .flatten()
            .tolist()
        )
        if flat and field == "length":
            # Public QICK length is the flat segment; cfg length includes the
            # authored ramp. Restore the cfg input, not the quantized ramp word.
            actual = [value + RAMP_US for value in actual]

    # Establish exact scalar re-encoding BEFORE using linear as a table oracle.
    expected = assert_scalar_oracle(
        linear,
        actual,
        field,
        flat=flat,
        branch=branch,
        reps_innermost=reps_innermost,
    )
    table, _ = compile_scan(
        field,
        actual,
        mode="table",
        flat=flat,
        branch=branch,
        reps_innermost=reps_innermost,
    )
    observed = analyze(table)
    assert tuple(map(relative_shot, observed.shots)) == tuple(
        map(relative_shot, expected.shots)
    )
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in observed.shots] == [
        (s.start_cycle, s.body_end_cycle, s.end_cycle) for s in expected.shots
    ]
    assert observed.end_cycle == expected.end_cycle
    assert tuple(replace(e, address=0) for e in observed.initial_events) == tuple(
        replace(e, address=0) for e in expected.initial_events
    )


@pytest.mark.parametrize(
    "field,values",
    [
        ("freq", [83.125, 76.75, 91.5]),
        ("phase", [10.5, -30.25, 80.75]),
        ("gain", [0.2, -0.1, 0.3]),
        ("length", [0.213, 0.107, 0.354]),
        ("delay", [0.0213, 0.0107, 0.0354]),
        ("readout_freq", [83.125, 101.5, 76.75]),
    ],
)
@pytest.mark.parametrize("flat", [False, True], ids=["const", "flat_top"])
@pytest.mark.parametrize("branch", [False, True], ids=["played", "unplayed_branch"])
@pytest.mark.parametrize(
    "reps_innermost", [False, True], ids=["reps_outer", "reps_inner"]
)
def test_nonaffine_table_points_preserve_scalar_linked_outputs_and_reference(
    field: Field, values: list[float], flat: bool, branch: bool, reps_innermost: bool
) -> None:
    table, _ = compile_scan(
        field,
        values,
        mode="table",
        flat=flat,
        branch=branch,
        reps_innermost=reps_innermost,
    )
    assert_scalar_oracle(
        table,
        values,
        field,
        flat=flat,
        branch=branch,
        reps_innermost=reps_innermost,
    )
