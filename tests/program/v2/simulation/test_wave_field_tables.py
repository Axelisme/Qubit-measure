"""Wave-field lookup sweeps through public build, simulation and compiled words."""

from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import SweepCfg, TableParam, TimelineBuilder
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request
from tests.program.v2.wave_observations import executed_words


def field_program(
    field,
    value,
    *,
    mode="scalar",
    flat=False,
    paired=False,
    mixer=None,
    outer=False,
    branch=False,
    extra: dict[str, float | QickParam] | None = None,
):
    soccfg = load_soccfg(2, 1)
    soccfg["gens"][0]["has_mixer"] = mixer is not None
    env = white_noise(soccfg, std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if outer:
        builder.add_loop("outer", 2)
    if mode == "table":
        parameter = builder.add_table_sweep("point", value)
    elif mode == "linear":
        parameter = builder.add_sweep(
            "point",
            SweepCfg(
                start=value[0],
                stop=value[-1],
                expts=len(value),
                step=(value[-1] - value[0]) / (len(value) - 1),
            ),
        )
    else:
        parameter = value
    knobs: dict[str, float | QickParam] = {
        "freq": 100.0 if paired else 80.0,
        "phase": 0.0,
        "gain": 0.2,
    }
    knobs[field] = parameter
    if extra:
        knobs.update(extra)
    waveform = (
        FlatTopWaveformCfg(
            length=0.2,
            raise_waveform=GaussWaveformCfg(length=0.04, sigma=0.01),
        )
        if flat
        else ConstWaveformCfg(length=0.1)
    )
    receiver = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=waveform if paired else ConstWaveformCfg(length=0.1),
        ro_length=0.1,
        mixer_freq=mixer,
    )
    if paired:
        module = PulseReadout(
            "readout",
            receiver,
            freq=knobs["freq"],
            phase=knobs["phase"],
            gain=knobs["gain"],
        )
        pulse = module.pulse
    else:
        pulse = Pulse(
            "drive",
            PulseCfg(
                ch=1,
                nqz=1,
                freq=80,
                gain=0.2,
                waveform=waveform,
            ),
            freq=knobs["freq"],
            phase=knobs["phase"],
            gain=knobs["gain"],
        )
        module = pulse
    if branch:
        with builder.branch(by="point", align_timeline=True) as cases, cases.case(2):
            builder.emit(module)
    else:
        builder.emit(module)
    if not paired:
        builder.emit(receiver.build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    return prog, env, pulse.pulse_name(prog)


def assert_scalar_points(prog, env, values, field, **options):
    observed = request(prog, env)
    scalar_words = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = field_program(field, value, **options)
        # Coordinates identify the scan, not the scalar hardware output.
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        scalar_words.append(executed_words(scalar)[0])
    assert executed_words(prog) == scalar_words * 2


def test_const_frequency_table_matches_scalar_program_at_every_point_and_rep():
    values = [83.125, 76.75, 91.5]
    prog, env, _ = field_program("freq", values, mode="table")
    assert_scalar_points(prog, env, values, "freq")


def test_readout_phase_override_table_matches_scalar_program():
    values = [10.5, -30.25, 80.75]
    prog, env, _ = field_program("phase", values, mode="table", paired=True)
    assert_scalar_points(prog, env, values, "phase", paired=True)


@pytest.mark.parametrize(
    "field,values",
    [
        ("freq", [83.125, 76.75, 91.5]),
        ("phase", [10.5, -30.25, 80.75]),
        ("gain", [0.2, -0.1, 0.3]),
    ],
)
@pytest.mark.parametrize(
    "flat,paired,mixer",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (False, True, 50.25),
        (True, True, None),
        (True, True, 50.25),
    ],
)
def test_wave_tables_match_scalar_words_and_actual_linear_points(
    field, values, flat, paired, mixer
):
    options = dict(flat=flat, paired=paired, mixer=mixer)
    prog, env, _ = field_program(field, values, mode="table", **options)
    assert_scalar_points(prog, env, values, field, **options)
    # Gain truncation need not round-trip arbitrary rounded physical values.
    # Keep nonlinear scalar parity above; require a re-encodable linear fixture below.
    linear_values = [0.0, 0.25, 0.5] if field == "gain" else values
    linear, linear_env, name = field_program(
        field, linear_values, mode="linear", **options
    )
    actual = (
        np.asarray(linear.get_pulse_param(name, field, as_array=True))
        .flatten()
        .tolist()
    )
    scalar_words = [
        executed_words(field_program(field, value, **options)[0])[0] for value in actual
    ]
    assert executed_words(linear) == scalar_words * 2
    table, table_env, _ = field_program(field, actual, mode="table", **options)
    linear_request = request(linear, linear_env)
    table_request = request(table, table_env)
    for point in range(len(actual)):
        assert table_request.materialize_point(
            point
        ) == linear_request.materialize_point(point)
    assert executed_words(table) == executed_words(linear)


@pytest.mark.parametrize("branch", [False, True])
def test_nested_table_wrap_and_unplayed_branches_advance_every_point(branch):
    values = [83.125, 76.75, 91.5]
    prog, env, _ = field_program(
        "freq", values, mode="table", outer=True, branch=branch
    )
    observed = request(prog, env)
    expected_words = []
    for point, value in enumerate(values):
        scalar, scalar_env, _ = field_program("freq", value)
        expected = request(scalar, scalar_env).materialize_point(0)
        words = executed_words(scalar)[0]
        if branch and point != 2:
            expected = replace(expected, events=expected.events[1:])
            words = words[1:]
        for outer in range(2):
            assert (
                replace(
                    observed.materialize_point(outer * len(values) + point),
                    coordinates=(),
                )
                == expected
            )
        expected_words.append(words)
    # Real linked execution covers both sweep wrap and outer/reps restoration.
    assert executed_words(prog) == expected_words * 4


@pytest.mark.parametrize("flat", [False, True])
def test_same_wave_table_and_linear_fields_keep_both_updates_and_resets(flat):
    values = [83.125, 76.75, 91.5]
    prog, env, name = field_program(
        "freq",
        values,
        mode="table",
        flat=flat,
        extra={"gain": QickSweep1D("point", 0.0, 0.5)},
    )
    actual_gains = np.asarray(
        prog.get_pulse_param(name, "gain", as_array=True)
    ).flatten()
    observed = request(prog, env)
    expected_words = []
    for point, (freq, gain) in enumerate(zip(values, actual_gains, strict=True)):
        scalar, scalar_env, _ = field_program(
            "freq", freq, flat=flat, extra={"gain": float(gain)}
        )
        assert replace(observed.materialize_point(point), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_words.append(executed_words(scalar)[0])
    assert executed_words(prog) == expected_words * 2


def test_independent_freq_phase_gain_tables_share_one_loop():
    frequencies = [83.125, 76.75, 91.5]
    phases = [10.5, -30.25, 80.75]
    gains = [0.2, -0.1, 0.3]
    prog, env, _ = field_program(
        "freq",
        frequencies,
        mode="table",
        flat=True,
        extra={
            "phase": TableParam("point", phases),
            "gain": TableParam("point", gains),
        },
    )
    observed = request(prog, env)
    expected_words = []
    for point, (freq, phase, gain) in enumerate(
        zip(frequencies, phases, gains, strict=True)
    ):
        scalar, scalar_env, _ = field_program(
            "freq", freq, flat=True, extra={"phase": phase, "gain": gain}
        )
        assert replace(observed.materialize_point(point), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_words.append(executed_words(scalar)[0])
    assert executed_words(prog) == expected_words * 2
