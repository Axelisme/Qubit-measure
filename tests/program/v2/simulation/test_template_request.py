"""Point templates match the scalar diagnostic interpreter.

The scalar parity oracle and legacy expansion checks retire when execution
controls/timelines are removed from AcquireRequest. Synthetic programs only.
"""

import sys
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord
from zcu_tools.program.v2.modules.pulse import PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.readout import (
    PulseReadout,
    PulseReadoutCfg,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.simulation.frontend import build_request
from zcu_tools.program.v2.simulation.timeline import TimelineInterpreter
from zcu_tools.program.v2.utils import readout_freq_words


def drive(*, ch=1, waveform=None):
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=80,
        gain=-0.3,
        waveform=waveform or ConstWaveformCfg(length=0.1),
    )


def played_drive(events, axis):
    """Evaluate the diagnostic linear-envelope/common-carrier playback contract."""
    values = np.zeros_like(axis, dtype=np.complex128)
    for event in events:
        local = axis - event.start_us
        active = (local >= 0) & (local < event.duration_us)
        envelope = np.ones_like(axis, dtype=np.complex128)
        if event.envelope is not None:
            envelope = np.interp(local, event.envelope.sample_us, event.envelope.values)
        values += np.where(
            active,
            event.gain
            * envelope
            * np.exp(
                1j
                * np.deg2rad(
                    event.phase_deg
                    + 360 * event.frequency_mhz * (axis - event.phase_reference_us)
                )
            ),
            0,
        )
    return values


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


def example(case, env):
    """Return modules and sweep axes for one supported semantic mechanism."""
    sweep = [("axis", 3)]
    tail = [receiver().build("readout")]
    if case == "sweep":
        cfg = drive()
        cfg.freq = QickParam(80, {"axis": 13.1, "outer": 2.3})
        cfg.gain = QickParam(-0.3, {"axis": 0.1})
        cfg.phase = QickParam(3, {"axis": 37})
        cfg.pre_delay = QickParam(0.013, {"axis": 0.031})
        return [
            SoftDelay("soft", QickParam(0.017, {"axis": 0.051})),
            cfg.build("drive"),
            Delay("delay", 0.011),
            *tail,
        ], [("outer", 2), *sweep]
    if case == "variable_repeat":
        return [
            LoadValue("count", [0, 2, 1], "axis", "n"),
            Repeat("inner", "n", range_hint=(0, 2)).add_content(drive().build("drive")),
            *tail,
        ], sweep
    if case == "fixed_repeat":
        return [Repeat("inner", 2).add_content(drive().build("drive")), *tail], sweep
    if case == "branch_route":
        return [
            Branch(
                "choice", drive().build("a"), drive(ch=0).build("b"), compare_by="axis"
            ),
            *tail,
        ], [("axis", 2)]
    if case == "computed":
        candidates = [
            drive(waveform=GaussWaveformCfg(length=0.1, sigma=0.02)),
            drive(),
        ]
        return [
            LoadValue("select", [1, 0, 1], "axis", "selected"),
            ComputedPulse("gate", val_reg="selected", pulses=candidates),
            *tail,
        ], sweep
    if case == "words":
        gen, adc = readout_freq_words(
            env.soccfg, [100, 200, 300], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
        )
        return [
            LoadWord("gen", gen, "axis", "gen_word"),
            LoadWord("adc", adc, "axis", "adc_word"),
            LoadWord("gain", [3276, (-6552) % 2**32, 0], "axis", "gain_word"),
            LoadValue("wait", [0, 43, 86], "axis", "cycles"),
            DelayAuto("loaded", "cycles"),
            PulseReadout(
                "readout",
                receiver(),
                freq_val="gen_word",
                ro_freq_val="adc_word",
                gain_val="gain_word",
            ),
        ], sweep
    if case == "branch":
        return [
            LoadValue("select", [1, 0, 1], "axis", "selected"),
            Branch(
                "choice",
                drive().build("a"),
                [drive().build("b"), drive().build("c")],
                compare_by="selected",
            ),
            *tail,
        ], sweep
    if case == "length":
        return [
            TableLengthPulse(
                "length", drive(), lengths=[0.1, 0.2, 0.3], idx_reg="axis"
            ),
            *tail,
        ], sweep
    if case == "table_hook":
        gen, adc = readout_freq_words(
            env.soccfg, [100, 200, 300], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
        )
        return [
            TablePulseReadout(
                "readout", receiver(), idx_reg="axis", freq_words=gen, ro_freq_words=adc
            )
        ], [("outer", 2), *sweep]
    if case == "join":
        return [
            Join(
                [SoftDelay("swept", QickParam(0.1, {"axis": 0.3})), drive().build("a")],
                [SoftDelay("fixed", 0.2), receiver().build("b")],
            )
        ], sweep
    if case in ("flat_top", "flat_top_sweep"):
        return [
            drive(
                waveform=FlatTopWaveformCfg(
                    length=QickParam(0.1078125, {"axis": 0.015625})
                    if case == "flat_top_sweep"
                    else 0.3,
                    raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                )
            ).build("flat"),
            *tail,
        ], sweep
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "sweep",
        "variable_repeat",
        "fixed_repeat",
        "computed",
        "branch_route",
        "words",
        "branch",
        "length",
        "table_hook",
        "join",
        "flat_top",
        "flat_top_sweep",
    ],
)
def test_templates_materialize_like_scalar_interpretation(case):
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    modules, sweep = example(case, env)
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=2), modules=modules, sweep=sweep
    )
    request = build_request(prog, "integrated", env.soc.binding, 0)
    assert request.templates, "frontend must produce structural point templates"
    assert request.point_names == tuple(name for name, _ in sweep)
    assert request.point_dims == tuple(n for _, n in sweep)
    assert request.reps == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([t.points for t in request.templates])),
        np.arange(np.prod(request.point_dims)),
    )
    oracle = TimelineInterpreter(prog)
    for i, indices in enumerate(product(*(range(n) for n in prog.loop_dict.values()))):
        point = dict(zip(prog.loop_dict, indices, strict=True))
        scalar = oracle.run_point(point)
        oracle.after_point()
        flat = np.ravel_multi_index(
            tuple(point[name] for name in request.point_names), request.point_dims
        )
        expected = replace(
            scalar,
            coordinates=tuple((name, point[name]) for name in request.point_names),
        )
        materialized = request.materialize_point(int(flat))
        if case.startswith("flat_top"):
            assert materialized.coordinates == expected.coordinates
            assert materialized.readouts == expected.readouts
            assert materialized.duration_us == expected.duration_us
            axis = np.arange(0, expected.duration_us, 1 / 12288)
            np.testing.assert_allclose(
                played_drive(materialized.events, axis),
                played_drive(expected.events, axis),
                atol=2e-10,
                rtol=2e-10,
            )
        else:
            assert materialized == expected
        assert request.timelines[i] == scalar
        assert request.controls[i] == scalar.events
    if case == "variable_repeat":
        assert len(request.templates) == 1
        assert [e.active.tolist() for e in request.templates[0].events[:2]] == [
            [False, True, True],
            [False, True, False],
        ]
    if case == "computed":
        assert len(request.templates) == 1
        slot = request.templates[0].events[0]
        assert len(slot.envelopes) == 2
        assert slot.envelope_index.tolist() == [1, 0, 1]
    if case in ("branch", "branch_route"):
        assert len(request.templates) == 2
    if case == "flat_top_sweep":
        assert len(request.templates) == 2
        assert sorted(len(t.events) for t in request.templates) == [2, 4]


@pytest.mark.parametrize("dependency", ["sweep", "register", "branch"])
def test_reps_dependent_timeline_rejected_before_backend(dependency):
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    cfg = drive()
    modules = [cfg.build("drive"), receiver().build("readout")]
    if dependency == "sweep":
        cfg.gain = QickParam(0.1, {"reps": 0.2})
    elif dependency == "register":
        modules.insert(0, LoadValue("count", [1, 2], "reps", "n"))
        modules[1] = Repeat("inner", "n", range_hint=(1, 2)).add_content(
            cfg.build("drive")
        )
    else:
        modules[0] = Branch("choice", cfg.build("a"), cfg.build("b"), compare_by="reps")
    prog = ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=2), modules=modules)
    with pytest.raises(
        ValueError, match="^simulation requires reps not to change the timeline$"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_interpreter_dispatch_work_is_independent_of_points_and_reps():
    def dispatches(points, reps):
        env = white_noise(std_adc=0)
        cfg = receiver()
        cfg.freq = QickParam(100, {"axis": 20})
        prog = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=reps),
            modules=[cfg.build("readout")],
            sweep=[("axis", points)],
        )
        calls = 0

        def observe(frame, event, arg):
            nonlocal calls
            if (
                event == "call"
                and frame.f_code.co_name == "run"
                and "/simulation/" in frame.f_code.co_filename
                and isinstance(frame.f_locals.get("module"), Module)
            ):
                calls += 1

        previous = sys.getprofile()
        try:
            sys.setprofile(observe)
            build_request(prog, "integrated", env.soc.binding, 0)
        finally:
            sys.setprofile(previous)
        return calls

    small = dispatches(3, 1)
    assert small > 0, "count actual interpreter Module dispatch, not elapsed time"
    assert dispatches(6, 1) == small
    assert dispatches(3, 4) == small


def test_table_hook_initial_state_that_changes_across_reps_is_rejected():
    env = white_noise(std_adc=0)
    gen, adc = readout_freq_words(
        env.soccfg, [200, 300], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    module = TablePulseReadout(
        "readout", receiver(), idx_reg="axis", freq_words=gen, ro_freq_words=adc
    )
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=2), modules=[module], sweep=[("axis", 2)]
    )
    with pytest.raises(
        ValueError, match="^simulation requires reps not to change the timeline$"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


@pytest.mark.parametrize("missing", ["generator", "route"])
def test_soc_routes_are_checked_from_templates_without_legacy_timelines(missing):
    env = white_noise(
        std_adc=0,
        n_gens=2,
        routes=((0, 0),) if missing == "generator" else ((0, 0), (1, 1)),
        n_readouts=2,
    )
    cfg = receiver()
    cfg.gen_ch = 1
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(),
        modules=[receiver().build("first"), cfg.build("readout")],
    )
    request = build_request(prog, "integrated", env.soc.binding, 0)
    request = replace(
        request,
        controls=(),
        timelines=(),
        readouts=tuple(
            replace(w, gen_ch=0, generator_channels=()) for w in request.readouts
        ),
    )
    with pytest.raises(
        ValueError,
        match="generator 1 is not bound"
        if missing == "generator"
        else "unbound generator/ADC route",
    ):
        env.soc.install_request(request)
