from zcu_tools.program.v2 import TimelineBuilder
"""Repeated compiled requests preserve numeric controls and detached envelopes."""

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.simulation.compiled import build_compiled_request

from tests.program.v2.simulation.time_observations import time_us


@pytest.fixture
def environment():
    env = white_noise(
        std_adc=0, seed=4, n_gens=2, n_readouts=2, routes=((0, 0), (1, 1))
    )
    yield env
    env.devices.close_all_devices()


def repeated_program(env, case):
    """Exercise changing masks/timestamps with one initialized pulse per Repeat."""
    cfg = PulseCfg(
        ch=1,
        nqz=1,
        freq=QickParam(80, {"axis": 13.1}),
        phase=QickParam(3, {"axis": 37}),
        gain=QickParam(-0.3, {"axis": 0.1}),
        pre_delay=QickParam(0.013, {"axis": 0.031}),
        post_delay=0.017,
        waveform=ConstWaveformCfg(length=QickParam(0.1, {"axis": 0.04})),
    )
    if case == "flat_top":
        cfg.waveform = FlatTopWaveformCfg(
            length=QickParam(0.1078125, {"axis": 0.015625}),
            raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
        )
    if case.startswith("computed") or case in ("dynamic", "branch"):
        cfg.pre_delay = 0.013
        cfg.waveform = ConstWaveformCfg(length=0.1)
    pulse: Module = Pulse("drive", cfg)
    if case.startswith("computed"):
        shaped = cfg.with_updates(waveform=GaussWaveformCfg(length=0.1, sigma=0.02))
        candidates = [cfg, shaped, cfg]
        if case != "computed":
            candidates = [
                cfg,
                cfg.with_updates(waveform=GaussWaveformCfg(length=0.27, sigma=0.04)),
                cfg.with_updates(waveform=ConstWaveformCfg(length=0.055)),
            ]
        if case == "computed_flat_top":
            candidates = [
                cfg.with_updates(
                    waveform=FlatTopWaveformCfg(
                        length=length,
                        raise_waveform=GaussWaveformCfg(length=0.05, sigma=0.01),
                    )
                )
                for length in (0.06, 0.15, 0.23)
            ]
        pulse = ComputedPulse(
            "gate",
            val_reg="axis" if case == "computed" else "selected",
            pulses=candidates,
        )
    elif case == "dynamic":
        cfg.waveform = FlatTopWaveformCfg(
            length=0.1,
            raise_waveform=GaussWaveformCfg(length=0.05, sigma=0.01),
        )
        pulse = TableLengthPulse(
            "dynamic", cfg, lengths=[0.06, 0.12, 0.2], idx_reg="axis"
        )
    body: list[Module] = [
        SoftDelay(
            "before",
            QickParam(0.017, {"axis": 0.019})
            if case == "computed_qick_time"
            else 0.017,
        ),
        pulse,
        SoftDelay("after", 0.009),
    ]
    if case.startswith("computed_"):
        body.insert(0, LoadValue("selection", [2, 0], "inner", "selected"))
    if case == "computed_tables":
        body.append(ComputedPulse("second_gate", val_reg="inner", pulses=[cfg, cfg]))
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    builder = TimelineBuilder(env.soccfg, reps=2)
    builder.add_loop("axis", 3)
    builder.emit(LoadValue("counts", [0, 2, 1], "axis", "count"))
    with builder.repeat("outer", "count", range_hint=(0, 2)):
        with builder.repeat("inner", 2):
            for module in body:
                if module is pulse and case == "branch":
                    with builder.branch(by="axis", align_timeline=True) as cases:
                        for index, label in enumerate(["drive", "other", "third"]):
                            with cases.case(index):
                                builder.emit(cfg.build(label))
                else:
                    builder.emit(module)
    builder.emit(readout.build("readout"))
    builder.delay_by_timeline()
    return builder.build()


@pytest.mark.parametrize(
    "case",
    [
        "swept",
        "flat_top",
        "computed",
        "computed_selection",
        "computed_tables",
        "computed_qick_time",
        "computed_flat_top",
        "dynamic",
        "branch",
    ],
)
def test_repeated_request_preserves_numeric_controls(environment, case):
    prog = repeated_program(environment, case)
    request = build_compiled_request(
        prog, "integrated", environment.soc.binding, 0, routes=environment.soc.routes
    )
    assert request.point_names == ("axis",)
    assert request.point_dims == (3,)
    assert request.reps == 2
    for point, count in enumerate([0, 2, 1]):
        iterations = {
            event.iterations
            for event in request.materialize_point(point).events
            if event.source != "readout"
        }
        assert iterations == {
            (("outer", outer), ("inner", inner))
            for outer in range(count)
            for inner in range(2)
        }
    if case == "dynamic":
        assert {
            event.source for template in request.templates for event in template.events
        } == {"dynamic", "readout"}
    if case.startswith("computed_"):
        assert not any(
            event.source == "gate" for event in request.materialize_point(0).events
        )
        for point in (1, 2):
            gates = [
                event
                for event in request.materialize_point(point).events
                if event.source == "gate"
            ]
            for event in gates:
                # The register changes on every inner iteration, not every point.
                inner = dict(event.iterations)["inner"]
                gain = -0.25 if point == 1 else -0.2
                if case == "computed_flat_top":
                    assert any(
                        event.gain == pytest.approx(value, abs=1e-4)
                        for value in (gain, gain / 2)
                    )
                else:
                    assert event.gain == pytest.approx(gain, abs=1e-4)
                if case != "computed_flat_top":
                    assert time_us(
                        request.time_base, event.duration_ticks
                    ) == pytest.approx(0.055 if inner == 0 else 0.1, abs=0.003)
            if case == "computed_tables":
                second = [
                    event
                    for event in request.materialize_point(point).events
                    if event.source == "second_gate"
                ]
                assert len(second) == len(gates)
                assert time_us(request.time_base, second[0].start_ticks) > time_us(
                    request.time_base, gates[0].start_ticks
                ) + time_us(request.time_base, gates[0].duration_ticks)


def test_next_request_reads_current_compiled_pulse_content(environment):
    prog = repeated_program(environment, "computed")
    first = build_compiled_request(
        prog, "integrated", environment.soc.binding, 0, routes=environment.soc.routes
    )
    original = tuple(first.materialize_point(point) for point in range(3))
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    next(iter(table.values()))["data"][:, 1] = 7
    second = build_compiled_request(
        prog, "integrated", environment.soc.binding, 0, routes=environment.soc.routes
    )
    changed = tuple(second.materialize_point(point) for point in range(3))
    assert changed != original
    assert tuple(first.materialize_point(point) for point in range(3)) == original
    envelopes = [
        event.envelope
        for point in changed
        for event in point.events
        if event.source == "gate" and event.envelope is not None
    ]
    assert envelopes
    assert all(
        value.imag == pytest.approx(7 / prog.soccfg.get_maxv(1))
        for envelope in envelopes
        for value in envelope.values
    )
    third = build_compiled_request(
        prog, "integrated", environment.soc.binding, 0, routes=environment.soc.routes
    )
    assert tuple(third.materialize_point(point) for point in range(3)) == changed
