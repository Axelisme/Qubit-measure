"""Compiled generator command scheduling, samples, and control-bit contracts."""

import numpy as np
import pytest
from qick.asm_v2 import QickRawParam

from tests.program.v2.simulation._compiled_support import program, request


def test_native_arb_preserves_actual_compiled_dac_samples_and_scale():
    def initialize(p):
        p.add_envelope(
            0, "shape", idata=np.full(48, -8191.0), qdata=np.full(48, 4096.0)
        )
        p.add_pulse(
            0, "shaped", style="arb", envelope="shape", freq=80, phase=10, gain=0.5
        )

    prog, env = program(lambda p: p.pulse(0, "shaped", tag="arb"), initialize, reps=1)
    event = request(prog, env).materialize_point(0).events[0]
    assert event.envelope is not None
    assert event.envelope.sample_us[:2] == pytest.approx((0, 1 / 6144))
    assert event.envelope.values[0] == pytest.approx(complex(-8191, 4096) / 32766)
    assert event.duration_us == pytest.approx(3 / 384)


@pytest.mark.parametrize(
    "timestamps,expected",
    [
        ([0, 0], [0, 38 / 384]),
        ([0, 0.05], [0, 38 / 384]),
        ([0, 0.1], [0, 43 / 430.08]),
        ([0, 0.3], [0, 129 / 430.08]),
        ([0.1, 0], [43 / 430.08, 43 / 430.08 + 38 / 384]),
    ],
)
def test_ordinary_generator_serializes_commands_in_instruction_order(
    timestamps, expected
):
    def body(p):
        for timestamp in timestamps:
            p.pulse(0, "p", t=timestamp)

    prog, env = program(body, reps=1)
    events = request(prog, env).materialize_point(0).events
    assert [e.start_us for e in events] == pytest.approx(expected)


def test_adjacent_ordinary_commands_keep_the_exact_shared_boundary():
    def initialize(p):
        p.add_pulse(
            0, "adjacent", style="const", freq=100, phase=0, gain=0.1, length=25 / 384
        )

    def body(p):
        p.pulse(0, "adjacent", t=0)
        p.pulse(0, "adjacent", t=25 / 384)

    prog, env = program(body, initialize, reps=1)
    first, second = request(prog, env).materialize_point(0).events
    assert first.start_us == 0
    assert first.duration_us == second.start_us == 25 / 384


def test_flat_top_segments_are_adjacent_without_delaying_another_generator():
    def initialize(p):
        p.add_envelope(0, "ramp", idata=np.arange(96, dtype=float))
        p.add_pulse(
            0,
            "flat",
            style="flat_top",
            envelope="ramp",
            freq=100,
            phase=0,
            gain=0.5,
            length=0.1,
        )
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "other", style="const", freq=200, phase=0, gain=0.1, length=0.1)

    def body(p):
        p.pulse(0, "flat", t=0)
        p.pulse(1, "other", t=0)

    prog, env = program(body, initialize, reps=1)
    events = request(prog, env).materialize_point(0).events
    assert [e.start_us for e in events] == pytest.approx([0, 3 / 384, 41 / 384, 0])
    assert [e.duration_us for e in events] == pytest.approx(
        np.array([3, 38, 3, 38]) / 384
    )


@pytest.mark.parametrize("outsel", ["zero", "dds", "product"])
def test_ordinary_outsel_preserves_command_occupancy_and_envelope(outsel):
    def initialize(p):
        p.add_envelope(0, "shape", idata=np.full(48, 8191.0))
        p.add_pulse(
            0,
            "controlled",
            style="arb",
            envelope="shape",
            freq=100,
            phase=0,
            gain=0.5,
            outsel=outsel,
        )

    def body(p):
        p.pulse(0, "controlled", t=0)
        p.pulse(0, "p", t=0)

    prog, env = program(body, initialize, reps=1)
    events = request(prog, env).materialize_point(0).events
    assert len(events) == (1 if outsel == "zero" else 2)
    assert events[-1].start_us == pytest.approx(3 / 384)
    if outsel == "product":
        envelope = events[0].envelope
        assert envelope is not None
        assert envelope.values[0] == pytest.approx(8191 / 32766)
    elif outsel == "dds":
        assert events[0].envelope is None


@pytest.mark.parametrize(
    "parameter,value,reason",
    [
        ("outsel", "input", "outsel=input"),
        ("mode", "periodic", "mode=periodic"),
        ("stdysel", "last", "stdysel=last"),
    ],
)
def test_unrepresentable_ordinary_control_bits_fail_with_context(
    parameter, value, reason
):
    def initialize(p):
        p.add_envelope(0, "shape", idata=np.full(48, 8191.0))
        p.add_pulse(
            0,
            "controlled",
            style="arb",
            envelope="shape",
            freq=100,
            phase=0,
            gain=0.5,
            **{parameter: value},
        )

    prog, env = program(lambda p: p.pulse(0, "controlled"), initialize, reps=1)
    with pytest.raises(ValueError, match=rf"PortWriteInst.*preopt.*reps.*{reason}"):
        request(prog, env)


@pytest.mark.parametrize("last_sample", [0.0, 1.0])
def test_stdysel_last_accepts_only_an_exact_zero_final_dac_sample(last_sample):
    def initialize(p):
        data = np.full(48, 8191.0)
        data[-1] = last_sample
        p.add_envelope(0, "shape", idata=data)
        p.add_pulse(
            0,
            "controlled",
            style="arb",
            envelope="shape",
            freq=100,
            phase=0,
            gain=0.5,
            stdysel="last",
        )

    prog, env = program(lambda p: p.pulse(0, "controlled"), initialize, reps=1)
    if last_sample:
        with pytest.raises(ValueError, match="stdysel=last"):
            request(prog, env)
    else:
        envelope = request(prog, env).materialize_point(0).events[0].envelope
        assert envelope is not None
        assert envelope.values[-1] == 0


@pytest.mark.parametrize("phrst", [0, 1])
def test_phase_reset_bit_is_not_silently_ignored(phrst):
    def initialize(p):
        p.add_pulse(
            0,
            "reset",
            style="const",
            freq=100,
            phase=30,
            gain=0.1,
            length=0.1,
            phrst=phrst,
        )

    prog, env = program(lambda p: p.pulse(0, "reset"), initialize, reps=1)
    if phrst:
        with pytest.raises(ValueError, match="PortWriteInst.*preopt.*reps.*phrst=1"):
            request(prog, env)
    else:
        assert request(prog, env).materialize_point(0).events[
            0
        ].phase_deg == pytest.approx(30, abs=1e-5)


def test_reps_varying_envelope_selection_fails_before_request():
    def initialize(p):
        p.add_reg("envaddr", init=QickRawParam("envaddr", 0, {}))
        for name, sample in (("first", 8191.0), ("second", -8191.0)):
            p.add_envelope(0, name, idata=np.full(48, sample))
        p.add_pulse(
            0, "shaped", style="arb", envelope="first", freq=100, phase=0, gain=0.5
        )

    def body(p):
        p.read_wmem(p.pulses["shaped"].get_wavenames()[0])
        p.write_reg("w_env", "envaddr")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})
        p.inc_reg("envaddr", 3)

    prog, env = program(body, initialize)
    with pytest.raises(
        ValueError, match="PortWriteInst.*preopt.*reps.*envelope.*carried"
    ):
        request(prog, env)


def test_wave_register_readback_and_wmem_field_widths_are_distinct():
    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_env", 16777223)
        p.write_reg("w_conf", 65545)
        p.write_reg("w_length", 65574)
        for time, field in ((0, "w_env"), (50, "w_conf")):
            p.write_reg("w_gain", field)
            p.asm_inst(
                {"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": f"@{time}"}
            )
        p.asm_inst({"CMD": "WMEM_WR", "DST": "&17"})
        p.asm_inst({"CMD": "REG_WR", "DST": "r_wave", "SRC": "wmem", "ADDR": "&17"})
        for time, field in ((100, "w_env"), (150, "w_conf")):
            p.write_reg("w_gain", field)
            p.asm_inst(
                {"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": f"@{time}"}
            )

    prog, env = program(body, reps=1)
    events = request(prog, env).materialize_point(0).events
    assert [e.gain for e in events] == pytest.approx(
        np.array([16777223, 65545, 7, 9]) / 32766
    )
    assert [e.duration_us for e in events] == pytest.approx([38 / 384] * 4)
