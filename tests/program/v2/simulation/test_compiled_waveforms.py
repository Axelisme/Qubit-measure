"""Compiled generator command scheduling, samples, and control-bit contracts."""

import numpy as np
import pytest
from qick.asm_v2 import QickRawParam
from zcu_tools.core.simulation import TimeBase

from tests.program._timing_expectations import generator_duration, generator_tproc_grid
from tests.program.v2.simulation._compiled_support import program, request
from tests.program.v2.simulation.time_observations import envelope_times_us, time_us


def test_native_arb_preserves_actual_compiled_dac_samples_and_scale():
    def initialize(p):
        p.add_envelope(
            0, "shape", idata=np.full(48, -8191.0), qdata=np.full(48, 4096.0)
        )
        p.add_pulse(
            0, "shaped", style="arb", envelope="shape", freq=80, phase=10, gain=0.5
        )

    prog, env = program(lambda p: p.pulse(0, "shaped", tag="arb"), initialize, reps=1)
    observed = request(prog, env)
    event = observed.materialize_point(0).events[0]
    assert event.envelope is not None
    gen = env.soccfg["gens"][0]
    sample_mhz = gen["f_fabric"] * gen["samps_per_clk"]
    expected_base = TimeBase.from_clocks(
        (
            env.soccfg["tprocs"][0]["f_time"],
            gen["f_fabric"],
            sample_mhz,
            env.soccfg["readouts"][0]["f_output"],
        )
    )
    sample_ticks = expected_base.period_ticks(sample_mhz)
    assert observed.time_base == expected_base
    assert event.envelope.sample_period_ticks == sample_ticks
    assert event.duration_ticks == 48 * sample_ticks
    assert envelope_times_us(observed.time_base, event.envelope)[:2] == pytest.approx(
        (0, 1 / sample_mhz)
    )
    assert event.envelope.values[0] == pytest.approx(
        complex(-8191, 4096) / gen["maxv"]
    )
    assert time_us(observed.time_base, event.duration_ticks) == pytest.approx(
        48 / sample_mhz
    )


@pytest.mark.parametrize(
    "timestamps",
    [[0, 0], [0, 0.05], [0, 0.1], [0, 0.3], [0.1, 0]],
)
def test_ordinary_generator_serializes_commands_in_instruction_order(timestamps):
    def body(p):
        for timestamp in timestamps:
            p.pulse(0, "p", t=timestamp)

    prog, env = program(body, reps=1)
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    scheduled = [prog.cycles2us(prog.us2cycles(t)) for t in timestamps]
    duration = float(generator_duration(env.soccfg, 0.1))
    expected = [scheduled[0], max(scheduled[1], scheduled[0] + duration)]
    assert [
        time_us(observed.time_base, e.start_ticks) for e in events
    ] == pytest.approx(expected)


def test_adjacent_ordinary_commands_keep_the_exact_shared_boundary():
    def initialize(p):
        p.add_pulse(
            0,
            "adjacent",
            style="const",
            freq=100,
            phase=0,
            gain=0.1,
            length=p.cycles2us(generator_tproc_grid(p.soccfg), gen_ch=0),
        )

    def body(p):
        p.pulse(0, "adjacent", t=0)
        p.pulse(0, "adjacent", t=p.cycles2us(generator_tproc_grid(p.soccfg), gen_ch=0))

    prog, env = program(body, initialize, reps=1)
    observed = request(prog, env)
    first, second = observed.materialize_point(0).events
    boundary = env.soccfg.cycles2us(generator_tproc_grid(env.soccfg), gen_ch=0)
    boundary_ticks = observed.time_base.ticks(generator_duration(env.soccfg, boundary))
    assert first.start_ticks == 0
    assert first.duration_ticks == second.start_ticks == boundary_ticks
    assert (
        time_us(observed.time_base, first.duration_ticks)
        == time_us(observed.time_base, second.start_ticks)
        == boundary
    )


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
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    gen = env.soccfg["gens"][0]
    ramp = 96 / (2 * gen["samps_per_clk"] * gen["f_fabric"])
    flat = float(generator_duration(env.soccfg, 0.1))
    assert [
        time_us(observed.time_base, e.start_ticks) for e in events
    ] == pytest.approx([0, ramp, ramp + flat, 0])
    assert [
        time_us(observed.time_base, e.duration_ticks) for e in events
    ] == pytest.approx([ramp, flat, ramp, flat])


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
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    assert len(events) == (1 if outsel == "zero" else 2)
    gen = env.soccfg["gens"][0]
    duration = 48 / (gen["samps_per_clk"] * gen["f_fabric"])
    assert time_us(observed.time_base, events[-1].start_ticks) == pytest.approx(duration)
    if outsel == "product":
        envelope = events[0].envelope
        assert envelope is not None
        assert envelope.values[0] == pytest.approx(8191 / gen["maxv"])
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
    with pytest.raises(ValueError, match="RegWriteInst.*preopt.*cross-shot.*live-in"):
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
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    assert [e.gain for e in events] == pytest.approx(
        np.array([16777223, 65545, 7, 9]) / env.soccfg["gens"][0]["maxv"]
    )
    assert [
        time_us(observed.time_base, e.duration_ticks) for e in events
    ] == pytest.approx([float(generator_duration(env.soccfg, 0.1))] * 4)
