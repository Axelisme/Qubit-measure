"""Compiled request outputs from public native and Module compile seams."""

import numpy as np
import pytest
from qick.asm_v2 import QickRawParam, QickSweep1D
from zcu_tools.core.simulation import ChargeDriveRequirement
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.ir import CompiledRecord, CompiledStream
from zcu_tools.program.v2.ir.instructions import MetaInst
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program.v2._mux_support import mux_soccfg, native_mux
from tests.program.v2.simulation._compiled_support import program, request
from tests.program.v2.simulation.time_observations import time_us, times_us


def test_native_pulse_request_has_detached_numeric_event_and_adc_slots():
    prog, env = program(lambda p: p.pulse(0, "p", tag="drive"))
    result = request(prog, env)
    point = result.materialize_point(0)
    assert result.loop_dims == (2,)
    assert result.point_dims == ()
    assert result.sequence == 7
    assert result.time_base.ticks_per_us == 43008
    assert result.relax_delay_ticks == 4300
    template = result.templates[0]
    assert template.events[0].start_ticks.dtype == np.int64
    assert template.events[0].start_ticks.shape == (1,)
    assert template.events[0].duration_ticks.tolist() == [4256]
    assert template.duration_ticks.dtype == np.int64
    assert template.readouts[0].trigger_ticks.dtype == np.int64
    assert template.readouts[0].trigger_ticks.shape == (1, 1)
    assert template.readouts[0].sample_period_ticks == 140
    assert point.time_base == result.time_base
    (event,) = point.events
    assert event.source == "drive"
    assert event.start_ticks == 0
    assert event.duration_ticks == 4256
    assert event.phase_reference_ticks == 0
    assert event.frequency_mhz == pytest.approx(100, abs=1e-5)
    assert event.phase_deg == pytest.approx(30, abs=1e-5)
    assert event.gain == pytest.approx(-0.25, abs=1 / 32766)
    (read,) = point.readouts
    assert read.gen_ch == 0
    assert read.samples == 61
    assert read.trigger_ticks == (0,)
    assert read.sample_period_ticks == 140
    assert read.frequency_mhz == pytest.approx((100,), abs=1e-5)
    assert env.soc.binding.last_request is None


def test_extra_declared_clock_preserves_original_channel_physical_time_bits():
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0),))
    env.soccfg["gens"][1]["f_fabric"] = 512.0

    def build(extra):
        class Program(TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare_gen(0, nqz=1)
                if extra:
                    self.declare_gen(1, nqz=1)
                self.add_envelope(0, "shape", idata=np.full(48, -8191.0))
                self.add_pulse(
                    0,
                    "p",
                    style="arb",
                    envelope="shape",
                    freq=100,
                    phase=30,
                    gain=-0.25,
                )
                self.declare_readout(0, length=0.2)
                self.add_readoutconfig(0, "ro", freq=100, gen_ch=0)

            def _body(self, cfg):
                self.send_readoutconfig(0, "ro", t=0)
                self.pulse(0, "p", t=0.1)
                self.trigger([0], t=0.1)

        prog = Program(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
        return request(prog, env)

    try:
        original, expanded = build(False), build(True)
        assert original.time_base.ticks_per_us == 43008
        assert expanded.time_base.ticks_per_us == 172032
        a, b = original.templates[0], expanded.templates[0]
        (event_a,), (event_b,) = a.events, b.events
        assert event_a.gen_ch == event_b.gen_ch == 0
        assert event_a.start_ticks.tolist() == [4300]
        for left, right in (
            (event_a.start_ticks, event_b.start_ticks),
            (
                event_a.start_ticks + event_a.duration_ticks,
                event_b.start_ticks + event_b.duration_ticks,
            ),
            (a.duration_ticks, b.duration_ticks),
        ):
            np.testing.assert_array_equal(
                original.time_base.ns(left).view(np.uint64),
                expanded.time_base.ns(right).view(np.uint64),
            )
        for left, right in (
            (event_a.frequency_mhz, event_b.frequency_mhz),
            (event_a.phase_deg, event_b.phase_deg),
            (event_a.gain, event_b.gain),
        ):
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        envelope_a = event_a.envelopes[int(event_a.envelope_index[0])]
        envelope_b = event_b.envelopes[int(event_b.envelope_index[0])]
        assert envelope_a is not None and envelope_b is not None
        np.testing.assert_array_equal(
            original.time_base.ns(
                np.arange(len(envelope_a.values), dtype=np.int64)
                * envelope_a.sample_period_ticks
            ).view(np.uint64),
            expanded.time_base.ns(
                np.arange(len(envelope_b.values), dtype=np.int64)
                * envelope_b.sample_period_ticks
            ).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(envelope_a.values).view(np.uint64),
            np.asarray(envelope_b.values).view(np.uint64),
        )
        ro_a, ro_b = a.readouts[0], b.readouts[0]
        np.testing.assert_array_equal(
            original.time_base.ns(
                ro_a.trigger_ticks[..., None]
                + np.arange(ro_a.samples, dtype=np.int64) * ro_a.sample_period_ticks
            ).view(np.uint64),
            expanded.time_base.ns(
                ro_b.trigger_ticks[..., None]
                + np.arange(ro_b.samples, dtype=np.int64) * ro_b.sample_period_ticks
            ).view(np.uint64),
        )
        np.testing.assert_array_equal(
            ro_a.frequency_mhz.view(np.uint64), ro_b.frequency_mhz.view(np.uint64)
        )
        np.testing.assert_array_equal(
            ro_a.phase_deg.view(np.uint64), ro_b.phase_deg.view(np.uint64)
        )
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_compiled_request_preserves_call_local_charge_requirements(mode):
    prog, env = program(lambda p: p.pulse(0, "p"))
    requirements = (
        ChargeDriveRequirement("pump", 1),
        ChargeDriveRequirement("probe", 0),
    )
    result = build_compiled_request(
        prog,
        mode,
        env.soc.binding,
        7,
        routes=env.soc.routes,
        charge_requirements=requirements,
    )
    assert result.charge_requirements == requirements
    ordinary = build_compiled_request(
        prog, mode, env.soc.binding, 8, routes=env.soc.routes
    )
    assert ordinary.charge_requirements == ()
    assert result.charge_requirements == requirements
    assert env.soc.binding.last_request is None


def test_native_sweep_shell_preserves_two_axes_wave_and_register_updates():
    def initialize(p):
        p.add_loop("outer", 2)
        p.add_loop("scan", 3)
        p.add_pulse(
            0,
            "swept",
            style="const",
            freq=80 + QickSweep1D("outer", 0, 100) + QickSweep1D("scan", 0, 20),
            phase=0,
            gain=QickSweep1D("scan", 0.1, 0.3),
            length=0.1,
        )

    prog, env = program(
        lambda p: p.pulse(0, "swept", t=QickSweep1D("scan", 0, 0.04)), initialize
    )
    result = request(prog, env)
    assert result.point_names == ("outer", "scan")
    assert result.point_dims == (2, 3)
    slot = result.templates[0].events[0]
    assert slot.start_ticks.dtype == np.int64
    assert slot.start_ticks.shape == (6,)
    assert slot.start_ticks.tolist() == [0, 800, 1600, 0, 800, 1600]
    assert slot.duration_ticks.tolist() == [4256] * 6
    events = [result.materialize_point(i).events[0] for i in range(6)]
    assert [e.frequency_mhz for e in events] == pytest.approx(
        [80, 90, 100, 180, 190, 200], abs=1e-5
    )
    assert [e.gain for e in events] == pytest.approx(
        np.tile([3276, 6552, 9828], 2) / 32766
    )
    assert [time_us(result.time_base, e.start_ticks) for e in events] == pytest.approx(
        np.tile([0, 8, 16], 2) / 430.08
    )


def test_register_and_wmem_live_in_is_rejected_across_points():
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("carry", init=QickRawParam("carry", 10, {}))

    def body(p):
        p.inc_reg("carry", 1)
        wave = p.pulses["p"].get_wavenames()[0]
        p.read_wmem(wave)
        p.write_reg("w_gain", "carry")
        p.write_wmem(wave)
        p.pulse(0, "p", tag="accumulated")

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize(
    "effect,instruction",
    [
        ("wait", "WaitInst"),
        ("wait_auto", "WaitInst"),
        ("resync", "RegWriteInst"),
        ("jump", "JumpInst"),
        ("read", "DportReadInst"),
    ],
)
def test_body_clock_and_unstructured_effects_fail_with_preopt_context(
    effect, instruction
):
    def body(p):
        p.pulse(0, "p")
        with p.source_scope("unsupported"):
            if effect == "wait":
                p.wait(0.1)
            elif effect == "wait_auto":
                p.wait_auto()
            elif effect == "resync":
                p.resync(0.1)
            elif effect == "jump":
                p.jump("HERE")
            else:
                p.read_input(0)

    prog, env = program(body, reps=1)
    with pytest.raises(ValueError, match=rf"{instruction}.*preopt.*reps.*unsupported"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_core_modules_keep_shared_cfg_sources_and_delay_effects():
    drive = PulseCfg(
        ch=1, nqz=1, freq=80, gain=0.1, waveform=ConstWaveformCfg(length=0.1)
    )
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0),))
    prog = ModularProgramV2(
        env.soccfg,
        modules=[
            SoftDelay("soft", 0.1),
            drive.build("one"),
            Delay("delay", 0.1),
            drive.build("two"),
            DelayAuto("align"),
            readout.build("readout"),
        ],
        reps=2,
    )
    result = request(prog, env)
    events = result.materialize_point(0).events
    assert [e.source for e in events[:2]] == ["one", "two"]
    assert [
        time_us(result.time_base, e.start_ticks) for e in events[:2]
    ] == pytest.approx([44 / 430.08, 132 / 430.08])
    assert len(events) == 3
    assert result.materialize_point(0).readouts[0].generator_channels == (0,)


def test_native_ordinals_ignore_module_pulses_in_mixed_compiled_body():
    module = PulseCfg(
        ch=1, nqz=1, freq=80, gain=0.1, waveform=ConstWaveformCfg(length=0.1)
    ).build("module")

    def initialize(p):
        p.declare(module)

    def body(p):
        p.pulse(0, "p")
        p.emit(module)
        p.pulse(0, "p", t=0.1)

    prog, env = program(body, initialize, reps=1)
    events = request(prog, env).materialize_point(0).events
    assert [e.source for e in events] == ["native:pulse:1", "module", "native:pulse:2"]


def test_compiled_mux_output_keeps_tone_and_declared_adc_order():
    prog = ModularProgramV2(mux_soccfg(), modules=[native_mux().build("mux")], reps=2)
    env = white_noise(std_adc=0, n_gens=2, n_readouts=9, routes=((0, 4), (0, 1)))
    result = request(prog, env)
    point = result.materialize_point(0)
    assert [e.frequency_mhz for e in point.events] == pytest.approx(
        [1000, 1080], abs=1e-5
    )
    assert [e.gain for e in point.events] == pytest.approx([0.1, 0.1], abs=1e-4)
    assert [r.channel for r in point.readouts] == [4, 1]
    assert [r.frequency_mhz[0] for r in point.readouts] == pytest.approx(
        [1000, 1080], abs=1e-5
    )


def test_static_mux_readout_retains_absolute_frequencies_without_any_pulse():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(
                0,
                nqz=1,
                mixer_freq=1000,
                ro_ch=4,
                mux_freqs=[1000, 1080],
                mux_gains=[0.1, 0.1],
                mux_phases=[0, 45],
            )
            for channel, frequency in ((4, 1000), (1, 1080)):
                self.declare_readout(
                    channel, length=0.1, freq=frequency, phase=0, gen_ch=0
                )
            self.add_pulse(0, "unused", style="const", mask=[0, 1], length=0.1)

        def _body(self, cfg):
            self.trigger([4, 1], t=0)

    prog = Program(mux_soccfg(), reps=1, initial_delay=0, relax_delay=0)
    env = white_noise(std_adc=0, n_gens=2, n_readouts=9, routes=((0, 4), (0, 1)))
    point = request(prog, env).materialize_point(0)
    assert point.events == ()
    assert [r.frequency_mhz[0] for r in point.readouts] == pytest.approx(
        [1000, 1080], abs=1e-5
    )


def test_native_high_carrier_and_adc_alias_decode_through_frequency_owner():
    def initialize(p):
        p.add_pulse(0, "high", style="const", freq=7000, phase=0, gain=0.1, length=0.1)
        p.add_readoutconfig(0, "high_ro", freq=7000, gen_ch=0)

    def body(p):
        p.send_readoutconfig(0, "high_ro")
        p.pulse(0, "high", t=0)

    prog, env = program(body, initialize, reps=1)
    point = request(prog, env).materialize_point(0)
    assert point.events[0].frequency_mhz == pytest.approx(7000, abs=1e-5)
    assert point.readouts[0].frequency_mhz == pytest.approx((7000,), abs=1e-5)


def test_dmem_store_initializes_an_unallocated_word_and_read_updates_pulse():
    def body(p):
        p.write_dmem(17, 1234)
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.read_dmem("w_gain", 17)
        p.write_wmem(p.pulses["p"].get_wavenames()[0])
        p.pulse(0, "p", t=0.1, tag="loaded")

    prog, env = program(body, reps=1)
    result = request(prog, env)
    assert result.materialize_point(0).events[-1].gain == pytest.approx(1234 / 32766)


def test_reps_varying_pulse_rejects_carried_register_and_memory_context():
    def initialize(p):
        p.add_reg("carry", init=QickRawParam("carry", 0, {}))

    def body(p):
        p.inc_reg("carry", 1)
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "carry")
        p.write_wmem(p.pulses["p"].get_wavenames()[0])
        p.pulse(0, "p")

    prog, env = program(body, initialize)
    register = prog.reg_dict["carry"].full_addr()
    with pytest.raises(
        ValueError,
        match=rf"RegWriteInst.*preopt.*cross-shot.*WMEM.*{register}",
    ):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize(
    "condition,gains",
    [
        ("S", [100, 10, 10]),
        ("Z", [10, 100, 10]),
        ("NS", [10, 100, 100]),
        ("NZ", [100, 10, 100]),
    ],
)
def test_flag_alu_uses_prewrite_operands_and_conditions_are_per_lane(condition, gains):
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("temp")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "scan")
        p.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": p.reg_dict["temp"].full_addr(),
                "SRC": "imm",
                "LIT": "#10",
                "OP": "w3 - #1",
                "UF": "1",
            }
        )
        p.write_reg("w_gain", 10)
        p.asm_inst(
            {"CMD": "REG_WR", "DST": "w3", "SRC": "imm", "LIT": "#100", "IF": condition}
        )
        p.asm_inst(
            {
                "CMD": "WPORT_WR",
                "DST": str(p.soccfg["gens"][0]["tproc_ch"]),
                "SRC": "r_wave",
                "TIME": "@0",
            }
        )

    prog, env = program(body, initialize)
    result = request(prog, env)
    assert [
        result.materialize_point(i).events[0].gain for i in range(3)
    ] == pytest.approx(np.asarray(gains) / 32766)


def test_conditionally_uninitialized_register_is_not_zero_filled_on_other_lanes():
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("temp")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "scan")
        p.asm_inst({"CMD": "TEST", "OP": "w3 - #1", "UF": "1"})
        p.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": p.reg_dict["temp"].full_addr(),
                "SRC": "imm",
                "LIT": "#5",
                "IF": "Z",
            }
        )
        p.write_reg("w_gain", "temp")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(
        ValueError,
        match=rf"preopt.*cross-shot.*live-in.*{prog.reg_dict['temp'].full_addr()}",
    ):
        request(prog, env)


@pytest.mark.parametrize("opcode", ["WPORT_WR", "WMEM_WR", "REG_WR"])
def test_wave_memory_and_output_coupled_effects_use_prewrite_words(opcode):
    def body(p):
        wave = p.pulses["p"].get_wavenames()[0]
        p.read_wmem(wave)
        p.write_reg("w_gain", 100)
        p.write_reg("s14", 0)
        address = f"&{p.wave2idx[wave]}"
        port = str(p.soccfg["gens"][0]["tproc_ch"])
        if opcode == "WPORT_WR":
            p.asm_inst(
                {
                    "CMD": opcode,
                    "SRC": "r_wave",
                    "DST": port,
                    "ADDR": address,
                    "WW": "1",
                    "TIME": "@0",
                }
            )
        elif opcode == "WMEM_WR":
            p.asm_inst(
                {
                    "CMD": opcode,
                    "DST": address,
                    "WP": "r_wave",
                    "PORT": port,
                    "TIME": "@0",
                }
            )
        else:
            p.asm_inst(
                {
                    "CMD": opcode,
                    "DST": "r_wave",
                    "SRC": "wmem",
                    "ADDR": address,
                    "WW": "1",
                    "WP": "r_wave",
                    "PORT": port,
                }
            )
        p.pulse(0, "p", t=0.1)

    prog, env = program(body, reps=1)
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    assert [event.gain for event in events] == pytest.approx([100 / 32766] * 2)
    assert [
        time_us(observed.time_base, event.start_ticks) for event in events
    ] == pytest.approx([0, 43 / 430.08])


def test_wmem_store_initializes_a_new_address_for_native_port_playback():
    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", 777)
        p.asm_inst({"CMD": "WMEM_WR", "DST": "&17"})
        p.asm_inst(
            {
                "CMD": "WPORT_WR",
                "DST": str(p.soccfg["gens"][0]["tproc_ch"]),
                "SRC": "wmem",
                "ADDR": "&17",
                "TIME": "@0",
            }
        )

    prog, env = program(body, reps=1)
    assert request(prog, env).materialize_point(0).events[0].gain == pytest.approx(
        777 / 32766
    )


@pytest.mark.parametrize("active,expected", [((1,), 1), ((), None), ((0, 1), None)])
def test_multi_generator_readout_route_uses_only_overlapping_wired_outputs(
    active, expected
):
    def initialize(p):
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "other", style="const", freq=200, phase=0, gain=0.1, length=0.1)

    def body(p):
        for ch in active:
            p.pulse(ch, "other" if ch else "p", t=0.31)

    prog, env = program(
        body, initialize, reps=1, routes=((0, 0), (1, 0)), trigger_time=0.3
    )
    if expected is None:
        with pytest.raises(ValueError, match="ADC 0 read 0.*candidates=.*time="):
            request(prog, env)
    else:
        assert request(prog, env).materialize_point(0).readouts[
            0
        ].generator_channels == (1,)


@pytest.mark.parametrize(
    "carriers,trigger_time,error",
    [
        ([], 0, "no pulse"),
        ([100], 0.3, "no pulse"),
        ([100, 3000], 0, "carriers=.*images="),
        ([100, 120], 0, None),
    ],
)
def test_runtime_adc_word_requires_one_common_window_frequency_image(
    carriers, trigger_time, error
):
    def initialize(p):
        for i, frequency in enumerate(carriers):
            p.add_pulse(
                0, f"c{i}", style="const", freq=frequency, phase=0, gain=0.1, length=0.1
            )

    def body(p):
        for i in range(len(carriers)):
            p.pulse(0, f"c{i}", t=i * 0.1)

    prog, env = program(body, initialize, reps=1, trigger_time=trigger_time)
    if error is not None:
        with pytest.raises(ValueError, match=rf"ADC 0 read 0.*{error}"):
            request(prog, env)
    else:
        assert request(prog, env).materialize_point(0).readouts[
            0
        ].frequency_mhz == pytest.approx((100,), abs=1e-5)


@pytest.mark.parametrize("boundary_tick", [7, 28])
@pytest.mark.parametrize("multiple_routes", [False, True])
def test_touching_window_endpoints_do_not_supply_routes_or_carrier_images(
    boundary_tick, multiple_routes
):
    def initialize(p):
        p.declare_gen(1, nqz=1)
        p.add_pulse(
            0,
            "boundary",
            style="const",
            freq=3000,
            phase=0,
            gain=0.1,
            length=7 / 430.08,
        )
        p.add_pulse(
            1 if multiple_routes else 0,
            "inside",
            style="const",
            freq=100,
            phase=0,
            gain=0.1,
            length=14 / 430.08,
        )

    def body(p):
        for tick, channel, name in sorted(
            [
                (boundary_tick, 0, "boundary"),
                (14, 1 if multiple_routes else 0, "inside"),
            ]
        ):
            p.pulse(channel, name, t=tick / 430.08)

    prog, env = program(
        body,
        initialize,
        reps=1,
        readout_length=10 / 307.2,
        trigger_time=14 / 430.08,
        routes=((0, 0), (1, 0)) if multiple_routes else ((0, 0),),
    )
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.generator_channels == ((1,) if multiple_routes else (0,))
    assert read.frequency_mhz == pytest.approx((100,), abs=1e-5)


def test_raw_trigger_count_mismatch_fails_with_static_and_actual_per_shot():
    def body(p):
        p.pulse(0, "p")
        p.asm_inst(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": "1", "TIME": "@0"}
        )
        p.asm_inst(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": "0", "TIME": "@10"}
        )

    prog, env = program(body, reps=1)
    with pytest.raises(ValueError, match="ADC 0.*static=1.*actual=2"):
        request(prog, env)


def test_declared_zero_read_adc_fails_before_request_with_channel_reason():
    env = white_noise(std_adc=0)

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_readout(0, length=0.1)
            self.add_readoutconfig(0, "ro", freq=100)

    prog = Program(env.soccfg, reps=1, initial_delay=0, relax_delay=0)
    with pytest.raises(ValueError, match="ADC 0.*zero-read"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_ordinary_adc_triggers_are_distinct_output_effects():
    env = white_noise(std_adc=0, n_gens=1, n_readouts=2, routes=((0, 0), (0, 1)))

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "read", style="const", freq=100, phase=0, gain=0.1, length=0.3
            )
            for ch in (0, 1):
                self.declare_readout(ch, length=0.1)
                self.add_readoutconfig(ch, f"ro{ch}", freq=100, gen_ch=0)

        def _body(self, cfg):
            self.pulse(0, "read")
            self.send_readoutconfig(0, "ro0")
            self.send_readoutconfig(1, "ro1")
            self.trigger([0], t=0)
            self.trigger([1], t=0.1)

    prog = Program(env.soccfg, reps=1, initial_delay=0, relax_delay=0)
    result = request(prog, env)
    reads = result.materialize_point(0).readouts
    assert times_us(result.time_base, reads[0].trigger_ticks) == (0,)
    assert times_us(result.time_base, reads[1].trigger_ticks) == pytest.approx(
        (43 / 430.08,)
    )


def test_reps_varying_body_end_is_an_observable_carry_error():
    def initialize(p):
        p.add_reg("carry", init=QickRawParam("carry", 100, {}))

    def body(p):
        p.pulse(0, "p")
        p.trigger([0], t=0)
        p.inc_reg("carry", 1)
        p.asm_inst(
            {"CMD": "TIME", "C_OP": "inc_ref", "R1": p.reg_dict["carry"].full_addr()}
        )

    prog, env = program(body, initialize, trigger_time=None)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)


def test_conditional_register_live_in_is_rejected():
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("carry", init=QickRawParam("carry", 10, {}))

    def body(p):
        p.write_reg("w_gain", "scan")
        p.asm_inst({"CMD": "TEST", "OP": "w3 - #1", "UF": "1"})
        p.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": p.reg_dict["carry"].full_addr(),
                "SRC": "imm",
                "LIT": "#20",
                "IF": "Z",
            }
        )
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "carry")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize(
    "effect,instruction",
    [("read", "DportReadInst"), ("clock", "RegWriteInst"), ("jump", "JumpInst")],
)
def test_cleanup_effects_are_validated_before_request(effect, instruction):
    def cleanup(p):
        if effect == "read":
            p.read_input(0)
        elif effect == "clock":
            p.write_reg("w_gain", "s_usr_time")
        else:
            p.jump("NEXT")

    prog, env = program(lambda p: p.pulse(0, "p"), reps=2, cleanup=cleanup)
    with pytest.raises(ValueError, match=rf"{instruction}.*preopt"):
        request(prog, env)


@pytest.mark.parametrize(
    "damage",
    [
        "missing-body",
        "missing-loop",
        "unclosed-source",
        "wrong-source",
        "unknown",
        "unbalanced-loop",
    ],
)
def test_malformed_compiled_markers_fail_before_request(damage):
    def transform(snapshot):
        records = []
        for record in snapshot.records:
            inst = record.instruction
            if isinstance(inst, MetaInst):
                if damage == "missing-body" and inst.type == "TIMING_BOUNDARY":
                    continue
                if damage == "missing-loop" and inst.type.startswith("LOOP_"):
                    continue
                if damage == "unclosed-source" and inst.type == "SOURCE_END":
                    continue
                if damage == "wrong-source" and inst.type == "SOURCE_END":
                    inst = MetaInst(type="SOURCE_END", name="wrong")
                if damage == "unknown" and inst.type == "SOURCE_START":
                    inst = MetaInst(type="UNKNOWN", name="unknown")
                if damage == "unbalanced-loop" and inst.type == "LOOP_BODY_END":
                    inst = MetaInst(type="LOOP_END", name=inst.name, info=inst.info)
            records.append(CompiledRecord(inst, record.p_addr))
        return CompiledStream(tuple(records))

    prog, env = program(lambda p: p.pulse(0, "p"), reps=1, stream_transform=transform)
    with pytest.raises(ValueError, match="MetaInst.*preopt"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize("source", ["imm", "op"])
def test_dmem_lane_addresses_and_simultaneous_side_writes_use_prewrite_sources(source):
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("address")
        p.add_reg("side")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("address", "scan")
        p.inc_reg("address", 17)
        address, side = (p.reg_dict[n].full_addr() for n in ("address", "side"))
        p.asm_inst(
            {
                "CMD": "DMEM_WR",
                "DST": f"[{address}]",
                "SRC": source,
                "LIT": "#1234",
                "OP": f"{address} + #100",
                "WR": f"{side} {source}",
                "UF": "1",
            }
        )
        p.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": "w3",
                "SRC": "dmem",
                "ADDR": address,
                "OP": f"{address} - #18",
                "WR": f"{address} op",
                "UF": "1",
            }
        )
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})
        p.write_reg("w_gain", "address")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@50"})
        p.write_reg("w_gain", "side")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@100"})

    prog, env = program(body, initialize, reps=1)
    result = request(prog, env)
    stored = [1234, 1234, 1234] if source == "imm" else [117, 118, 119]
    for point in range(3):
        assert [
            e.gain for e in result.materialize_point(point).events
        ] == pytest.approx(np.array([stored[point], point - 1, stored[point]]) / 32766)


def test_dmem_live_in_is_rejected_across_points():
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("carry")
        p.write_dmem(17, 10)

    def body(p):
        p.read_dmem("carry", 17)
        p.inc_reg("carry", 1)
        p.write_dmem(17, "carry")
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "carry")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize("source", ["native", "module", "asm"])
def test_equivalent_output_effects_choose_the_same_per_read_physical_route(source):
    module = PulseCfg(
        ch=1, nqz=1, freq=200, gain=0.1, waveform=ConstWaveformCfg(length=0.1)
    ).build("routed")

    def initialize(p):
        p.declare(module)
        p.add_pulse(1, "other", style="const", freq=200, phase=0, gain=0.1, length=0.1)

    def body(p):
        if source == "native":
            p.pulse(1, "other", t=0)
        elif source == "module":
            p.emit(module)
        else:
            address = p.wave2idx[p.pulses["other"].get_wavenames()[0]]
            p.asm_inst(
                {
                    "CMD": "WPORT_WR",
                    "DST": "1",
                    "SRC": "wmem",
                    "ADDR": f"&{address}",
                    "TIME": "@0",
                }
            )
        p.trigger([0], t=0)
        p.pulse(0, "p", t=0.3)
        p.trigger([0], t=0.3)

    prog, env = program(
        body, initialize, reps=1, routes=((0, 0), (1, 0)), trigger_time=None
    )
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.generator_channels == (1, 0)
    assert read.frequency_mhz == pytest.approx((100, 100), abs=1e-5)


def test_register_selected_wmem_addresses_and_side_writes_sample_prewrite_words():
    def initialize(p):
        p.add_loop("scan", 3)
        p.add_reg("address")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("address", "scan")
        p.inc_reg("address", 17)
        p.write_reg("w_gain", "scan")
        p.inc_reg("w_gain", 100)
        address = p.reg_dict["address"].full_addr()
        p.asm_inst({"CMD": "WMEM_WR", "DST": address, "OP": "w3 + #1", "WR": "w3 op"})
        p.write_reg("s14", 0)
        p.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": "r_wave",
                "SRC": "wmem",
                "ADDR": address,
                "WP": "r_wave",
                "PORT": "0",
            }
        )
        p.write_reg("s14", 50)
        p.asm_inst(
            {
                "CMD": "WPORT_WR",
                "DST": "0",
                "SRC": "wmem",
                "ADDR": address,
                "OP": f"{address} + #1",
                "WR": f"{address} op",
            }
        )
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@100"})

    prog, env = program(body, initialize, reps=1)
    result = request(prog, env)
    for point in range(3):
        assert [
            e.gain for e in result.materialize_point(point).events
        ] == pytest.approx(np.array([101 + point, 100 + point, 100 + point]) / 32766)


def test_unused_register_live_in_is_rejected_across_reps():
    def initialize(p):
        p.add_reg("unused", init=QickRawParam("unused", 0, {}))

    def body(p):
        p.inc_reg("unused", 1)
        p.pulse(0, "p")

    prog, env = program(body, initialize, reps=5)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)
    assert env.soc.binding.last_request is None
