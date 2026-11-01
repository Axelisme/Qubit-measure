"""Wave-port routing through native compilation and detached requests."""

from collections.abc import Callable
from dataclasses import replace

import pytest
from qick import QickConfig
from zcu_tools.core.simulation import AcquireRequest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.ir import CompiledRecord, CompiledStream
from zcu_tools.program.v2.ir.instructions import RegWriteInst
from zcu_tools.program.v2.ir.io_instructions import DportWriteInst, TrigInst
from zcu_tools.program.v2.ir.operands import Immediate, ImmValue, Register, SrcKeyword
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import profile_soccfg
from tests.program.v2.simulation._compiled_support import program, request


def port_program(
    soccfg: QickConfig,
    transform: Callable[[CompiledStream], CompiledStream] | None = None,
) -> TimelineProgramV2:
    """Compile one carrier and two independently configured ADC windows."""

    class Program(TimelineProgramV2):
        @property
        def compiled_stream(self):
            stream = super().compiled_stream
            return stream if transform is None else transform(stream)

        def _initialize(self, cfg):
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "carrier", style="const", freq=80, phase=0, gain=0.3, length=1
            )
            for ch, freq in enumerate((80, 120)):
                self.declare_readout(ch, length=0.2)
                self.add_readoutconfig(ch, f"config{ch}", freq=freq, gen_ch=0)

        def _body(self, cfg):
            for ch in self.ro_chs:
                self.send_readoutconfig(ch, f"config{ch}", t=0)
            self.pulse(0, "carrier", t=0.1)
            self.trigger([0, 1], t=0.1)
            self.trigger([0, 1], t=0.5)

    return Program(soccfg, reps=2, initial_delay=0, relax_delay=0.1)


def build_port_request(
    soccfg: QickConfig,
    transform: Callable[[CompiledStream], CompiledStream] | None = None,
) -> AcquireRequest:
    """Build a request and release the synthetic environment on all paths."""
    env = white_noise(soccfg, std_adc=0, seed=None, routes=((0, 0), (0, 1)))
    try:
        return request(port_program(soccfg, transform), env)
    finally:
        env.devices.close_all_devices()


def test_shared_wave_port_routes_carrier_and_each_readout_config():
    soccfg = profile_soccfg("tmux_tport")
    # Isolate wave dispatch from TRIG support in this first acceptance slice.
    for ch, readout in enumerate(soccfg["readouts"]):
        readout.update(trigger_type="dport", trigger_port=3, trigger_bit=ch)
    point = build_port_request(soccfg).materialize_point(0)
    (pulse,) = point.events
    assert pulse.gen_ch == 0
    assert pulse.frequency_mhz == pytest.approx(80, abs=1e-5)
    assert [read.frequency_mhz for read in point.readouts] == [
        pytest.approx((80, 80), abs=1e-5),
        pytest.approx((120, 120), abs=1e-5),
    ]
    assert all(read.gen_ch == 0 for read in point.readouts)


def test_tport_set_and_clear_preserve_dport_ticks_and_declared_read_counts():
    points = []
    for name in ("separate_dport", "tmux_tport"):
        soccfg = profile_soccfg(name)
        points.append(build_port_request(soccfg).materialize_point(0))
    dport, tport = points
    for expected, actual in zip(dport.readouts, tport.readouts, strict=True):
        assert actual.trigger_ticks == expected.trigger_ticks
        # QICK emits clr after each set; only the two sets count as reads.
        assert len(actual.trigger_ticks) == 2
        assert actual.frequency_mhz == expected.frequency_mhz


def test_explicit_trig_time_does_not_read_cross_shot_s14():
    soccfg = profile_soccfg("tmux_tport")

    def explicit_times(stream: CompiledStream) -> CompiledStream:
        records = []
        for record in stream.records:
            records.append(record)
            if isinstance(record.instruction, TrigInst):
                assert record.instruction.time is not None
                records.append(
                    CompiledRecord(
                        RegWriteInst(
                            dst=Register("s14"),
                            src=SrcKeyword.IMM,
                            lit=Immediate(999),
                        ),
                        record.p_addr,
                    )
                )
        return CompiledStream(tuple(records))

    native = build_port_request(soccfg).materialize_point(0)
    explicit = build_port_request(soccfg, explicit_times).materialize_point(0)
    assert [read.trigger_ticks for read in explicit.readouts] == [
        read.trigger_ticks for read in native.readouts
    ]


@pytest.mark.parametrize("tmux", [(None, None), (7, None), (None, 11), (7, 7)])
def test_colliding_wave_users_are_rejected(tmux):
    soccfg = profile_soccfg("separate_dport")
    gen = soccfg["gens"][0]
    readout = soccfg["readouts"][0]
    gen["tmux_ch"], readout["tmux_ch"] = tmux
    readout["tproc_ctrl"] = gen["tproc_ch"]
    with pytest.raises(ValueError, match="ambiguous wave output port"):
        build_port_request(soccfg)


def test_unregistered_wave_tmux_reports_port_and_encoded_channel():
    soccfg = profile_soccfg("tmux_tport")
    prog = port_program(soccfg)
    port, encoded_tmux = (soccfg["gens"][0][key] for key in ("tproc_ch", "tmux_ch"))
    # Keep the compiled word but withdraw its destination from the input board.
    soccfg["gens"][0]["tmux_ch"] = encoded_tmux + 1
    env = white_noise(soccfg, std_adc=0, seed=None, routes=((0, 0), (0, 1)))
    try:
        with pytest.raises(
            ValueError, match=rf"port {port}, tmux_ch={encoded_tmux}"
        ):
            request(prog, env)
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("profile", ["tmux_tport", "separate_dport"])
def test_each_trigger_instruction_only_reads_its_own_port_type(profile):
    soccfg = profile_soccfg(profile)

    def extra_triggers(stream: CompiledStream) -> CompiledStream:
        records = []
        for record in stream.records:
            inst = record.instruction
            records.append(record)
            if isinstance(inst, TrigInst):
                other = DportWriteInst(
                    dst=inst.dst, src="imm", data=ImmValue(3), time=inst.time
                )
            elif isinstance(inst, DportWriteInst):
                other = TrigInst(dst=inst.dst, src="set", time=inst.time)
            else:
                continue
            records.append(CompiledRecord(other, record.p_addr))
        return CompiledStream(tuple(records))

    native = build_port_request(soccfg).materialize_point(0)
    extra = build_port_request(soccfg, extra_triggers).materialize_point(0)
    assert [read.trigger_ticks for read in extra.readouts] == [
        read.trigger_ticks for read in native.readouts
    ]


def test_lane_dependent_wave_tmux_is_rejected():
    soccfg = profile_soccfg("tmux_tport")

    def initialize(p):
        p.add_loop("scan", 2)

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg_op("w_conf", "scan", "SL", 8)
        p.inc_reg("w_conf", (soccfg["gens"][0]["tmux_ch"] << 8) | 9)
        p.asm_inst(
            {
                "CMD": "WPORT_WR",
                "DST": str(soccfg["gens"][0]["tproc_ch"]),
                "SRC": "r_wave",
                "TIME": "@0",
            }
        )

    prog, env = program(body, initialize, soccfg=soccfg)
    try:
        with pytest.raises(ValueError, match="lane-dependent tmux channel"):
            request(prog, env)
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("src", ["set", "clr"])
def test_lane_dependent_trigger_port_is_rejected(src):
    soccfg = profile_soccfg("tmux_tport")

    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("port", addr=9)

    def body(p):
        p.write_reg_op("port", "scan", "+", soccfg["readouts"][0]["trigger_port"])
        p.pulse(0, "p")

    def transform(stream):
        return CompiledStream(
            tuple(
                CompiledRecord(
                    replace(record.instruction, dst=Register("r9"), src=src)
                    if isinstance(record.instruction, TrigInst)
                    else record.instruction,
                    record.p_addr,
                )
                for record in stream.records
            )
        )

    prog, env = program(body, initialize, soccfg=soccfg, stream_transform=transform)
    try:
        with pytest.raises(ValueError, match="lane-dependent trigger port"):
            request(prog, env)
    finally:
        env.devices.close_all_devices()
