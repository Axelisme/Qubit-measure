"""Wave-port routing through native compilation and detached requests."""

from collections.abc import Callable
from dataclasses import replace

import pytest
from qick import QickConfig
from zcu_tools.core.simulation import AcquireRequest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.ir import CompiledRecord, CompiledStream
from zcu_tools.program.v2.ir.instructions import RegWriteInst
from zcu_tools.program.v2.ir.io_instructions import TrigInst
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import profile_soccfg
from tests.program.v2.simulation._compiled_support import request


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
