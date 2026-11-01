"""Wave-port routing through native compilation and detached requests."""

import pytest
from qick import QickConfig
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import profile_soccfg
from tests.program.v2.simulation._compiled_support import request


def port_program(soccfg: QickConfig) -> TimelineProgramV2:
    """Compile one carrier and two independently configured ADC windows."""

    class Program(TimelineProgramV2):
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


def test_shared_wave_port_routes_carrier_and_each_readout_config():
    soccfg = profile_soccfg("tmux_tport")
    # Isolate wave dispatch from TRIG support in this first acceptance slice.
    for ch, readout in enumerate(soccfg["readouts"]):
        readout.update(trigger_type="dport", trigger_port=3, trigger_bit=ch)
    env = white_noise(soccfg, std_adc=0, seed=None, routes=((0, 0), (0, 1)))
    try:
        result = request(port_program(soccfg), env)
        point = result.materialize_point(0)
        (pulse,) = point.events
        assert pulse.gen_ch == 0
        assert pulse.frequency_mhz == pytest.approx(80, abs=1e-5)
        assert [read.frequency_mhz for read in point.readouts] == [
            pytest.approx((80, 80), abs=1e-5),
            pytest.approx((120, 120), abs=1e-5),
        ]
        assert all(read.gen_ch == 0 for read in point.readouts)
    finally:
        env.devices.close_all_devices()


def test_tport_set_and_clear_preserve_dport_ticks_and_declared_read_counts():
    points = []
    for name in ("separate_dport", "tmux_tport"):
        soccfg = profile_soccfg(name)
        env = white_noise(soccfg, std_adc=0, seed=None, routes=((0, 0), (0, 1)))
        try:
            result = request(port_program(soccfg), env)
            points.append(result.materialize_point(0))
        finally:
            env.devices.close_all_devices()
    dport, tport = points
    for expected, actual in zip(dport.readouts, tport.readouts, strict=True):
        assert actual.trigger_ticks == expected.trigger_ticks
        # QICK emits clr after each set; only the two sets count as reads.
        assert len(actual.trigger_ticks) == 2
        assert actual.frequency_mhz == expected.frequency_mhz
