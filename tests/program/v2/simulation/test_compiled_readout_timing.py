"""Runtime readout timing through public native compile/request seams."""

import pytest
from qick.asm_v2 import QickSweep1D

from tests.program.v2.simulation._compiled_support import program, request


def send_config(p, name, time, phase=0):
    """Emit runtime config words, including phase effects absent from mock declarations."""
    if phase == 0:
        p.send_readoutconfig(0, name, t=time)
        return
    wave = p.pulses[name].get_wavenames()[0]
    p.read_wmem(wave)
    p.write_reg("w_phase", phase * 2**32 // 360)
    port = p.soccfg["readouts"][0]["tproc_ctrl"]
    ticks = p.us2cycles(time)
    p.asm_inst(
        {"CMD": "WPORT_WR", "DST": str(port), "SRC": "r_wave", "TIME": f"@{ticks}"}
    )


@pytest.mark.parametrize("read_time, expected", [(0.75, 100), (1.0, 300)])
def test_readout_config_fifo_serializes_reverse_timestamps(read_time, expected):
    def initialize(p):
        p.add_readoutconfig(0, "first", freq=200, gen_ch=0)
        p.add_readoutconfig(0, "second", freq=300, gen_ch=0)

    def body(p):
        p.send_readoutconfig(0, "first", t=1.0)
        p.send_readoutconfig(0, "second", t=0.5)
        p.pulse(0, "p", t=read_time)

    prog, env = program(body, initialize, reps=1, trigger_time=read_time)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((expected,), abs=1e-5)


@pytest.mark.parametrize("frequency, phase", [(200, 0), (100, 90)])
@pytest.mark.parametrize("after_trigger", [False, True])
def test_changed_config_inside_window_is_rejected(frequency, phase, after_trigger):
    def initialize(p):
        p.add_readoutconfig(0, "changed", freq=frequency, gen_ch=0)

    def body(p):
        p.pulse(0, "p", t=0)
        if after_trigger:
            p.trigger([0], t=0)
        send_config(p, "changed", 0.1, phase)
        if not after_trigger:
            p.trigger([0], t=0)

    prog, env = program(body, initialize, reps=1, trigger_time=None)
    with pytest.raises(ValueError, match=r"ADC 0 read 0 window") as error:
        request(prog, env)
    message = str(error.value)
    for detail in (
        "ADC 0",
        "read 0",
        "window",
        "effective",
        "frequency",
        "phase",
        "preopt",
        "PortWriteInst",
    ):
        assert detail in message
    assert "initial=(" in message
    assert "updated=(" in message
    if phase:
        assert "phase=90" in message


@pytest.mark.parametrize("phase", [0, 90])
def test_equal_config_inside_window_is_accepted(phase):
    def body(p):
        send_config(p, "ro", 0, phase)
        p.pulse(0, "p", t=0)
        p.trigger([0], t=0)
        send_config(p, "ro", 0.1, phase)

    prog, env = program(body, reps=1, trigger_time=None, config_time=None)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((100,), abs=1e-5)
    assert read.phase_deg == (phase,)


@pytest.mark.parametrize("update_time, expected", [(0, 200), (60 / 307.2, 100)])
def test_config_at_window_boundary_uses_closed_start_and_open_end(
    update_time, expected
):
    def initialize(p):
        p.add_readoutconfig(0, "changed", freq=200, gen_ch=0)

    def body(p):
        send_config(p, "changed", update_time, 90)
        p.pulse(0, "p", t=0)

    prog, env = program(body, initialize, reps=1, readout_length=60 / 307.2)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((expected,), abs=1e-5)
    assert read.phase_deg == ((90,) if update_time == 0 else (0,))


@pytest.mark.parametrize("config_time", [None, 1.0])
def test_runtime_read_without_effective_config_at_start_is_rejected(config_time):
    prog, env = program(lambda p: p.pulse(0, "p", t=0), reps=1, config_time=config_time)
    with pytest.raises(ValueError, match=r"ADC 0.*read 0.*no.*config.*start"):
        request(prog, env)


def carried_config_program(*, phase=0, frequency=200, points=1, reps=1):
    def initialize(p):
        p.add_loop("scan", points)
        p.add_readoutconfig(0, "carried", freq=frequency, gen_ch=0)
        p.send_readoutconfig(0, "ro", t=0)

    def body(p):
        p.pulse(0, "p", t=0)
        p.trigger([0], t=0)
        send_config(p, "carried", 0.25, phase)

    return program(body, initialize, reps=reps, trigger_time=None, config_time=None)


@pytest.mark.parametrize("frequency, phase", [(200, 0), (100, 90)])
def test_runtime_config_live_in_is_rejected_across_scan_points(frequency, phase):
    prog, env = carried_config_program(frequency=frequency, phase=phase, points=3)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*config"):
        request(prog, env)


@pytest.mark.parametrize("frequency, phase", [(200, 0), (100, 90)])
def test_rep_carried_config_changing_readout_is_rejected(frequency, phase):
    prog, env = carried_config_program(frequency=frequency, phase=phase, reps=2)
    with pytest.raises(ValueError, match=r"preopt.*cross-shot.*config"):
        request(prog, env)


def test_delayed_equal_config_cannot_supply_a_previous_points_start():
    def initialize(p):
        p.add_loop("scan", 2)

    def body(p):
        p.send_readoutconfig(0, "ro", t=QickSweep1D("scan", 0, 0.1))
        p.pulse(0, "p", t=0)

    prog, env = program(body, initialize, reps=1, config_time=None)
    with pytest.raises(ValueError, match="ADC 0.*no.*config.*start"):
        request(prog, env)


def test_equal_rep_carried_config_is_rejected():
    prog, env = carried_config_program(frequency=100, reps=3, points=2)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*config"):
        request(prog, env)


def test_pending_config_fifo_carries_into_later_rep_window():
    def initialize(p):
        p.add_readoutconfig(0, "future", freq=200, gen_ch=0)

    def body(p):
        p.send_readoutconfig(0, "future", t=1.0)
        p.pulse(0, "p", t=0)

    prog, env = program(body, initialize, reps=4)
    with pytest.raises(ValueError, match=r"preopt.*cross-shot.*pending config FIFO"):
        request(prog, env)


def test_each_read_uses_its_own_start_config_in_trigger_order():
    def initialize(p):
        p.add_readoutconfig(0, "second", freq=200, gen_ch=0)

    def body(p):
        p.pulse(0, "p", t=0)
        p.pulse(0, "p", t=0.5)
        p.send_readoutconfig(0, "second", t=0.5)
        p.trigger([0], t=0)
        p.trigger([0], t=0.5)

    prog, env = program(body, initialize, reps=1, trigger_time=None)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((100, 200), abs=1e-5)


def test_readout_ports_keep_independent_config_fifos():
    def initialize(p):
        p.declare_readout(1, length=0.2)
        p.add_readoutconfig(1, "other", freq=300, gen_ch=0)
        p.add_readoutconfig(0, "future", freq=200, gen_ch=0)

    def body(p):
        p.send_readoutconfig(0, "future", t=1.0)
        p.send_readoutconfig(1, "other", t=0)
        p.pulse(0, "p", t=0)
        p.trigger([0, 1], t=0)

    prog, env = program(
        body,
        initialize,
        reps=1,
        trigger_time=None,
        routes=((0, 0), (0, 1)),
        n_readouts=2,
    )
    reads = request(prog, env).materialize_point(0).readouts
    assert [r.channel for r in reads] == [0, 1]
    assert [r.frequency_mhz[0] for r in reads] == pytest.approx([100, 300], abs=1e-5)


def test_equal_config_words_after_wmem_roundtrip_are_accepted():
    def body(p):
        p.pulse(0, "p", t=0)
        wave = p.pulses["ro"].get_wavenames()[0]
        p.read_wmem(wave)
        p.write_wmem(wave)
        p.send_readoutconfig(0, "ro", t=0.1)

    prog, env = program(body, reps=1)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((100,), abs=1e-5)


def test_future_config_does_not_replace_config_at_read_start():
    def initialize(p):
        p.add_readoutconfig(0, "future", freq=200, gen_ch=0)

    def body(p):
        p.send_readoutconfig(0, "future", t=1.0)
        p.pulse(0, "p", t=0)

    prog, env = program(body, initialize, reps=1)
    read = request(prog, env).materialize_point(0).readouts[0]
    assert read.frequency_mhz == pytest.approx((100,), abs=1e-5)
