"""Compile native and named scheduling through the public program interface."""

from collections.abc import Callable

import numpy as np
import pytest
from qick.asm_v2 import AveragerProgramV2, QickSweep1D
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _program(
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    name=f"p{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=0.1,
                )
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(
        make_mock_soccfg(2, 1),
        reps=1,
        initial_delay=0.0,
        relax_delay=0.0,
    )


def _pulse_ticks(prog: AveragerProgramV2) -> list[int]:
    return [
        int(str(inst["TIME"])[1:])
        for inst in prog.prog_list
        if inst["CMD"] == "WPORT_WR"
    ]


def test_named_delay_places_pulses_without_advancing_reference():
    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")
        prog.pulse(1, "p1", timeline="drive")

    prog = _program(body)
    # ceil(0.1 * 430.08) = 44. Emission itself does not move the cursor.
    assert _pulse_ticks(prog) == [44, 44]
    # The shared shot shell advances through the emitted pulses at body end.
    assert [i["LIT"] for i in prog.prog_list if i.get("C_OP") == "inc_ref"] == ["#87"]


@pytest.mark.parametrize("delay, expected", [(0.3, 173), (0.5, 215)])
def test_native_delay_decrements_named_cursor_with_reference_floor(delay, expected):
    def body(prog: TimelineProgramV2):
        prog.delay(0.4, timeline="drive")
        prog.delay(delay)
        prog.pulse(0, "p0", timeline="drive")

    prog = _program(body)
    # TimedMergePass folds the reference advance into the pulse's absolute time.
    assert _pulse_ticks(prog) == [expected]


def test_recompile_resets_named_cursors():
    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")

    prog = _program(body)
    original = prog.prog_list.copy()
    prog.compile()
    assert prog.prog_list == original
    assert _pulse_ticks(prog) == [44]


def test_named_delay_auto_aligns_to_own_emissions_not_long_flux():
    def body(prog: TimelineProgramV2):
        prog.pulse(1, "p1", t=1.0)  # default timeline flux ends at 1.1 us
        prog.pulse(0, "p0", timeline="drive")
        prog.delay_auto(0.05, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")

    # drive ends at 38/384 us; ceil((38/384 + .05)*430.08) = 65.
    assert _pulse_ticks(_program(body)) == [430, 0, 65]


@pytest.mark.parametrize("length", [None, 0.8])
def test_named_config_is_instantaneous_and_not_another_trigger_window(length):
    def initialize(prog: TimelineProgramV2):
        prog.declare_readout(0, length=0.4)
        if length is None:
            prog.add_readoutconfig(0, "ro", freq=100)
        else:
            prog.add_readoutconfig(0, "ro", freq=100, length=length)

    def body(prog: TimelineProgramV2):
        prog.trigger(ros=[0], t=1.0)
        prog.delay(0.1, timeline="drive")
        prog.send_readoutconfig(0, "ro", t=0.05, timeline="drive")
        prog.delay_auto(0.05, gens=False, ros=True, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")

    # Config is at 44 + .05*430.08 -> 66 cycles. Its end is that
    # instantaneous time, not the unrelated ADC window or config length.
    assert _pulse_ticks(_program(body, initialize)) == [66, 88]


@pytest.mark.parametrize(
    "gens, ros, expected", [(True, False, 87), (False, True, 217), (False, False, 53)]
)
def test_named_trigger_and_generator_ends_are_selected_and_survive_advance(
    gens, ros, expected
):
    def initialize(prog: TimelineProgramV2):
        prog.declare_readout(0, length=0.4)

    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")
        prog.trigger(ros=[0], timeline="drive")
        prog.delay(0.02, timeline="drive")  # advance must not discard earlier ends
        prog.delay_auto(gens=gens, ros=ros, timeline="drive")
        prog.pulse(1, "p1", timeline="drive")

    prog = _program(body, initialize)
    assert _pulse_ticks(prog) == [44, expected]
    assert [i["TIME"] for i in prog.prog_list if i["CMD"] == "DPORT_WR"] == [
        "@44",
        "@54",
    ]


@pytest.mark.parametrize("offset, expected", [(0.05, 43), (0.4, 173)])
def test_named_pulse_auto_uses_later_cursor_or_generator_end(offset, expected):
    def body(prog: TimelineProgramV2):
        prog.pulse(0, "p0")
        prog.delay(offset, timeline="drive")
        prog.pulse(0, "p0", t="auto", timeline="drive")

    assert _pulse_ticks(_program(body)) == [0, expected]


def test_sync_rebases_all_named_cursors_with_one_reference_increment():
    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.delay(0.4, timeline="flux")
        prog.sync(timeline="drive")
        prog.pulse(0, "p0", timeline="drive", tag="drive")
        prog.pulse(1, "p1", timeline="flux", tag="flux")

    prog = _program(body)
    assert _pulse_ticks(prog) == [44, 173]
    assert prog.get_time_param("drive", "t") == pytest.approx(0.0)
    assert prog.get_time_param("flux", "t") == pytest.approx(129 / 430.08)


def test_named_trigger_rejects_implicit_output_time_at_entry():
    def body(prog: TimelineProgramV2):
        with pytest.raises(ValueError, match="t=None"):
            prog.trigger(t=None, timeline="drive")

    _program(body)


def test_named_auto_rounds_only_final_offset_to_114_cycles():
    def initialize(prog: TimelineProgramV2):
        prog.declare_readout(0, length=0.4)
        prog.add_readoutconfig(0, "ro", freq=100)

    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.send_readoutconfig(0, "ro", t=0.2139 - 44 / 430.08, timeline="drive")
        prog.delay_auto(0.05, timeline="drive")
        prog.pulse(0, "p0", timeline="drive", tag="end")

    # max(offset, .2139)+.05 = .2639 us, 113.50 cycles -> 114.
    prog = _program(body, initialize)
    assert _pulse_ticks(prog) == [92, 114]
    assert prog.get_time_param("end", "t") == pytest.approx(114 / 430.08)


def test_named_auto_keeps_cursor_when_later_than_selected_ends():
    def body(prog: TimelineProgramV2):
        prog.pulse(0, "p0", timeline="drive")
        prog.delay(0.4, timeline="drive")
        prog.delay_auto(0.05, timeline="drive")
        prog.pulse(1, "p1", timeline="drive")

    # ceil(.4*430.08)=173, then ceil(173+.05*430.08)=195.
    assert _pulse_ticks(_program(body)) == [0, 195]


def test_crossing_sweeps_use_native_qick_maximum_not_elementwise_max():
    def initialize(prog: TimelineProgramV2):
        prog.add_loop("scan", 3)
        prog.declare_readout(0, length=0.4)
        prog.add_readoutconfig(0, "ro", freq=100)

    def body(prog: TimelineProgramV2):
        prog.send_readoutconfig(
            0, "ro", t=QickSweep1D("scan", 0.2, 0.1), timeline="drive"
        )
        prog.delay(QickSweep1D("scan", 0.1, 0.3), timeline="drive")
        prog.delay_auto(timeline="drive")
        prog.pulse(0, "p0", timeline="drive", tag="out")

    prog = _program(body, initialize)
    # QICK range comparison keeps the first max candidate (cursor) for these
    # crossing ranges. Ceil start/span -> 44/87, native stepping -> 43 per point.
    ticks = np.asarray(prog.get_time_param("out", "t", as_array=True)) * 430.08
    np.testing.assert_allclose(ticks, [44, 87, 130], atol=1e-10)


@pytest.mark.parametrize("operation", ["delay", "delay_auto", "resync"])
def test_native_reference_changes_rebase_named_ends_and_cursor(operation):
    def body(prog: TimelineProgramV2):
        prog.delay(0.1, timeline="drive")
        prog.pulse(0, "p0", timeline="drive")
        if operation == "delay":
            prog.delay(0.05)
        elif operation == "delay_auto":
            prog.delay_auto(0.05)
        else:
            prog.resync(0.05)
        prog.delay_auto(timeline="drive")
        prog.pulse(1, "p1", timeline="drive", tag="after")

    # Native delay/resync subtract round(.05*430.08)=22 cycles from
    # end=44+38/384*430.08=86.56; ceil(64.56)=65. Global auto also
    # consumes the pulse end and clamps both named values to zero.
    expected = 0 if operation == "delay_auto" else 65
    prog = _program(body)
    assert prog.get_time_param("after", "t") == pytest.approx(expected / 430.08)


@pytest.mark.parametrize("operation", ["wait_auto", "resync"])
def test_wait_and_resync_reject_timeline_but_allow_native_after_named_use(operation):
    def body(prog):
        prog.pulse(0, "p0", timeline="drive")
        if operation == "wait_auto":
            with pytest.raises(TypeError, match="timeline"):
                prog.wait_auto(timeline="drive")
            prog.wait_auto()
        else:
            with pytest.raises(TypeError, match="timeline"):
                prog.resync(timeline="drive")
            prog.resync()

    _program(body)


def test_long_default_flux_overlaps_named_drive_and_readout():
    def initialize(prog: TimelineProgramV2):
        prog.add_pulse(
            1, "flux", style="const", freq=1000, phase=0, gain=0.5, length=1.0
        )
        prog.declare_readout(0, length=0.4)

    def body(prog: TimelineProgramV2):
        prog.pulse(1, "flux")
        prog.pulse(0, "p0", timeline="drive")
        prog.delay_auto(timeline="drive")
        prog.trigger(ros=[0], timeline="drive")

    prog = _program(body, initialize)
    assert _pulse_ticks(prog) == [0, 0]
    assert [i["TIME"] for i in prog.prog_list if i["CMD"] == "DPORT_WR"] == [
        "@43",
        "@53",
    ]
    # Readout starts at cycle 43 while the 1-us flux lasts 430.08 cycles.


def test_swept_sync_leaves_no_residual_time_on_its_own_cursor():
    def initialize(prog: TimelineProgramV2):
        prog.add_loop("scan", 3)

    def body(prog: TimelineProgramV2):
        prog.delay(QickSweep1D("scan", 0.1, 0.3), timeline="drive")
        prog.sync(timeline="drive")
        prog.pulse(0, "p0", timeline="drive", tag="after")

    prog = _program(body, initialize)
    np.testing.assert_allclose(
        prog.get_time_param("after", "t", as_array=True), 0.0, atol=1e-12
    )


def test_named_swept_advances_emit_no_instructions_or_sweep_registers():
    def initialize(prog: TimelineProgramV2):
        prog.add_loop("scan", 3)

    def body(prog: TimelineProgramV2):
        prog.delay(QickSweep1D("scan", 0.1, 0.3), timeline="drive")
        prog.delay_auto(0.05, timeline="drive")

    advanced = _program(body, initialize)
    empty = _program(lambda prog: None, initialize)
    assert advanced.prog_list == empty.prog_list
    assert advanced.labels == empty.labels


def test_native_auto_without_channels_does_not_move_named_cursor():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            pass

        def _body(self, cfg):
            self.delay(0.1, timeline="drive")
            self.delay_auto()
            self.sync(timeline="drive")

    prog = Program(
        make_mock_soccfg(0, 0),
        reps=1,
        initial_delay=0.0,
        relax_delay=0.0,
    )
    assert [i["LIT"] for i in prog.prog_list if i.get("C_OP") == "inc_ref"] == ["#44"]
