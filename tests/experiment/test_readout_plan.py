"""Readout classification and waveform contents at the shared planning seam."""

from tests.core._simulation import TIME_BASE, tick_array, ticks
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import EventSlot, ReadoutSlot, TimelineTemplate
from zcu_tools.experiment import readout_plan


def pulse(start_us: float, duration_us: float = 0.004) -> EventSlot:
    """One active calibrated-generator pulse at two points."""
    return EventSlot(
        0,
        "synthetic",
        (),
        tick_array(np.full(2, start_us)),
        tick_array(np.full(2, duration_us)),
        np.full(2, 6000.0),
        np.full(2, 30.0),
        np.full(2, -0.2),
        (None,),
        np.zeros(2, dtype=np.int64),
        np.ones(2, dtype=bool),
        ticks(0.001),
    )


def template(
    events: tuple[EventSlot, ...], triggers=(0.011, 0.031)
) -> TimelineTemplate:
    """Two points and one receiver with independent per-read demodulation."""
    shape = (2, len(triggers))
    window = ReadoutSlot(
        2,
        3,
        TIME_BASE.period_ticks(1000.0),
        tick_array(np.tile(triggers, (2, 1))),
        np.full(shape, 5900.0),
        np.full(shape, 45.0),
        np.zeros(shape, dtype=np.int64),
    )
    return TimelineTemplate(
        np.array([0, 1], dtype=np.int64), events, (window,), tick_array(np.full(2, 0.050))
    )


def test_logical_contents_keep_unobserved_terminal_and_group_repeated_triggers() -> (
    None
):
    base = template((pulse(0.010), pulse(0.030)), (0.011, 0.015))
    contents = readout_plan.logical_event_contents(
        base, readout_plan.ReadoutDrive(0, 2.0), base.readouts[0], {}
    , time_base=TIME_BASE)
    assert len(contents) == 2
    for point in range(2):
        herald, terminal = (content.inverse[point] for content in contents)
        assert len(contents[0].grids[herald]) == 2
        assert contents[1].grids[terminal] == ()
        assert contents[1].pulses[terminal][0].start_ns == 30
        # Common-origin phases agree with the original pulse, not a second
        # pulse-local rephase. The program reference is one ns at six GHz.
        assert contents[1].pulses[terminal][0].phase_rad == pytest.approx(
            np.pi / 6 - 2 * np.pi * 6
        )


def test_event_contents_pair_each_trigger_with_its_own_point_local_pulse() -> None:
    first, second, terminal = pulse(0.010), pulse(0.020), pulse(0.030)
    shifts = tick_array([0.0, 0.002])
    events = tuple(
        replace(p, start_ticks=p.start_ticks + shifts) for p in (terminal, first, second)
    )
    base = template(events, (0.011, 0.021, 0.031))
    window = replace(
        base.readouts[0],
        trigger_ticks=base.readouts[0].trigger_ticks + shifts[:, None],
        frequency_mhz=np.tile([5800.0, 5900.0, 6000.0], (2, 1)),
        phase_deg=np.tile([0.0, 45.0, 90.0], (2, 1)),
    )
    contents = readout_plan.read_event_contents(
        replace(base, readouts=(window,)), readout_plan.ReadoutDrive(0, 2.0), window, {}
    , time_base=TIME_BASE)
    assert len(contents) == 3
    for read, content in enumerate(contents):
        np.testing.assert_array_equal(content.end_ns, [10 + 10 * read, 12 + 10 * read])
        for point, index in enumerate(content.inverse):
            pulses, grids = content.pulses[index], content.grids[index]
            assert len(pulses) == len(grids) == 1
            assert pulses[0].start_ns == 10 + 10 * read + 2 * point
            assert pulses[0].phase_rad == pytest.approx(np.pi / 6 - 2 * np.pi * 6)
            np.testing.assert_allclose(
                grids[0].times_ns,
                [
                    11 + 10 * read + 2 * point,
                    12 + 10 * read + 2 * point,
                    13 + 10 * read + 2 * point,
                ],
            )
            assert grids[0].demodulation_frequency_ghz == pytest.approx(
                5.8 + 0.1 * read
            )
            assert grids[0].phase_rad == pytest.approx(np.pi / 4 * read)


def test_partition_separates_triggered_read_from_cavity_drive_and_terminal() -> None:
    terminal, herald, cavity = pulse(0.030), pulse(0.010), pulse(0.0)
    plan = readout_plan.classify_readout(
        template((terminal, cavity, herald)), readout_plan.ReadoutDrive(0, 2.0), {}
    , time_base=TIME_BASE)
    assert [slot.active.tolist() for slot in plan.terminal_slots] == [
        [True, True],
        [False, False],
        [False, False],
    ]
    assert [[p.start_ns for p in point] for point in plan.cavity_drives] == [[0], [0]]
    assert tuple(map(len, plan.projective_reads)) == (1, 1)
    read = plan.projective_reads[0][0]
    assert read.start_ns == 10
    assert len(read.pulses) == 1
    assert read.pulses[0].amplitude == -0.4
    assert read.pulses[0].phase_rad == pytest.approx(np.pi / 6 - 2 * np.pi * 6 * 11)
    trigger = read.triggers[0]
    assert (trigger.channel, trigger.read_index, trigger.trigger_ns) == (2, 0, 11)
    assert trigger.frequency_ghz == 5.9
    assert trigger.phase_rad == pytest.approx(np.pi / 4)


def test_mux_block_projects_once_for_multiple_triggers_and_receivers() -> None:
    herald, terminal = pulse(0.010), pulse(0.030)
    other_tone = replace(herald, frequency_mhz=np.full(2, 6100.0))
    base = template((terminal, herald, other_tone), (0.011, 0.012, 0.031))
    second_receiver = replace(
        base.readouts[0],
        channel=3,
        trigger_ticks=tick_array(np.tile([0.0115, 0.031, 0.032], (2, 1))),
    )
    base = replace(base, readouts=(*base.readouts, second_receiver))
    drive = readout_plan.ReadoutDrive(0, 2.0, "mux")
    plan = readout_plan.classify_readout(base, drive, {}, time_base=TIME_BASE)
    assert tuple(map(len, plan.projective_reads)) == (1, 1)
    read = plan.projective_reads[0][0]
    assert [p.frequency_ghz for p in read.pulses] == [6.0, 6.1]
    assert [(t.channel, t.read_index) for t in read.triggers] == [
        (2, 0),
        (2, 1),
        (3, 0),
    ]
    assert plan.cavity_drives == ((), ())
    contents = readout_plan.read_event_contents(base, drive, base.readouts[0], {}, time_base=TIME_BASE)
    assert [len(c.pulses[0]) for c in contents] == [2, 2, 1]
    assert [c.grids[0][0].times_ns[0] for c in contents] == [11, 12, 31]


def test_partition_uses_active_point_windows_not_slot_order() -> None:
    herald, terminal = pulse(0.010), pulse(0.030)
    alternate = replace(herald, active=np.array([False, True]))
    herald = replace(herald, active=np.array([True, False]))
    base = template((terminal, alternate, herald))
    plan = readout_plan.classify_readout(base, readout_plan.ReadoutDrive(0, 1.0), {}, time_base=TIME_BASE)
    assert tuple(map(len, plan.projective_reads)) == (1, 1)
    assert plan.cavity_drives == ((), ())
    assert plan.projective_reads[1][0].start_ns == 10


@pytest.mark.parametrize(
    "second_triggers,difference",
    [
        ((0.031, 0.032, 0.033), "projective.*0.*1"),
        ((0.011, 0.031, 0.032), "ADC 2.*grid.*1.*2"),
    ],
)
def test_partition_rejects_point_with_different_logical_event_topology(
    second_triggers, difference
) -> None:
    base = template((pulse(0.010), pulse(0.030)), (0.011, 0.012, 0.031))
    window = replace(
        base.readouts[0],
        trigger_ticks=tick_array(np.array([[0.011, 0.012, 0.031], second_triggers])),
    )
    with pytest.raises(ValueError, match=rf"point 1.*{difference}"):
        readout_plan.classify_readout(
            replace(base, readouts=(window,)), readout_plan.ReadoutDrive(0, 1.0), {}
        , time_base=TIME_BASE)


def test_terminal_only_event_contents_preserve_late_triggers_and_deduplication() -> (
    None
):
    base = template((pulse(0.030),), (0.031, 0.040))
    drive = readout_plan.ReadoutDrive(0, 2.0)
    window = base.readouts[0]
    combined = readout_plan.readout_contents(base, drive, window, {}, time_base=TIME_BASE)
    separated = readout_plan.read_event_contents(base, drive, window, {}, time_base=TIME_BASE)
    assert len(combined.pulses) == 1
    np.testing.assert_array_equal(combined.inverse, [0, 0])
    for read, content in enumerate(separated):
        np.testing.assert_array_equal(content.end_ns, combined.end_ns)
        np.testing.assert_array_equal(content.inverse, combined.inverse)
        assert content.pulses == combined.pulses
        np.testing.assert_array_equal(
            content.grids[0][0].times_ns, combined.grids[0][read].times_ns
        )


@pytest.mark.parametrize("trigger_us", [0.014, 0.017])
def test_ringdown_trigger_belongs_to_previous_block_until_next_start(
    trigger_us,
) -> None:
    base = template(
        (pulse(0.030), pulse(0.010), pulse(0.020)), (trigger_us, 0.021, 0.031)
    )
    drive = readout_plan.ReadoutDrive(0, 1.0)
    plan = readout_plan.classify_readout(base, drive, {}, time_base=TIME_BASE)
    assert [[r.start_ns for r in point] for point in plan.projective_reads] == [
        [10, 20],
        [10, 20],
    ]
    assert plan.projective_reads[0][0].triggers[0].trigger_ns == trigger_us * 1000
    assert plan.cavity_drives == ((), ())
    contents = readout_plan.read_event_contents(base, drive, base.readouts[0], {}, time_base=TIME_BASE)
    np.testing.assert_array_equal(contents[0].end_ns, [10, 10])


@pytest.mark.parametrize("trigger_us", [0.018, 0.019])
def test_nonterminal_samples_must_finish_strictly_before_next_block(trigger_us) -> None:
    base = template(
        (pulse(0.030), pulse(0.010), pulse(0.020)), (trigger_us, 0.021, 0.031)
    )
    with pytest.raises(ValueError, match="last sample.*must precede") as rejection:
        readout_plan.classify_readout(base, readout_plan.ReadoutDrive(0, 1.0), {}, time_base=TIME_BASE)
    message = str(rejection.value)
    assert "block start=10 ns" in message
    assert "ADC 2 trigger index=0" in message
    assert "next block start=20 ns" in message


def test_trigger_at_next_start_pairs_with_that_block_not_the_previous_one() -> None:
    base = template((pulse(0.010), pulse(0.020), pulse(0.030)), (0.020, 0.030))
    drive = readout_plan.ReadoutDrive(0, 1.0)
    plan = readout_plan.classify_readout(base, drive, {}, time_base=TIME_BASE)
    assert [[r.start_ns for r in point] for point in plan.projective_reads] == [
        [20],
        [20],
    ]
    assert [[p.start_ns for p in point] for point in plan.cavity_drives] == [[10], [10]]
    contents = readout_plan.read_event_contents(base, drive, base.readouts[0], {}, time_base=TIME_BASE)
    np.testing.assert_array_equal(contents[0].end_ns, [20, 20])
    np.testing.assert_array_equal(contents[1].end_ns, [30, 30])


def test_trigger_before_any_incident_pulse_is_rejected_at_classification() -> None:
    base = template((pulse(0.010), pulse(0.030)), (0.009, 0.031))
    with pytest.raises(ValueError, match="trigger"):
        readout_plan.classify_readout(base, readout_plan.ReadoutDrive(0, 1.0), {}, time_base=TIME_BASE)


@pytest.mark.parametrize("kind", ["pulse", "mux"])
def test_readout_block_may_end_at_next_start_but_must_not_cross_it(kind) -> None:
    first, second, terminal = pulse(0.010, 0.010), pulse(0.020), pulse(0.030)
    base = template((terminal, second, first), (0.011, 0.021, 0.031))
    drive = readout_plan.ReadoutDrive(0, 2.0, kind)
    plan = readout_plan.classify_readout(base, drive, {}, time_base=TIME_BASE)
    assert [[r.start_ns for r in point] for point in plan.projective_reads] == [
        [10, 20],
        [10, 20],
    ]
    with pytest.raises(ValueError, match="cross|overlap"):
        readout_plan.classify_readout(
            replace(
                base,
                events=(
                    terminal,
                    second,
                    replace(first, duration_ticks=first.duration_ticks + 1),
                ),
            ),
            drive,
            {},
         time_base=TIME_BASE)
