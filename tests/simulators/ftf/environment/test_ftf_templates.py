"""Template-to-engine composition and per-acquire FTF round semantics."""

from collections.abc import Sequence
from dataclasses import replace
from threading import Event
from typing import override

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.device import DeviceManager
from zcu_tools.simulate.adc import ConditionalReadout
from zcu_tools.simulate.boundary import (
    CancellationQuery,
    RoundSignals,
    RoundStatistics,
    TimeBase,
)
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.environment import (
    FTFReadoutConfig,
    FTFReceiverConfig,
    FTFTemplatePlan,
)

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.program._profiles import profile_soccfg
from tests.simulators.ftf.environment._templates import (
    expect_statistics,
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


def test_template_units_common_phase_flat_readout_and_dc_model() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    plan = ftf.plan_template(
        settings(),
        snapshot(),
        template(tones=1),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    track = plan.batch.tracks[0]
    np.testing.assert_array_equal(track.start_ns, [0.0, 2.0])
    np.testing.assert_array_equal(track.duration_ns, [2.0, 2.0])
    np.testing.assert_array_equal(track.frequency_ghz, [0.25, 0.25])
    np.testing.assert_allclose(track.phase_rad, [0.0, 0.0], atol=1e-14)
    np.testing.assert_array_equal(track.amplitude, [-0.2, -0.2])
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    pulse = plan.readouts[0].batch.events[0][0].pulses[0]
    assert pulse.envelope is None
    assert pulse.amplitude == pytest.approx(0.2j)
    assert pulse.phase_rad == pytest.approx(np.pi / 6 - 12 * np.pi)
    np.testing.assert_array_equal(
        plan.readouts[0].batch.events[0][0].grids[0].times_ns, [8, 9, 10, 11]
    )
    np.testing.assert_array_equal(
        plan.readouts[0].batch.model.receiver.noise_covariance, np.zeros((2, 2))
    )


@pytest.mark.parametrize("tones,path", [(0, "t4"), (3, "dop853")])
def test_template_engine_paths_match_strict_secular_reference(
    tones: int, path: str
) -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template(tones=tones)
    # Keep the 2 ns drive support required by FTF's quantization, but shorten
    # the idle before readout. Unit conversion has its own unscaled test above.
    tiny = replace(
        base,
        events=(
            *base.events[:-1],
            replace(
                base.events[-1], start_ticks=base.events[-1].start_ticks - ticks(0.004)
            ),
        ),
        readouts=tuple(
            replace(r, trigger_ticks=r.trigger_ticks - ticks(0.004))
            for r in base.readouts
        ),
    )
    plan = ftf.plan_template(
        settings(), snapshot(), tiny, soccfg=profile_soccfg("ftf"), time_base=TIME_BASE
    )
    engine = Engine(1)
    try:
        actual = engine.evolve(plan.source, (plan.batch,), cancel=Event())
    finally:
        engine.close()
    assert actual.paths.segments_by_kernel[path] > 0
    model = plan.source.model((0, 0, 0))
    rates = model.rates_per_ns
    loss = rates.sum(axis=0)
    decay = (loss[:, None] + loss[None, :]) / 2 + model.dephasing_per_ns
    for p in range(2):

        def rhs(t, flat, p=p):
            rho = flat.reshape(2, 2)
            h = np.diag(model.energies_ghz).astype(complex)
            for track in plan.batch.tracks:
                if track.start_ns[p] <= t < track.start_ns[p] + track.duration_ns[p]:
                    drive = next(d for d in model.drives if d.name == track.port)
                    z = track.amplitude[p] * np.exp(
                        1j
                        * (2 * np.pi * track.frequency_ghz[p] * t + track.phase_rad[p])
                    )
                    h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
            out = -2j * np.pi * (h @ rho - rho @ h) - decay * rho
            out[np.diag_indices(2)] += rates @ rho.diagonal()
            return out.ravel()

        state = plan.source.initial_density.ravel()
        boundaries = sorted(
            {
                0.0,
                float(plan.batch.end_ns[p]),
                *(float(t.start_ns[p]) for t in plan.batch.tracks),
                *(float(t.start_ns[p] + t.duration_ns[p]) for t in plan.batch.tracks),
            }
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if end > start:
                state = solve_ivp(
                    rhs,
                    (start, end),
                    state,
                    method="DOP853",
                    rtol=1e-11,
                    atol=1e-13,
                    max_step=0.05,
                ).y[:, -1]
        np.testing.assert_allclose(
            actual.densities[p], state.reshape(2, 2), atol=1e-5, rtol=0
        )


def test_readout_ignores_controls_after_start_and_rejects_multiple_terminal_pulses() -> (
    None
):
    from zcu_tools.simulators.ftf import environment as ftf

    base = template()
    # Charge controls beginning after readout must have no effect.
    extra = replace(
        base.events[0],
        gen_ch=2,
        start_ticks=tick_array(np.full(2, 0.015)),
        gain=np.full(2, 0.5),
    )
    overlap = replace(base, events=(*base.events, extra))
    plans = [
        ftf.plan_template(
            settings(), snapshot(), t, soccfg=profile_soccfg("ftf"), time_base=TIME_BASE
        )
        for t in (base, overlap)
    ]
    engine = Engine(1)
    try:
        results = [engine.evolve(p.source, (p.batch,), cancel=Event()) for p in plans]
    finally:
        engine.close()
    np.testing.assert_array_equal(results[0].densities, results[1].densities)
    np.testing.assert_array_equal(
        plans[0].readouts[0].batch.model.frequencies_ghz,
        plans[1].readouts[0].batch.model.frequencies_ghz,
    )
    with pytest.raises(ValueError, match="one.*readout"):
        ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(*base.events, base.events[0])),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )


def test_cavity_drives_are_classified_per_point_with_calibrated_envelopes() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template()
    terminal = base.events[-1]
    earlier = replace(
        terminal,
        start_ticks=tick_array(np.array([0.0, 0.002])),
        duration_ticks=tick_array(np.full(2, 0.004)),
        frequency_mhz=np.full(2, 5900.0),
        gain=np.array([0.3, -0.4]),
    )
    # Mutually exclusive slots must classify by active point, not slot order.
    alternate = replace(earlier, active=np.array([False, True]))
    earlier = replace(earlier, active=np.array([True, False]))
    try:
        plan = ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(terminal, earlier, alternate)),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )
    except ValueError as error:
        pytest.fail(f"nonterminal cavity drives must be accepted: {error}")
    drives = getattr(plan, "cavity_drives", None)
    assert drives is not None, "plan must retain nonterminal readout drives"
    assert tuple(map(len, drives)) == (1, 1)
    first, second = drives[0][0], drives[1][0]
    assert (first.start_ns, first.end_ns, first.frequency_ghz) == (0, 4, 5.9)
    assert (second.start_ns, second.end_ns) == (2, 6)
    assert first.amplitude == pytest.approx(0.6j)
    assert second.amplitude == pytest.approx(-0.8j)
    assert first.envelope is None
    assert second.phase_rad == pytest.approx(np.pi / 6 - 2 * np.pi * 5.9 * 3)
    np.testing.assert_array_equal(plan.batch.end_ns, [8, 10])
    assert all(
        len(content) == 1
        for events in plan.readouts[0].batch.events
        for content in (events[0].pulses,)
    )


def test_herald_plan_aligns_two_events_when_one_adc_only_observes_terminal() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base, config = herald_template()
    plan = ftf.plan_template(
        settings(),
        snapshot(),
        base,
        soccfg=profile_soccfg("ftf"),
        readout=config,
        time_base=TIME_BASE,
    )
    assert tuple(item.channel for item in plan.readouts) == (4, 1)
    np.testing.assert_array_equal(plan.batch.end_ns, [8, 10])
    assert plan.cavity_drives == ((), ())
    assert tuple(len(reads) for reads in plan.projective_reads) == (1, 1)
    for point, start in enumerate((0.0, 2.0)):
        assert plan.projective_reads[point][0].start_ns == start
        assert start in plan.batch.segment_boundaries[point]
        for item, expected_grids in zip(plan.readouts, ((1, 1), (0, 1)), strict=True):
            events = item.batch.events[item.batch.content_index[point]]
            assert len(events) == 2
            assert tuple(len(event.grids) for event in events) == expected_grids
            herald_frequencies = (5.9,) if item.channel == 4 else (5.9, 5.98)
            assert tuple(p.start_ns for p in events[0].pulses) == (start,) * len(
                herald_frequencies
            )
            assert (
                tuple(p.frequency_ghz for p in events[0].pulses) == herald_frequencies
            )
            assert tuple(p.amplitude for p in events[0].pulses) == (0.6j,) * len(
                herald_frequencies
            )
            assert events[0].pulses[0].phase_rad == pytest.approx(
                np.pi / 6 - 2 * np.pi * 5.9 + 2 * np.pi * 0.1 * start
            )
            frequency = 6.0 if item.channel == 4 else 6.08
            assert tuple(p.start_ns for p in events[1].pulses) == (start + 8,)
            assert tuple(p.frequency_ghz for p in events[1].pulses) == (frequency,)
            expected_phase = np.deg2rad(30 - 360 * frequency)
            expected_phase -= 2 * np.pi * (frequency - 6.0) * (start + 8)
            assert events[1].pulses[0].phase_rad == pytest.approx(expected_phase)
            np.testing.assert_array_equal(
                events[1].grids[0].times_ns, np.arange(start + 8, start + 12)
            )
        observed = plan.readouts[0].batch
        herald_event = observed.events[observed.content_index[point]][0]
        np.testing.assert_array_equal(
            herald_event.grids[0].times_ns, np.arange(start + 3, start + 7)
        )
        assert herald_event.grids[0].demodulation_frequency_ghz == 5.9
        assert herald_event.grids[0].phase_rad == pytest.approx(np.deg2rad(10))


def _plan_cross_block_readout(
    time_of_flight_ns: float, *, first_trigger_us: float = 1.110801
) -> FTFTemplatePlan:
    from zcu_tools.simulators.ftf.environment import plan_template

    # Zero post-delay: the terminal pulse starts at the herald pulse's end.
    time_base = TimeBase.from_clocks((1_000_000.0, 1_000.0))
    base = template()
    herald = replace(
        base.events[0],
        start_ticks=np.zeros(2, dtype=np.int64),
        duration_ticks=np.full(2, time_base.ticks(1.2), dtype=np.int64),
        phase_reference_ticks=0,
    )
    terminal = replace(
        herald,
        start_ticks=np.full(2, time_base.ticks(1.2), dtype=np.int64),
        gain=np.full(2, 0.5),
    )
    window = replace(
        base.readouts[0],
        samples=200,
        sample_period_ticks=time_base.period_ticks(1_000.0),
        trigger_ticks=np.tile(
            [time_base.ticks(first_trigger_us), time_base.ticks(2.310801)], (2, 1)
        ),
        frequency_mhz=np.full((2, 2), 6000.0),
        phase_deg=np.zeros((2, 2)),
        generator_channels=np.zeros((2, 2), dtype=np.int64),
    )
    model = settings()
    model = model.model_copy(
        update={
            "readout": model.readout.model_copy(
                update={
                    "adc": model.readout.receivers[0].model_copy(
                        update={"time_of_flight_ns": time_of_flight_ns}
                    )
                }
            )
        }
    )
    return plan_template(
        model,
        snapshot(),
        replace(
            base,
            events=(herald, terminal),
            readouts=(window,),
            duration_ticks=np.full(2, time_base.ticks(2.6), dtype=np.int64),
        ),
        soccfg=profile_soccfg("ftf"),
        time_base=time_base,
    )


def test_later_readout_pulse_arriving_after_herald_window_stays_in_its_event() -> None:
    plan = _plan_cross_block_readout(time_of_flight_ns=387.370)
    np.testing.assert_array_equal(plan.batch.end_ns, [1200, 1200])
    batch = plan.readouts[0].batch
    for content in batch.content_index:
        herald, terminal = batch.events[content]
        assert tuple(p.start_ns for p in herald.pulses) == (0,)
        assert tuple(p.start_ns for p in terminal.pulses) == (1200,)
        (herald_grid,) = herald.grids
        (terminal_grid,) = terminal.grids
        np.testing.assert_allclose(herald_grid.times_ns, 1110.801 + np.arange(200))
        np.testing.assert_allclose(terminal_grid.times_ns, 2310.801 + np.arange(200))


def test_window_crossing_a_block_boundary_uses_the_later_event() -> None:
    # Device-time samples run from 1010.801 to 1209.801 ns and reach terminal.
    plan = _plan_cross_block_readout(time_of_flight_ns=100.0)
    np.testing.assert_array_equal(plan.batch.end_ns, [1200, 1200])
    assert plan.projective_reads == ((), ())
    batch = plan.readouts[0].batch
    for content in batch.content_index:
        (terminal,) = batch.events[content]
        assert {p.start_ns for p in terminal.pulses} == {0, 1200}
        assert len(terminal.grids) == 2
        np.testing.assert_allclose(
            terminal.grids[0].times_ns, 1110.801 + np.arange(200)
        )


def test_readout_window_belongs_to_the_in_flight_herald_pulse() -> None:
    plan = _plan_cross_block_readout(
        time_of_flight_ns=387.370, first_trigger_us=1.27906
    )
    batch = plan.readouts[0].batch
    for content in batch.content_index:
        herald, terminal = batch.events[content]
        assert tuple(p.start_ns for p in herald.pulses) == (0,)
        assert tuple(p.start_ns for p in terminal.pulses) == (1200,)
        (grid,) = herald.grids
        np.testing.assert_allclose(grid.times_ns, 1279.06 + np.arange(200))


def test_herald_only_observations_keep_an_unobserved_terminal_event() -> None:
    from zcu_tools.simulators.ftf.environment import plan_template

    base, config = herald_template()
    both = base.readouts[0]
    herald_only = replace(
        both,
        trigger_ticks=both.trigger_ticks[:, :1],
        frequency_mhz=both.frequency_mhz[:, :1],
        phase_deg=both.phase_deg[:, :1],
        generator_channels=both.generator_channels[:, :1],
    )
    base = replace(base, readouts=(herald_only, replace(herald_only, channel=1)))
    plan = plan_template(
        settings(),
        snapshot(),
        base,
        soccfg=profile_soccfg("ftf"),
        readout=config,
        time_base=TIME_BASE,
    )
    np.testing.assert_array_equal(plan.batch.end_ns, [8, 10])
    for item in plan.readouts:
        for point, index in enumerate(item.batch.content_index):
            herald, terminal = item.batch.events[index]
            assert len(herald.grids) == 1
            assert terminal.grids == ()
            assert tuple(p.start_ns for p in terminal.pulses) == (8 + 2 * point,) * 2
            assert tuple(p.frequency_ghz for p in terminal.pulses) == (6.0, 6.08)


def test_window_before_all_pulses_keeps_receiver_baseline_without_projection() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template()
    receiver = FTFReceiverConfig(
        0,
        1000.0,
        ((0.0, 0.0), (0.0, 0.0)),
        offset_adc=(11.0, -7.0),
        ripple_period_mhz=1000.0,
        pivot_ghz=0.0,
    )
    early = replace(base.readouts[0], trigger_ticks=np.zeros((2, 1), dtype=np.int64))
    normal, lookback = (
        ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, readouts=(window,)),
            readout=FTFReadoutConfig(receiver, None),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )
        for window in (base.readouts[0], early)
    )
    assert lookback.projective_reads == normal.projective_reads == ((), ())
    np.testing.assert_array_equal(lookback.batch.end_ns, normal.batch.end_ns)
    batch = lookback.readouts[0].batch
    for events in batch.events:
        (event,) = events
        assert event.pulses == ()
        (grid,) = event.grids
        np.testing.assert_array_equal(grid.times_ns, [0, 1, 2, 3])
    sums = ConditionalReadout(batch.model).responses(batch).sums
    np.testing.assert_array_equal(sums[..., 0], 44.0)
    np.testing.assert_array_equal(sums[..., 1], -28.0)


def test_readout_rejects_missing_pulse_at_only_some_points() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template()
    pulse = replace(base.events[0], active=np.array([False, True]))
    with pytest.raises(
        ValueError, match="requires one terminal readout pulse per point"
    ):
        ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(pulse,)),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )


def test_cavity_drive_must_not_cross_terminal_start_but_may_end_there() -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template()
    terminal = base.events[-1]
    earlier = replace(
        terminal,
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=terminal.start_ticks,
    )
    try:
        plan = ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(earlier, terminal)),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )
    except ValueError as error:
        pytest.fail(f"cavity drives may end exactly at terminal start: {error}")
    drives = getattr(plan, "cavity_drives", None)
    assert drives is not None, "plan must retain earlier readout drives"
    assert tuple(len(p) for p in drives) == (1, 1)
    crossing = replace(earlier, duration_ticks=earlier.duration_ticks + 1)
    with pytest.raises(ValueError, match="cross|overlap"):
        ftf.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(crossing, terminal)),
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )


class CountingEngine(Engine):
    def __init__(self):
        super().__init__(1)
        self.calls = 0

    @override
    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
        capture: CaptureBuffer | None = None,
    ) -> EvolutionResult:
        self.calls += 1
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
        )


def test_round_statistics_reuse_snapshot_and_engine_across_acquires() -> None:
    from zcu_tools.simulators.ftf.environment import FTFSource

    assert callable(FTFSource.open_acquire)
    engine = CountingEngine()
    source = FTFSource(
        settings(), DeviceManager(), soccfg=profile_soccfg("ftf"), engine=engine
    )
    try:
        session = source.open_acquire(request())
        first = expect_statistics(session.sample_round(snapshot(), 0))
        second = expect_statistics(session.sample_round(snapshot(), 1))
        assert isinstance(first, RoundStatistics)
        assert first.channels[0].sums.shape == (2, 1, 2)
        assert engine.calls == 1
        assert not np.array_equal(first.channels[0].sums, second.channels[0].sums)
        session.sample_round(snapshot(0.4), 2)
        assert engine.calls == 2
        source.open_acquire(request()).sample_round(snapshot(), 0)
        assert engine.calls == 3
        traces = source.open_acquire(request("decimated")).sample_round(snapshot(), 0)
        assert isinstance(traces, RoundSignals)
        assert traces.channels[0].iq.shape == (6, 1, 4, 2)
    finally:
        source.close()


class ResidualEngine(Engine):
    """Configured executor supplying controlled densities at the C3 boundary."""

    def __init__(self, populations):
        super().__init__(1)
        self.populations = populations

    @override
    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
        capture: CaptureBuffer | None = None,
    ) -> EvolutionResult:
        from zcu_tools.simulate.evolution import PathDiagnostics

        points = np.concatenate([batch.points for batch in batches])
        densities = np.repeat(np.diag(self.populations)[None], len(points), axis=0)
        return EvolutionResult(points, densities, PathDiagnostics())


@pytest.mark.parametrize("inactive_slot", [False, True])
def test_pulseless_readout_samples_only_receiver_offset_and_noise(
    inactive_slot: bool,
) -> None:
    from zcu_tools.simulators.ftf import environment as ftf

    base = template(tones=1)
    control, readout = base.events
    events = (control,)
    if inactive_slot:
        events += (replace(readout, active=np.zeros(2, dtype=bool)),)
    pulseless = replace(base, events=events)
    offset = (11.0, -7.0)
    noise_psd = ((4.0, 0.5), (0.5, 2.0)) if inactive_slot else ((0.0, 0.0), (0.0, 0.0))
    config = FTFReadoutConfig(
        FTFReceiverConfig(
            0,
            1000.0,
            noise_psd,
            offset_adc=offset,
            ripple_period_mhz=1000.0,
            pivot_ghz=0.0,
        )
    )
    numeric = settings()
    dc = snapshot()
    soccfg = profile_soccfg("ftf")
    plan = ftf.plan_template(
        numeric, dc, pulseless, soccfg=soccfg, time_base=TIME_BASE, readout=config
    )
    assert len(plan.batch.tracks) == 1
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    (receiver,) = plan.readouts
    assert receiver.channel == 0
    assert receiver.batch.model.reference_frequency_ghz == 0.0
    for point, index in enumerate(receiver.batch.content_index):
        (event,) = receiver.batch.events[index]
        assert event.pulses == ()
        (grid,) = event.grids
        np.testing.assert_array_equal(grid.times_ns, np.arange(8, 12) + 2 * point)
        assert grid.demodulation_frequency_ghz == 6.0

    source = ftf.FTFSource(
        numeric,
        DeviceManager(),
        soccfg=soccfg,
        engine=ResidualEngine([0.25, 0.75]),
        readout=config,
    )
    acquire = replace(request("decimated", templates=(pulseless,)), reps=2048)
    try:
        result = source.open_acquire(acquire).sample_round(dc, 0)
        assert isinstance(result, RoundSignals)
        (channel,) = result.channels
        assert channel.channel == 0
        assert channel.iq.shape == (2 * acquire.reps, 1, 4, 2)
        observations = channel.iq.reshape(-1, 2)
        if inactive_slot:
            # This fixture's sample period is 1 ns; its sample covariance equals the PSD.
            np.testing.assert_allclose(observations.mean(axis=0), offset, atol=0.09)
            np.testing.assert_allclose(np.cov(observations.T), noise_psd, atol=0.2)
        else:
            np.testing.assert_array_equal(
                observations, np.broadcast_to(offset, observations.shape)
            )
    finally:
        source.close()


@pytest.mark.parametrize(
    "populations, normalized",
    [
        ([0.25 * (1 + 9e-7), 0.75 * (1 + 9e-7)], [0.25, 0.75]),
        ([-1e-6, 1.000001], [0, 1]),
        ([-2e-6, 1.000002], [0, 1]),
        ([0, 0.999999], [0, 1]),
    ],
)
def test_acquire_explicitly_normalizes_in_budget_evolution(populations, normalized):
    from zcu_tools.simulators.ftf.environment import FTFSource

    source = FTFSource(
        settings(),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=ResidualEngine(populations),
    )
    acquire = replace(request(), reps=1_000_000_000)
    try:
        try:
            result = expect_statistics(
                source.open_acquire(acquire).sample_round(snapshot(), 0)
            )
        except ValueError as error:
            pytest.fail(f"in-budget evolution must complete acquire: {error}")
        identity = np.random.SeedSequence((acquire.root_seed, acquire.sequence, 0, 0))
        rng = np.random.Generator(np.random.PCG64(identity))
        expected = rng.multinomial(acquire.reps, np.tile(normalized, (2, 1)))
        np.testing.assert_array_equal(result.state_counts, expected)
    finally:
        source.close()


@pytest.mark.parametrize("populations", [[0, 1.000002]])
def test_acquire_rejects_out_of_budget_evolution_and_retains_diagnostics(populations):
    from zcu_tools.simulators.ftf.environment import FTFSource

    source = FTFSource(
        settings(),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=ResidualEngine(populations),
    )
    try:
        with pytest.raises(ValueError, match="2 rows") as rejection:
            source.open_acquire(request()).sample_round(snapshot(), 0)
        assert len(source.evaluations) == 1
        assert source.evaluations[0].round_index == 0
        assert getattr(source.evaluations[0], "rejected_rows", None) == 2
        assert getattr(rejection.value, "rejected_rows", None) == 2
    finally:
        source.close()


def test_template_wires_receiver_delay_and_mismatch_without_delaying_evolution() -> (
    None
):
    from zcu_tools.simulators.ftf.environment import plan_template

    base = settings()
    pair = base.readout.resonators
    configured = base.model_copy(
        update={
            "readout": base.readout.model_copy(
                update={
                    "adc": base.readout.receivers[0].model_copy(
                        update={"time_of_flight_ns": 400.0}
                    ),
                    "resonators": pair.model_copy(
                        update={
                            "r1": pair.r1.model_copy(
                                update={"mismatch_phase_rad": 0.6}
                            ),
                            "r2": pair.r2.model_copy(
                                update={"mismatch_phase_rad": -0.4}
                            ),
                        }
                    ),
                }
            ),
        }
    )
    plan = plan_template(
        configured,
        snapshot(),
        template(),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    assert (
        getattr(plan.readouts[0].batch.model.receiver, "time_of_flight_ns", None)
        == 400.0
    )
    np.testing.assert_array_equal(
        getattr(plan.readouts[0].batch.model, "mismatch_phase_rad", None), [0.6, -0.4]
    )
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    # At this delay, the unchanged grid samples before the first pulse arrives.
    assert plan.readouts[0].batch.events[0][0].pulses == ()
    np.testing.assert_array_equal(
        plan.readouts[0].batch.events[0][0].grids[0].times_ns, [8, 9, 10, 11]
    )


def test_sampled_readout_template_preserves_linear_envelope_without_field_ode(
    monkeypatch,
):
    from zcu_tools.simulate import adc
    from zcu_tools.simulate.adc import response
    from zcu_tools.simulate.boundary import PulseEnvelope
    from zcu_tools.simulators.ftf.environment import plan_template

    base = template()
    samples = PulseEnvelope(ticks(0.006), (0.1, 0.8 + 0.2j, 1.0 - 0.1j, 0.2))
    shaped = replace(base, events=(replace(base.events[0], envelopes=(samples,)),))
    batch = (
        plan_template(
            settings(),
            snapshot(),
            shaped,
            soccfg=profile_soccfg("ftf"),
            time_base=TIME_BASE,
        )
        .readouts[0]
        .batch
    )
    expected = np.stack(
        [
            adc.ConditionalReadout(batch.model, rtol=1e-12, atol=1e-14)
            .response(ps, gs[0])
            .traces.sum(axis=-2)
            for events in batch.events
            for ps, gs in ((events[0].pulses, events[0].grids),)
        ]
    )
    calls = []
    solve = response.solve_ivp

    def observe(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(response, "solve_ivp", observe)
    actual = adc.ConditionalReadout(batch.model).responses(batch)
    assert not calls, "FTF sampled readout must expose its numeric envelope to ADC"
    np.testing.assert_allclose(actual.sums[:, 0], expected, rtol=1e-6, atol=1e-9)
