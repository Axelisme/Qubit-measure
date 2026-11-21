"""Acquire composition advances density and cavity memory at the public seam."""

import re
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationCallbacks
from zcu_tools.simulate.adc import (
    CavityAverageCache,
    cavity_field_average,
    cavity_fields,
)
from zcu_tools.simulate.boundary import CancellationQuery, PulseEnvelope
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
    segment_spans,
)
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template
from zcu_tools.simulators.ftf.environment.numeric import resonators
from zcu_tools.simulators.ftf.physics import CavityDrive

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._templates import (
    expect_statistics,
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class ObservingEngine(Engine):
    """Delegate numerical evolution unchanged and observe its public boundary."""

    def __init__(self):
        super().__init__(1)
        self.calls: list[
            tuple[ModelSource, tuple[TimelineBatch, ...], EvolutionResult]
        ] = []

    def evolve(self, source, batches, **kwargs):
        result = super().evolve(source, batches, **kwargs)
        self.calls.append((source, tuple(batches), result))
        return result


@pytest.fixture
def cavity_span_owners(monkeypatch):
    owners: set[CavityAverageCache] = set()
    solve = CavityAverageCache.span

    def observe_span(owner, *args, **kwargs):
        owners.add(owner)
        return solve(owner, *args, **kwargs)

    # Engine inputs expose the physics, not whether ADC matched an exact span.
    monkeypatch.setattr(CavityAverageCache, "span", observe_span)
    return owners


def _assert_preserved_controls(batch, original, density):
    np.testing.assert_array_equal(batch.points, original.points)
    np.testing.assert_array_equal(batch.flux.level, original.flux.level)
    np.testing.assert_array_equal(batch.flux.start_ns, original.flux.start_ns)
    for actual, prior in zip(batch.tracks, original.tracks, strict=True):
        for name in (
            "start_ns",
            "duration_ns",
            "phase_rad",
            "frequency_ghz",
            "amplitude",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(prior, name))
    np.testing.assert_array_equal(batch.initial_densities, density)


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "shaped"])
def test_acquire_continues_each_point_span_density_field_and_common_origin(
    shaped, cavity_span_owners
):
    engine = ObservingEngine()
    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
    )
    base = template(tones=1)
    earlier = replace(
        base.events[-1],
        start_ticks=tick_array(np.full(2, 0.004)),
        duration_ticks=tick_array(np.full(2, 0.002)),
        gain=np.array([0.3, 0.5]),
        envelopes=(PulseEnvelope(ticks(0.003), (0j, 1j, 0.3)),) if shaped else (None,),
    )
    driven = replace(base, events=(earlier, *base.events))
    try:
        plan = plan_template(
            source.settings,
            snapshot(),
            driven,
            soccfg=profile_soccfg("ftf"),
            facts=source.model_facts(snapshot()),
            time_base=TIME_BASE,
        )
        spans = segment_spans(plan.batch)
        acquire = request(templates=(driven,))
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == max(map(len, spans)), (
            "cavity drives must evolve spans sequentially, not one vacuum-start timeline"
        )
        facts = source.model_facts(snapshot())
        pair = resonators(source.settings)
        uncached = [
            CavityDrive(facts.joint, pair, pulses) for pulses in plan.cavity_drives
        ]
        memory = np.zeros((2, 2), complex)
        density: NDArray[np.complex128] = np.tile(
            facts.source.initial_density, (2, 1, 1)
        )
        for step, (numeric_source, batches, result) in enumerate(engine.calls):
            assert numeric_source is facts.source
            assert len(batches) == 1
            batch = batches[0]
            assert batch.start_ns is not None
            assert batch.initial_densities is not None
            _assert_preserved_controls(batch, plan.batch, density)
            for point, point_spans in enumerate(spans):
                start, end = point_spans[min(step, len(point_spans) - 1)]
                if step >= len(point_spans):
                    start = end
                assert batch.start_ns[point] == start
                assert batch.end_ns[point] == end
                assert batch.segment_boundaries[point] == tuple(
                    t for t in plan.batch.segment_boundaries[point] if t <= end
                )
                if start == end:
                    assert not batch.segment_shifts[point]
                    continue
                (shift,) = batch.segment_shifts[point]
                assert (shift.start_ns, shift.end_ns) == (start, end)
                response, memory[point] = uncached[point].advance(
                    start, end, density[point].diagonal().real, memory[point]
                )
                assert (
                    shift.energy_shift_ghz.tobytes()
                    == response.stark_shift_ghz.tobytes()
                )
                assert (
                    shift.dephasing_per_ns.tobytes()
                    == response.dephasing_per_ns.tobytes()
                )
                if (
                    start == 6
                ):  # No drive remains in the final span; memory supplies the shift.
                    assert np.any(shift.energy_shift_ghz)
            density = result.densities
        assert sum(owner.hits for owner in cavity_span_owners) == (
            0 if shaped else 2
        ), "two flat prefixes should share second moments and endpoints across gains"
        (owner,) = cavity_span_owners
        assert owner.resident_bytes == 0
        # Equal-content acquire checks every span's route and does not replay it.
        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 1)
        assert len(engine.calls) == max(map(len, spans))
        assert source.preparation_cache.hits == 1
    finally:
        source.close()


def test_acquire_warns_once_for_reversed_stark_order_and_reports_gain_limit(recwarn):
    config = settings().with_telemetry(enabled=False)
    # Keep nominal gains in range while driving this synthetic cavity hard.
    readout = config.readout.model_copy(
        update={
            "drive": config.readout.drive.model_copy(
                update={"input_per_gain": 40_000.0}
            )
        }
    )
    source = FTFSource(
        config.model_copy(update={"readout": readout}),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=Engine(1),
    )
    binding = SimulationBinding(
        snapshot,
        source.execute,
        backend_version="ftf-stark-test",
        callbacks=SimulationCallbacks(
            on_begin=source.begin_acquire, on_end=source.finish_acquire
        ),
    )
    base = template()
    drive = replace(
        base.events[0],
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=tick_array(np.full(2, 0.002)),
        gain=np.array([0.5, 1.0]),
        envelopes=(None,),
    )
    warnings.simplefilter("always", UserWarning)

    def acquire(scale: float) -> None:
        driven = replace(
            base, events=(replace(drive, gain=drive.gain * scale), *base.events)
        )
        with binding.acquire() as sequence:
            inputs = replace(request(templates=(driven,)), sequence=sequence)
            for index in range(2):
                stats = expect_statistics(binding.sample_round(inputs, index))
                np.testing.assert_array_equal(stats.state_counts.sum(axis=-1), [3, 3])
                assert not recwarn.list

    try:
        acquire(1.0)
        message = str(recwarn.pop(UserWarning).message)
        assert not recwarn.list
        assert "states 0 and 1" in message
        assert "affected points: 2, affected segments: 4" in message
        assert "first at point 0 segment [0, 2] ns" in message
        assert "outside the dispersive model's validity" in message
        scale = re.search(r"common gain scale factor of at most ([\d.e+-]+)", message)
        assert scale is not None
        limit = float(scale[1])
        assert 0 < limit < 1

        # An equal-content acquire also reports once, even when it reuses the solve.
        acquire(1.0)
        assert source.preparation_cache.hits == 1
        assert str(recwarn.pop(UserWarning).message) == message
        assert not recwarn.list

        # Ringdown is included: the reported limit brackets warning onset.
        acquire(0.99 * limit)
        assert not recwarn.list
        acquire(1.01 * limit)
        assert recwarn.pop(UserWarning)
        assert not recwarn.list
    finally:
        source.close()


class ShiftSensitiveEngine(EngineDouble):
    """At terminal evolution, distinguish equal pure states by their input shifts."""

    def __init__(self) -> None:
        super().__init__()
        self.shifts: dict[tuple[float, float], tuple[NDArray[np.float64], ...]] = {}
        self.terminal_calls: list[tuple[tuple[float, float], set[tuple[int, int]]]] = []

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
        result = super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
        )
        rows = [(batch, row) for batch in batches for row in range(len(batch.points))]
        densities = result.densities.copy()
        terminal = set()
        span = None
        for i, (batch, row) in enumerate(rows):
            assert batch.start_ns is not None
            bounds = (float(batch.start_ns[row]), float(batch.end_ns[row]))
            if bounds in self.shifts:
                # A repeated projected state can only be identified by the
                # shift inherited from the first herald's cavity field.
                (shift,) = batch.segment_shifts[row]
                assert (shift.start_ns, shift.end_ns) == bounds
                (first_state,) = (
                    state
                    for state, expected in enumerate(self.shifts[bounds])
                    if np.allclose(
                        shift.energy_shift_ghz, expected, rtol=1e-10, atol=1e-15
                    )
                )
                second_state = int(densities[i, 1, 1].real == 1.0)
                np.testing.assert_array_equal(
                    densities[i], np.diag(np.eye(2)[second_state])
                )
                terminal.add((first_state, second_state))
                assert span is None or span == bounds
                span = bounds
                densities[i] = np.diag(np.eye(2)[first_state])
            elif bounds[1] in {start for start, _ in self.shifts}:
                # Both next states must remain reachable after each first state.
                densities[i] = np.diag([0.5, 0.5])
        if span is not None:
            assert len(rows) == 4
            self.terminal_calls.append((span, terminal))
        return EvolutionResult(result.points[::-1], densities[::-1], result.paths)


def test_projective_results_follow_history_when_equal_states_have_different_shifts():
    engine = ShiftSensitiveEngine()
    base, readout = herald_template()
    # A second herald gives cavity memory a projected continuation boundary.
    terminal = tuple(
        replace(event, start_ticks=event.start_ticks + ticks(0.012))
        for event in base.events[-2:]
    )
    middle = tuple(
        replace(event, duration_ticks=tick_array(np.full(2, 0.002)))
        for event in base.events[-2:]
    )
    windows = tuple(
        replace(
            window,
            trigger_ticks=np.column_stack(
                (window.trigger_ticks, window.trigger_ticks[:, -1] + ticks(0.012))
            ),
            frequency_mhz=np.column_stack(
                (window.frequency_mhz, window.frequency_mhz[:, -1])
            ),
            phase_deg=np.column_stack((window.phase_deg, window.phase_deg[:, -1])),
            generator_channels=np.column_stack(
                (window.generator_channels, window.generator_channels[:, -1])
            ),
        )
        for window in base.readouts
    )
    driven = replace(
        base,
        events=(*base.events[:2], *middle, *terminal),
        readouts=windows,
        duration_ticks=base.duration_ticks + ticks(0.012),
    )
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "chip": numeric.chip.model_copy(
                update={
                    "temperature_k": 1.0,
                }
            )
        }
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=engine,
    )
    try:
        facts = source.model_facts(snapshot())
        plan = plan_template(
            numeric,
            snapshot(),
            driven,
            soccfg=profile_soccfg("ftf"),
            readout=readout,
            facts=facts,
            time_base=TIME_BASE,
        )
        model = facts.readout
        bare = np.array(
            [
                numeric.readout.resonators.r1.frequency_ghz,
                numeric.readout.resonators.r2.frequency_ghz,
            ]
        )
        chi = model.frequencies_ghz - bare
        for reads, end in zip(plan.projective_reads, plan.batch.end_ns, strict=True):
            first, second = reads
            pulses = (*first.pulses, *second.pulses)
            fields = cavity_fields(
                model,
                pulses,
                np.array([first.start_ns, second.start_ns]),
                initial_time_ns=first.start_ns,
            )
            shifts = []
            for memory in fields[-1]:
                average = cavity_field_average(
                    model,
                    pulses,
                    second.start_ns,
                    end,
                    initial_field=memory,
                    initial_time_ns=second.start_ns,
                )
                shifts.append((chi * average.photons).sum(axis=-1))
            ground_shift, excited_shift = shifts
            assert not np.allclose(ground_shift, excited_shift, rtol=1e-10, atol=1e-15)
            engine.shifts[second.start_ns, float(end)] = (ground_shift, excited_shift)

        acquire = replace(request(templates=(driven,)), reps=1024)
        stats = expect_statistics(
            source.open_acquire(acquire).sample_round(snapshot(), 0)
        )
        assert len(engine.terminal_calls) == 2
        assert dict(engine.terminal_calls) == {
            span: {(0, 0), (0, 1), (1, 0), (1, 1)} for span in engine.shifts
        }
        # C-order axes: original point, first herald, second herald, terminal state.
        # All four histories must be sampled, and the terminal state follows
        # the first herald's field even for histories sharing a second state.
        counts = stats.state_counts.reshape(2, 2, 2, 2)
        assert np.all(counts.sum(axis=-1) > 0)
        np.testing.assert_array_equal(counts[:, 0, :, 1], 0)
        np.testing.assert_array_equal(counts[:, 1, :, 0], 0)
    finally:
        source.close()


def test_mixed_acquire_keeps_undriven_templates_on_the_single_evolve_path():
    engine = ObservingEngine()
    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
    )
    base = template()
    earlier = replace(
        base.events[-1],
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=tick_array(np.full(2, 0.002)),
        envelopes=(None,),
    )
    driven = replace(base, events=(earlier, *base.events))
    undriven = replace(base, points=np.array([2, 3]))
    acquire = replace(
        request(templates=(driven, undriven)), point_dims=(4,), loop_dims=(3, 4)
    )
    try:
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        plain_batches = [
            batch
            for _, batches, _ in engine.calls
            for batch in batches
            if 2 in batch.points
        ]
        assert len(plain_batches) == 1, (
            "undriven templates retain one full evolution even in mixed acquires"
        )
        assert plain_batches[0].start_ns is None
        assert plain_batches[0].initial_densities is None
        assert source.evaluations[0].paths.points_by_class
        for name, members in source.evaluations[0].paths.point_members.items():
            assert source.evaluations[0].paths.points_by_class[name] == len(members)
    finally:
        source.close()


def test_no_cavity_acquire_keeps_single_evolve_without_continuation():
    engine = ObservingEngine()
    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
    )
    try:
        acquire = request()
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == 1
        (batch,) = engine.calls[0][1]
        assert batch.start_ns is None
        assert batch.initial_densities is None
        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 1)
        assert len(engine.calls) == 1
    finally:
        source.close()
