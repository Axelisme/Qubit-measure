"""Acquire composition advances density and cavity memory at the public seam."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.core.simulation import PulseEnvelope
from zcu_tools.device import DeviceManager
from zcu_tools.simulate.adc import cavity_field_average, cavity_fields
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
    segment_spans,
)
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.program._profiles import profile_soccfg
from tests.simulators.ftf.environment._templates import (
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


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "shaped"])
def test_acquire_continues_each_point_span_density_field_and_common_origin(shaped):
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
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=tick_array(np.array([0.002, 0.006])),
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
        model = facts.readout
        chi = model.frequencies_ghz - np.array(
            [
                r.frequency_ghz
                for r in (
                    source.settings.readout.resonators.r1,
                    source.settings.readout.resonators.r2,
                )
            ]
        )
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
            np.testing.assert_array_equal(batch.points, plan.batch.points)
            np.testing.assert_array_equal(batch.flux.level, plan.batch.flux.level)
            np.testing.assert_array_equal(batch.flux.start_ns, plan.batch.flux.start_ns)
            for actual, original in zip(batch.tracks, plan.batch.tracks, strict=True):
                for name in (
                    "start_ns",
                    "duration_ns",
                    "phase_rad",
                    "frequency_ghz",
                    "amplitude",
                ):
                    np.testing.assert_array_equal(
                        getattr(actual, name), getattr(original, name)
                    )
            np.testing.assert_array_equal(batch.initial_densities, density)
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
                expected = cavity_field_average(
                    model,
                    plan.cavity_drives[point],
                    start,
                    end,
                    initial_field=memory[point],
                    initial_time_ns=start,
                )
                np.testing.assert_allclose(
                    shift.energy_shift_ghz,
                    (chi * expected.photons).sum(axis=-1),
                    rtol=1e-10,
                    atol=1e-15,
                )
                fields = cavity_fields(
                    model,
                    plan.cavity_drives[point],
                    np.array([start, end]),
                    initial_field=memory[point],
                    initial_time_ns=start,
                )
                branch = np.argmax(density[point].diagonal().real)
                memory[point] = fields[-1, branch]
            density = result.densities
        # Equal-content acquire checks every span's route and does not replay it.
        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 1)
        assert len(engine.calls) == max(map(len, spans))
        assert source.preparation_cache.hits == 1
    finally:
        source.close()


def test_projection_outcomes_carry_their_own_cavity_memory_into_the_next_span():
    engine = ObservingEngine()
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
                    "f1": numeric.chip.f1.model_copy(update={"q_cap": 10.0}),
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
        source.open_acquire(request(templates=(driven,))).sample_round(snapshot(), 0)
        calls = [
            batches[0] for _, batches, _ in engine.calls if batches[0].points[0] == 0
        ]
        assert len(calls) == 6
        pulses = tuple(
            pulse for read in plan.projective_reads[0] for pulse in read.pulses
        )
        model = facts.readout
        bare = np.array(
            [
                numeric.readout.resonators.r1.frequency_ghz,
                numeric.readout.resonators.r2.frequency_ghz,
            ]
        )
        chi = model.frequencies_ghz - bare
        for index, batch in enumerate(calls[2:]):
            first_state, second_state = divmod(index, 2)
            initial = np.zeros((1, 2, 2), complex)
            initial[0, second_state, second_state] = 1
            np.testing.assert_array_equal(batch.initial_densities, initial)
            fields = cavity_fields(
                model, pulses, np.array([0.0, 8.0]), initial_time_ns=0.0
            )
            memory = fields[-1, first_state]
            average = cavity_field_average(
                model, pulses, 8.0, 20.0, initial_field=memory, initial_time_ns=8.0
            )
            shift = batch.segment_shifts[0][0]
            np.testing.assert_allclose(
                shift.energy_shift_ghz,
                (chi * average.photons).sum(axis=-1),
                rtol=1e-10,
                atol=1e-15,
            )
        assert not np.array_equal(
            calls[2].segment_shifts[0][0].energy_shift_ghz,
            calls[4].segment_shifts[0][0].energy_shift_ghz,
        )
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
