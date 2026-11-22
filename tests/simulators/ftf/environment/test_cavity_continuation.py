"""Acquire composition advances density and cavity memory at the public seam."""

import re
import warnings
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationCallbacks
from zcu_tools.simulate.adc import (
    CavityAverageCache,
    cavity_field_average,
    cavity_fields,
)
from zcu_tools.simulate.boundary import PulseEnvelope
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    TimelineBatch,
    segment_spans,
)
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template
from zcu_tools.simulators.ftf.environment.numeric import resonators
from zcu_tools.simulators.ftf.physics import CavityDrive
from zcu_tools.simulators.ftf.physics.model import readout_model

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


def _prescribed_evolution(points, initial_densities):
    # Point 0 ties ground and excited; point 1 selects the excited branch.
    densities = np.zeros_like(initial_densities)
    densities[0, 0, 0] = densities[0, 1, 1] = 0.5
    densities[1, 1, 1] = 1.0
    return EvolutionResult(points, densities, PathDiagnostics())


def _assert_cavity_shift(shift, response):
    assert shift.energy_shift_ghz.tobytes() == response.stark_shift_ghz.tobytes()
    assert shift.dephasing_per_ns.tobytes() == response.dephasing_per_ns.tobytes()
    if shift.start_ns == 6:  # No drive remains; cavity memory supplies the shift.
        assert np.any(shift.energy_shift_ghz)


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "shaped"])
def test_acquire_continues_each_point_span_density_field_and_common_origin(
    shaped, cavity_span_owners
):
    engine = EngineDouble()
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
        facts = source.model_facts(snapshot())
        density = np.tile(facts.source.initial_density, (2, 1, 1))
        prescribed = _prescribed_evolution(plan.batch.points, density)
        engine.result = prescribed
        acquire = request(templates=(driven,))
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == max(map(len, spans)), (
            "cavity drives must evolve spans sequentially, not one vacuum-start timeline"
        )
        pair = resonators(source.settings)
        model = readout_model(facts.joint, pair)
        uncached = [
            CavityDrive(facts.joint, pair, pulses) for pulses in plan.cavity_drives
        ]
        memory = np.zeros((2, 2), complex)
        for step, (numeric_source, batches) in enumerate(engine.calls):
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
                response = uncached[point].average(
                    start, end, initial_field=memory[point], initial_time_ns=start
                )
                fields = cavity_fields(
                    model,
                    plan.cavity_drives[point],
                    np.array([start, end]),
                    initial_field=memory[point],
                    initial_time_ns=start,
                )
                # The first span uses the source's initial ground state.
                memory[point] = fields[-1, 0 if step == 0 else point]
                _assert_cavity_shift(shift, response)
            density = prescribed.densities
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


def test_projective_continuation_starts_from_vacuum():
    engine = EngineDouble()
    base, readout = herald_template()
    # Drive the cavity at the first herald, then project again before the terminal read.
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
        acquire = request(templates=(driven,))
        source.open_acquire(acquire).sample_round(snapshot(), 0)

        model = facts.readout
        chi = model.frequencies_ghz - [
            numeric.readout.resonators.r1.frequency_ghz,
            numeric.readout.resonators.r2.frequency_ghz,
        ]
        shifts = {}
        for reads, end in zip(plan.projective_reads, plan.batch.end_ns, strict=True):
            first, second = reads
            pulses = (*first.pulses, *second.pulses)
            start = second.start_ns
            prior_fields = cavity_fields(
                model,
                first.pulses,
                np.array([first.start_ns, start]),
                initial_time_ns=first.start_ns,
            )
            assert np.any(prior_fields[-1]), "the first herald must leave a field"
            vacuum = cavity_field_average(
                model, pulses, start, end, initial_time_ns=start
            )
            shifts[start, end] = (chi * vacuum.photons).sum(axis=-1)

        seen = set()
        for _, batches in engine.calls:
            for batch in batches:
                assert batch.start_ns is not None
                for row, bounds in enumerate(
                    zip(batch.start_ns, batch.end_ns, strict=True)
                ):
                    if bounds in shifts:
                        (shift,) = batch.segment_shifts[row]
                        np.testing.assert_allclose(
                            shift.energy_shift_ghz,
                            shifts[bounds],
                            rtol=1e-10,
                            atol=1e-15,
                        )
                        seen.add(bounds)
        assert seen == shifts.keys()
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
