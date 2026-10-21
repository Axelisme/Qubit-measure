"""Acquire composition advances density and cavity memory at the public seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import PulseEnvelope
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource, plan_template
from zcu_tools.simulate.adc import cavity_field_average, cavity_fields
from zcu_tools.simulate.evolution import Engine, segment_spans

from tests.experiment._ftf_templates import request, settings, snapshot, template

pytestmark = pytest.mark.sim


class ObservingEngine(Engine):
    """Delegate numerical evolution unchanged and observe its public boundary."""

    def __init__(self):
        super().__init__(1, 8 * 1024**2)
        self.calls = []

    def evolve(self, source, batches, **kwargs):
        result = super().evolve(source, batches, **kwargs)
        self.calls.append((source, tuple(batches), result))
        return result


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "shaped"])
def test_acquire_continues_each_point_span_density_field_and_common_origin(shaped):
    engine = ObservingEngine()
    source = FTFSource(settings().with_telemetry(enabled=False), DeviceManager(), engine=engine)
    base = template(tones=1)
    earlier = replace(
        base.events[-1], start_us=np.zeros(2), duration_us=np.array([.002, .006]),
        gain=np.array([.3, .5]),
        envelopes=(PulseEnvelope((0., .001, .006), (0j, 1j, .3)),) if shaped else (None,),
    )
    driven = replace(base, events=(earlier, *base.events))
    try:
        plan = plan_template(source.settings, snapshot(), driven, facts=source.model_facts(snapshot()))
        spans = segment_spans(plan.batch)
        acquire = request(templates=(driven,))
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == max(map(len, spans)), "cavity drives must evolve spans sequentially, not one vacuum-start timeline"
        facts = source.model_facts(snapshot())
        model = facts.readout
        chi = model.frequencies_ghz - np.array([r.frequency_ghz for r in (source.settings.readout.resonators.r1, source.settings.readout.resonators.r2)])
        memory = np.zeros((2, 2), complex)
        density = np.tile(facts.source.initial_density, (2, 1, 1))
        for step, (numeric_source, batches, result) in enumerate(engine.calls):
            assert numeric_source is facts.source
            assert len(batches) == 1
            batch = batches[0]
            np.testing.assert_array_equal(batch.points, plan.batch.points)
            np.testing.assert_array_equal(batch.flux.level, plan.batch.flux.level)
            np.testing.assert_array_equal(batch.flux.start_ns, plan.batch.flux.start_ns)
            for actual, original in zip(batch.tracks, plan.batch.tracks, strict=True):
                for name in ("start_ns", "duration_ns", "phase_rad", "frequency_ghz", "amplitude"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))
            np.testing.assert_array_equal(batch.initial_densities, density)
            for point, point_spans in enumerate(spans):
                start, end = point_spans[min(step, len(point_spans) - 1)]
                if step >= len(point_spans):
                    start = end
                assert batch.start_ns[point] == start
                assert batch.end_ns[point] == end
                assert batch.segment_boundaries[point] == tuple(t for t in plan.batch.segment_boundaries[point] if t <= end)
                if start == end:
                    assert not batch.segment_shifts[point]
                    continue
                shift, = batch.segment_shifts[point]
                assert (shift.start_ns, shift.end_ns) == (start, end)
                expected = cavity_field_average(model, plan.cavity_drives[point], start, end, initial_field=memory[point])
                np.testing.assert_allclose(shift.energy_shift_ghz, (chi * expected.photons).sum(axis=-1), rtol=1e-10, atol=1e-15)
                fields = cavity_fields(model, plan.cavity_drives[point], np.array([start, end]), initial_field=memory[point])
                branch = np.argmax(np.diag(density[point]).real)
                memory[point] = fields[-1, branch]
            density = result.densities
        # Equal-content acquire checks every span's route and does not replay it.
        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 1)
        assert len(engine.calls) == max(map(len, spans))
        assert source.preparation_cache.hits == 1
    finally:
        source.close()


def test_mixed_acquire_keeps_undriven_templates_on_the_single_evolve_path():
    engine = ObservingEngine()
    source = FTFSource(settings().with_telemetry(enabled=False), DeviceManager(), engine=engine)
    base = template()
    earlier = replace(base.events[-1], start_us=np.zeros(2), duration_us=np.full(2, .002), envelopes=(None,))
    driven = replace(base, events=(earlier, *base.events))
    undriven = replace(base, points=np.array([2, 3]))
    acquire = replace(request(templates=(driven, undriven)), point_dims=(4,), loop_dims=(3, 4))
    try:
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        plain_batches = [batch for _, batches, _ in engine.calls for batch in batches if 2 in batch.points]
        assert len(plain_batches) == 1, "undriven templates retain one full evolution even in mixed acquires"
        assert plain_batches[0].start_ns is None
        assert plain_batches[0].initial_densities is None
        assert source.evaluations[0].paths.points_by_class
        for name, members in source.evaluations[0].paths.point_members.items():
            assert source.evaluations[0].paths.points_by_class[name] == len(members)
    finally:
        source.close()


def test_no_cavity_acquire_keeps_single_evolve_without_continuation():
    engine = ObservingEngine()
    source = FTFSource(settings().with_telemetry(enabled=False), DeviceManager(), engine=engine)
    try:
        acquire = request()
        source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == 1
        batch, = engine.calls[0][1]
        assert batch.start_ns is None
        assert batch.initial_densities is None
        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 1)
        assert len(engine.calls) == 1
    finally:
        source.close()
