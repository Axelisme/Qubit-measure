"""Environment-wide residency limits through public evolution and ADC seams."""

from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.six_fluxonium.runtime import SixFluxoniumSource
from zcu_tools.resources.six_fluxonium import parse_six_fluxonium
from zcu_tools.simulate import adc
from zcu_tools.simulate.evolution import Engine, ModelSource

from tests.resources._six_fluxonium_data import six_document
from tests.simulate.evolution._engine import batch, source


@contextmanager
def backend(**limits: int):
    document: dict[str, object] = {**six_document(), "numerics": limits}
    try:
        settings = parse_six_fluxonium(document)
    except ValueError as error:
        pytest.fail(
            f"six-source numerics must accept environment cache budgets: {error}"
        )
    devices = DeviceManager()
    owner = SixFluxoniumSource(settings, devices)
    try:
        yield owner
    finally:
        owner.close()
        devices.close_all_devices()


@pytest.mark.parametrize(("total", "retained"), [(6000, False), (600_000, True)])
def test_prepare_budget_is_divided_across_all_six_sources(total, retained):
    # A static three-level channel fits 6000 bytes, but not a 1000-byte share.
    with backend(
        prepare_cache_bytes=total, result_memo_bytes=0, routing_cache_bytes=0
    ) as owner:
        for single in owner.sources:
            src = source()
            first = single.engine.evolve(src, (batch(1),), cancel=Event())
            moved = replace(batch(1), points=np.array([1]))
            second = single.engine.evolve(src, (moved,), cancel=Event(), telemetry=True)
            assert second.paths.routing["t4"].cache_hits == int(retained)
            assert second.densities.tobytes() == first.densities.tobytes()


@pytest.mark.parametrize(("total", "retained"), [(6000, False), (600_000, True)])
def test_result_memo_has_its_own_divided_budget_without_prepare_residency(
    total, retained
):
    # One density result fits the total, but exceeds the small per-source share.
    with backend(
        prepare_cache_bytes=0, result_memo_bytes=total, routing_cache_bytes=0
    ) as owner:
        for single in owner.sources:
            src = source()
            first = single.engine.evolve(src, (batch(1),), cancel=Event())
            second = single.engine.evolve(
                src, (batch(1),), cancel=Event(), telemetry=True
            )
            assert second.paths.routing["t4"].cache_hits == int(retained)
            assert second.densities.tobytes() == first.densities.tobytes()


@pytest.mark.parametrize(("total", "retained"), [(6000, False), (600_000, True)])
def test_routing_budget_is_divided_and_never_authorizes_another_source(total, retained):
    key = b"complete-caller-content" * 100
    with backend(
        prepare_cache_bytes=0, result_memo_bytes=0, routing_cache_bytes=total
    ) as owner:
        for single in owner.sources:
            src = source()
            single.engine.evolve(src, (batch(1),), cancel=Event(), reuse_key=key)
            assert single.engine.reuse(src, key) is retained
            assert single.engine.reuse(source(), key) is False


def adc_batch():
    model = adc.ReadoutModel(
        "synthetic-cavity",
        ("g", "e"),
        np.array([[6.0], [6.01]]),
        6.0,
        np.array([0.02]),
        np.array([0.04]),
        np.zeros((2, 2)),
        mismatch_phase_rad=np.zeros(1),
    )
    return adc.ReadoutBatch(
        model,
        ((adc.ReadoutPulse(0, 10, 0.2, 6.0),),),
        ((adc.SampleGrid(np.arange(10, dtype=float), 6.0),),),
        np.array([0], dtype=np.int64),
        False,
    )


@pytest.mark.parametrize(
    ("limit", "retained"), [(0, False), (1000, False), (100_000, True)]
)
def test_adc_budget_applies_to_each_source_without_changing_responses(limit, retained):
    inputs = adc_batch()
    with backend(adc_cache_bytes=limit) as owner:
        for single in owner.sources:
            first = single.readout_cache.responses(inputs)
            second = single.readout_cache.responses(inputs)
            assert single.readout_cache.resident_bytes <= limit
            assert single.readout_cache.hits == int(retained)
            assert second.sums.tobytes() == first.sums.tobytes()


def test_independent_sources_and_model_keys_keep_distinct_physics_after_warmup():
    with backend(
        prepare_cache_bytes=600_000,
        result_memo_bytes=600_000,
        routing_cache_bytes=600_000,
    ) as owner:
        expected = []
        sources = []
        for index in range(6):
            base = source()
            model = replace(
                base.model((0,)),
                identity=f"circuit-{index}",
                rates_per_ns=base.model((0,)).rates_per_ns * (index + 1),
            )
            src = ModelSource(
                base.initial_density, lambda level, m=model: replace(m, level=level)
            )
            sources.append(src)
            with_uncached = Engine(1, 0)
            try:
                result = with_uncached.evolve(src, (batch(1),), cancel=Event())
                expected.append(result.densities.tobytes())
            finally:
                with_uncached.close()
        assert len(set(expected)) == 6
        for _ in range(2):
            for single, src, truth in zip(
                owner.sources, sources, expected, strict=True
            ):
                # All engines see identical row keys, but circuit identities differ.
                actual = single.engine.evolve(src, (batch(1),), cancel=Event())
                assert actual.densities.tobytes() == truth
        # Changing the model on an already warm engine must miss both caches.
        actual = owner.sources[0].engine.evolve(
            sources[-1], (batch(1),), cancel=Event()
        )
        assert actual.densities.tobytes() == expected[-1]
