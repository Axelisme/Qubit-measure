"""Routing-facts allowances and independent source physics through Engine."""

from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.six_fluxonium.runtime import SixFluxoniumSource
from zcu_tools.resources.six_fluxonium import parse_six_fluxonium
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
def test_routing_budget_is_divided_and_never_authorizes_another_source(total, retained):
    key = b"complete-caller-content" * 100
    with backend(routing_cache_bytes=total) as owner:
        for single in owner.sources:
            src = source()
            single.engine.evolve(src, (batch(1),), cancel=Event(), reuse_key=key)
            assert single.engine.reuse(src, key) is retained
            assert single.engine.reuse(source(), key) is False


def test_independent_sources_and_model_keys_keep_distinct_physics_after_warmup():
    with backend(
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
            reference = Engine(1)
            try:
                result = reference.evolve(src, (batch(1),), cancel=Event())
                expected.append(result.densities.tobytes())
            finally:
                reference.close()
        assert len(set(expected)) == 6
        for _ in range(2):
            for single, src, truth in zip(
                owner.sources, sources, expected, strict=True
            ):
                # All engines see identical row keys, but circuit identities differ.
                actual = single.engine.evolve(src, (batch(1),), cancel=Event())
                assert actual.densities.tobytes() == truth
        # A different source must use its own circuit's rates.
        actual = owner.sources[0].engine.evolve(
            sources[-1], (batch(1),), cancel=Event()
        )
        assert actual.densities.tobytes() == expected[-1]
