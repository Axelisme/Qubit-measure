"""Physical model reuse at the acquire/engine and numeric-builder boundaries."""

import weakref
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource, replay
from zcu_tools.experiment.ftf import template as template_module
from zcu_tools.simulate.evolution import Engine, ModelSource
from zcu_tools.simulate.ftf import JointModel

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class ObservedEngine(Engine):
    """Delegate real evolution without keeping the physical model alive."""

    def __init__(self):
        super().__init__(1, 1_000_000)
        self.sources: list[weakref.ReferenceType[ModelSource]] = []

    def evolve(
        self, source, batches, *, budget=1e-3, cancel, telemetry=False, reuse_key=None
    ):
        self.sources.append(weakref.ref(source))
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
        )


@pytest.fixture
def observed_source():
    engine = ObservedEngine()
    source = FTFSource(settings(), DeviceManager(), engine=engine)
    # Isolate physical facts; test_prepared_reuse covers the outer preparation cache.
    source.preparation_cache.max_bytes = 0
    try:
        yield source, engine
    finally:
        source.close()


def test_equal_snapshot_acquires_reuse_model_and_close_releases_it(observed_source):
    source, engine = observed_source
    first = expect_statistics(
        source.open_acquire(request()).sample_round(snapshot(), 0)
    )
    second = expect_statistics(
        source.open_acquire(request()).sample_round(snapshot(), 0)
    )
    assert engine.sources[0]() is not None
    assert engine.sources[1]() is engine.sources[0]()
    np.testing.assert_array_equal(first.channels[0].sums, second.channels[0].sums)
    source.close()
    assert engine.sources[0]() is None


@pytest.mark.parametrize(
    "change", ["values", "identity", "version", "settings", "numeric"]
)
def test_full_physical_key_invalidates_and_only_last_snapshot_is_retained(
    observed_source, monkeypatch, change
):
    source, engine = observed_source
    original = snapshot()
    source.open_acquire(request()).sample_round(original, 0)
    # Establish the missing reuse behavior before checking invalidation.
    source.open_acquire(request()).sample_round(original, 0)
    assert engine.sources[0]() is engine.sources[1]() is not None
    changed = original
    if change == "values":
        changed = snapshot(0.21)
    elif change == "identity":
        changed = replace(original, model_identity="other-calibration")
    elif change == "version":
        changed = replace(original, version="other-snapshot-revision")
    elif change == "settings":
        source.settings = source.settings.model_copy(
            update={
                "numerics": source.settings.numerics.model_copy(update={"seed": 456})
            }
        )
    else:
        monkeypatch.setattr(replay, "SEGMENT_NUMERIC_VERSION", "other-numeric")
    source.open_acquire(request()).sample_round(changed, 0)
    assert engine.sources[2]() is not None
    assert engine.sources[0]() is None
    source.open_acquire(request()).sample_round(original, 0)
    if change in ("values", "identity", "version"):
        assert engine.sources[2]() is None


def test_multiple_templates_build_one_joint_for_one_evolve(
    observed_source, monkeypatch
):
    source, engine = observed_source
    builds = []

    class ObservedJoint(JointModel):
        def __init__(self, *args, **kwargs):
            builds.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(template_module, "JointModel", ObservedJoint)
    base = template()
    templates = tuple(
        replace(
            base,
            points=np.array([point], dtype=np.int64),
            events=tuple(
                replace(
                    event,
                    start_us=event.start_us[point : point + 1],
                    duration_us=event.duration_us[point : point + 1],
                    frequency_mhz=event.frequency_mhz[point : point + 1],
                    phase_deg=event.phase_deg[point : point + 1],
                    gain=event.gain[point : point + 1],
                    envelope_index=event.envelope_index[point : point + 1],
                    active=event.active[point : point + 1],
                )
                for event in base.events
            ),
            readouts=tuple(
                replace(
                    window,
                    trigger_us=window.trigger_us[point : point + 1],
                    frequency_mhz=window.frequency_mhz[point : point + 1],
                    phase_deg=window.phase_deg[point : point + 1],
                    generator_channels=window.generator_channels[point : point + 1],
                )
                for window in base.readouts
            ),
            duration_us=base.duration_us[point : point + 1],
        )
        for point in range(2)
    )
    source.open_acquire(request(templates=templates)).sample_round(snapshot(), 0)
    assert len(engine.sources) == 1
    assert len(builds) == 1
