"""Source-scoped deterministic readout reuse, separate from fresh sampling."""

from dataclasses import replace

import pytest
import scipy.linalg
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
)

pytestmark = pytest.mark.sim


def test_acquires_share_solver_and_exact_sums_but_not_sampling(monkeypatch):
    schur = scipy.linalg.schur
    builds = []

    def observed(*args, **kwargs):
        builds.append(1)
        return schur(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "schur", observed)
    source = FTFSource(settings(), DeviceManager())
    try:
        first = expect_statistics(
            source.open_acquire(request()).sample_round(snapshot(), 0)
        )
        initial_builds = len(builds)
        second = expect_statistics(
            source.open_acquire(replace(request(), sequence=8)).sample_round(
                snapshot(), 0
            )
        )
        assert initial_builds > 0
        assert len(builds) == initial_builds, (
            "identical acquires must reuse the physical solver"
        )
        assert (
            first.channels[0].conditional_sums.tobytes()
            == second.channels[0].conditional_sums.tobytes()
        )
        assert first.channels[0].sums.tobytes() != second.channels[0].sums.tobytes()
    finally:
        source.close()


def test_receiver_configuration_restored_to_exact_content_hits_old_solver():
    source = FTFSource(settings(), DeviceManager())
    try:
        first = expect_statistics(
            source.open_acquire(request()).sample_round(snapshot(), 0)
        )
        assert hasattr(source, "readout_cache"), (
            "source must retain exact receiver products"
        )
        original = source.readout
        source.readout = replace(
            original,
            pulse=replace(original.pulse, gain=2 * original.pulse.gain),
        )
        source.open_acquire(request()).sample_round(snapshot(), 0)
        assert source.readout_cache.builds == 2
        source.readout = original
        restored = expect_statistics(
            source.open_acquire(request()).sample_round(snapshot(), 0)
        )
        assert source.readout_cache.builds == 2
        assert source.preparation_cache.hits == 1
        assert (
            first.channels[0].conditional_sums.tobytes()
            == restored.channels[0].conditional_sums.tobytes()
        )
    finally:
        source.close()


@pytest.mark.parametrize("change", ["snapshot", "settings"])
def test_source_generation_changes_drop_solver(change):
    source = FTFSource(settings(), DeviceManager())
    try:
        source.open_acquire(request()).sample_round(snapshot(), 0)
        assert hasattr(source, "readout_cache"), "source must own bounded readout reuse"
        assert source.readout_cache.builds == 1
        source.open_acquire(request()).sample_round(snapshot(), 0)
        assert source.readout_cache.builds == 1
        changed = snapshot()
        if change == "snapshot":
            changed = replace(changed, model_identity="new-calibration")
        else:
            source.settings = source.settings.model_copy(
                update={
                    "numerics": source.settings.numerics.model_copy(
                        update={"seed": 456}
                    )
                }
            )
        source.open_acquire(request()).sample_round(changed, 0)
        assert source.readout_cache.builds == 2
        source.close()
        assert source.readout_cache.resident_bytes == 0
    finally:
        source.close()


def test_new_template_reuses_readout_solver_after_preparation_miss():
    source = FTFSource(settings(), DeviceManager())
    try:
        source.open_acquire(request()).sample_round(snapshot(), 0)
        # A different reps count needs another sampling plan, not another solver.
        source.open_acquire(replace(request(), reps=4, loop_dims=(4, 2))).sample_round(
            snapshot(), 0
        )
        assert source.readout_cache.builds == 1
        assert source.readout_cache.hits == 1
    finally:
        source.close()
