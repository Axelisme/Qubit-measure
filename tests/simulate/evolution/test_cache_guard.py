"""Resident payload admission through Engine's injected kernel boundary."""

from contextlib import closing
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult

from ._engine import RecordingKernel, batch, source


class CountedGuardKernel(RecordingKernel):
    """Identity builder with an explicit payload domain and row admission."""

    guard_builds = True

    def __init__(self):
        super().__init__("t4")
        self.builds = 0
        self.admissions = 0
        self.domain_covers = True

    def guard_prepared(self, table, rows, context):
        self.builds += 1
        keys = self.prepare_keys(table, rows, context)
        prepared = self.prepare(keys, table, rows, context)
        return GuardResult(
            np.ones(len(rows), dtype=bool), (GuardPreparation(keys, prepared),)
        )

    def admit_prepared(self, prepared, table, rows, context):
        self.admissions += 1
        if not self.domain_covers:
            return None
        keys = self.prepare_keys(table, rows, context)
        return GuardResult(
            np.full(len(rows), self.accepts), (GuardPreparation(keys, prepared),)
        )


@pytest.mark.parametrize("timelines", [1, 2])
def test_resident_guard_payload_skips_build_but_rechecks_rows_and_domain(timelines):
    kernel = CountedGuardKernel()
    src = source()
    with closing(Engine(1, 4096, kernels=(kernel,))) as engine:
        engine.evolve(src, (batch(2),), cancel=Event())
        cold_builds = kernel.builds
        result = engine.evolve(src, (batch(timelines),), cancel=Event())
        assert result.paths.segments_by_kernel == {"t4": timelines}
        assert kernel.builds == cold_builds, "resident domain hit must not build"
        assert kernel.admissions > 0

        kernel.domain_covers = False
        engine.evolve(src, (batch(timelines),), cancel=Event())
        assert kernel.builds > cold_builds, "domain miss must retain ordinary build"
        rebuilt = kernel.builds

        kernel.domain_covers = True
        kernel.accepts = False
        with pytest.raises(RuntimeError, match="no valid kernel"):
            engine.evolve(src, (batch(timelines),), cancel=Event())
        assert kernel.builds == rebuilt, "row rejection is not a domain miss"
