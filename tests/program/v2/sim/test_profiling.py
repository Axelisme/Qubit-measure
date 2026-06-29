from __future__ import annotations

import logging

from zcu_tools.program.v2.sim._profiling import (
    PROFILE_ENV,
    PerfStats,
    profiling_enabled,
)


def test_sim_profiling_enabled_by_env(monkeypatch) -> None:
    logger = logging.getLogger("tests.program.v2.sim.profiling.env")
    logger.setLevel(logging.INFO)
    monkeypatch.setenv(PROFILE_ENV, "1")

    assert profiling_enabled(logger)


def test_sim_perf_stats_logs_slow_when_enabled(monkeypatch, caplog) -> None:
    logger = logging.getLogger("tests.program.v2.sim.profiling.stats")
    logger.setLevel(logging.INFO)
    monkeypatch.setenv(PROFILE_ENV, "1")
    stats = PerfStats("unit", logger, interval_s=0.0, slow_ms=1.0)

    with caplog.at_level(logging.INFO, logger=logger.name):
        stats.record(2.0, detail="phase=x")

    messages = [record.getMessage() for record in caplog.records]
    assert any("autofluxdep profile slow unit" in msg for msg in messages)
    assert any("thread=" in msg for msg in messages)
    assert any("phase=x" in msg for msg in messages)
    assert any("autofluxdep profile unit: count=1" in msg for msg in messages)
