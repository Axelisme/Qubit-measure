"""Tests for autofluxdep profiling probes."""

from __future__ import annotations

import logging

from zcu_tools.gui.app.autofluxdep.profiling import (
    PROFILE_ENV,
    PerfStats,
    profiling_enabled,
)


def test_profiling_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    logger = logging.getLogger("tests.autofluxdep.profile.disabled")
    logger.setLevel(logging.INFO)
    stats = PerfStats("unit", logger, interval_s=0.0, slow_ms=1.0)

    with caplog.at_level(logging.INFO, logger=logger.name):
        stats.record(2.0, detail="node=x")

    assert profiling_enabled(logger) is False
    assert "autofluxdep profile" not in caplog.text


def test_profiling_env_enables_slow_and_summary_logs(monkeypatch, caplog):
    monkeypatch.setenv(PROFILE_ENV, "1")
    logger = logging.getLogger("tests.autofluxdep.profile.enabled")
    logger.setLevel(logging.INFO)
    stats = PerfStats("unit", logger, interval_s=0.0, slow_ms=1.0)

    with caplog.at_level(logging.INFO, logger=logger.name):
        stats.record(2.0, queue_ms=3.0, detail="node=x")

    assert profiling_enabled(logger) is True
    assert "autofluxdep profile slow unit" in caplog.text
    assert "duration_ms=2.0" in caplog.text
    assert "thread=" in caplog.text
    assert "queue_ms=3.0" in caplog.text
    assert "autofluxdep profile unit: count=1" in caplog.text
