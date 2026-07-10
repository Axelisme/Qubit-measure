from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.nodes import acquire as acquire_mod
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder


def test_run_env_knob_reads_run_start_snapshot() -> None:
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        node_name="qubit_freq",
        knobs_snapshot={"acquire_retry": 2},
    )

    assert env.knob("acquire_retry") == 2
    assert env.knob("missing", "fallback") == "fallback"
    with pytest.raises(KeyError, match="missing"):
        env.knob("missing")

    mutable_copy = env.knobs()
    mutable_copy["acquire_retry"] = 99
    assert env.knob("acquire_retry") == 2


def test_snr_stop_condition_waits_until_probe_has_value(monkeypatch: Any):
    calls = {"count": 0}

    def fake_snr_checker(ctx: Any, snr_threshold: float | None, signal2real_fn: Any):
        assert snr_threshold == 50.0

        def check() -> bool:
            calls["count"] += 1
            assert ctx.value is not None
            signal2real_fn(ctx.value)
            return True

        return check

    monkeypatch.setattr(acquire_mod, "snr_checker", fake_snr_checker)
    probe = acquire_mod.SnrProbe()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
    )

    stop_condition = acquire_mod.build_stop_condition(
        env,
        probe,
        lambda signals: np.asarray(signals, dtype=np.complex128).real,
    )

    assert stop_condition is not None
    assert stop_condition() is False
    assert calls["count"] == 0

    probe.value = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    assert stop_condition() is True
    assert calls["count"] == 1


def test_snr_stop_condition_uses_run_start_knob_snapshot() -> None:
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        knobs_snapshot={"earlystop_snr": None},
    )

    stop_condition = acquire_mod.build_stop_condition(
        env,
        acquire_mod.SnrProbe(),
        lambda signals: np.asarray(signals, dtype=np.complex128).real,
    )

    assert stop_condition is None


def test_acquire_retry_reads_default_and_validates_non_negative():
    schema = QubitFreqBuilder().make_default_schema()
    env = RunEnv(flux=0.0, flux_idx=0, schema=schema, knobs_snapshot={})
    assert acquire_mod.acquire_retry(env) == acquire_mod.DEFAULT_ACQUIRE_RETRY

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 0},
    )
    assert acquire_mod.acquire_retry(env) == 0

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 2},
    )
    assert acquire_mod.acquire_retry(env) == 2

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        node_name="qubit_freq",
        knobs_snapshot={"acquire_retry": -1},
    )
    with pytest.raises(RuntimeError, match="acquire_retry must be non-negative"):
        acquire_mod.acquire_retry(env)
