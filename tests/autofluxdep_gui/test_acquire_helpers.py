from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.utils import estimate_snr
from zcu_tools.gui.app.autofluxdep.nodes import acquire as acquire_mod
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult


def test_snr_stop_checker_waits_until_probe_has_value(monkeypatch: Any):
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

    checkers = acquire_mod.build_stop_checkers(
        env,
        probe,
        lambda signals: np.asarray(signals, dtype=np.complex128).real,
    )

    assert len(checkers) == 1
    assert checkers[0]() is False
    assert calls["count"] == 0

    probe.value = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    assert checkers[0]() is True
    assert calls["count"] == 1


def test_make_on_round_records_current_snr():
    result = Sweep1DResult.allocate(
        np.array([0.0]),
        np.arange(8, dtype=np.float64),
        x_label="x",
    )
    probe = acquire_mod.SnrProbe()
    notified: list[int] = []
    rounds: list[int] = []
    real = np.array([0.0, 0.2, 0.7, 1.0, 0.8, 0.3, 0.1, 0.0], dtype=np.float64)
    avg_d = [[np.column_stack([real, np.zeros_like(real)])]]

    on_round = acquire_mod.make_on_round(
        result,
        0,
        lambda signals: np.asarray(signals, dtype=np.complex128).real,
        notified.append,
        probe=probe,
        round_progress_hook=rounds.append,
    )

    on_round(3, avg_d)

    np.testing.assert_allclose(result.signal[0], real)
    assert result.snr[0] == pytest.approx(estimate_snr(real))
    assert probe.snr == pytest.approx(result.snr[0])
    assert notified == [0]
    assert rounds == [3]
