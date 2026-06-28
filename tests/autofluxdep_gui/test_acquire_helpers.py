from __future__ import annotations

from typing import Any

import numpy as np
from zcu_tools.gui.app.autofluxdep.nodes import acquire as acquire_mod
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder


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
        lambda signals: np.asarray(signals).real,
    )

    assert len(checkers) == 1
    assert checkers[0]() is False
    assert calls["count"] == 0

    probe.value = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    assert checkers[0]() is True
    assert calls["count"] == 1
