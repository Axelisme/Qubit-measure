"""Regression test for the overnight-T1 liveplot env bug (commit 42290169).

Both the overnight-T1 and the overnight-singleshot-T1 ``update_plotter``
callbacks used to read ``ctx.env_dict[...]``, but the run context
(``TaskState``) only exposes ``env``. On the first liveplot tick this raised
``AttributeError`` -- uncovered by any test. These tests pin the seam: drive
``update_plotter`` with a *real* ``TaskState`` (whose ``env`` is populated and
which has no ``env_dict`` attribute at all) and assert the callback reads
``ctx.env``. Against the pre-fix code (``ctx.env_dict``) they fail with
``AttributeError``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from zcu_tools.experiment.v2.overnight.singleshot.t1 import (
    T1PlotAndSaveMixin as SingleshotT1Mixin,
)
from zcu_tools.experiment.v2.overnight.t1 import T1PlotAndSaveMixin as OvernightT1Mixin
from zcu_tools.experiment.v2.runner import TaskState


class _RecordingPlotter:
    """Liveplot double that records the args of each ``update`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def _make_ctx(env: dict[str, Any]) -> TaskState[Any, Any, Any]:
    """A real run context: populated ``env`` and provably no ``env_dict``.

    ``cfg`` / ``root_data`` are irrelevant to ``update_plotter`` (it only touches
    ``ctx.env``), so minimal stand-ins are fine.
    """
    ctx: TaskState[Any, Any, Any] = TaskState(root_data={}, cfg=None, env=env)
    assert not hasattr(ctx, "env_dict"), (
        "TaskState must not expose env_dict; the seam under test reads ctx.env"
    )
    return ctx


def test_overnight_t1_update_plotter_reads_ctx_env() -> None:
    mixin = object.__new__(OvernightT1Mixin)  # __init__ pulls in hardware setup
    iters = np.array([0, 1, 2], dtype=np.int64)
    ctx = _make_ctx({"iters": iters})

    plotter = _RecordingPlotter()
    plotters = {"t1": plotter}  # T1PlotDict at runtime; doubled for the seam test

    lengths = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    # signals: (iters, times) complex; t1_overnight_signal2real handles NaN-free.
    signals = (np.arange(3 * 4, dtype=np.float64).reshape(3, 4) + 1j).astype(
        np.complex128
    )
    results = {"lengths": lengths, "signals": signals}

    mixin.update_plotter(cast(Any, plotters), ctx, results)

    assert len(plotter.calls) == 1
    pos_args, _ = plotter.calls[0]
    # First positional arg is the iters pulled from ctx.env.
    assert pos_args[0] is iters


def test_singleshot_t1_update_plotter_reads_ctx_env() -> None:
    mixin = object.__new__(SingleshotT1Mixin)
    iters = np.array([0, 1], dtype=np.int64)
    repeat_idx = 1
    ctx = _make_ctx({"iters": iters, "repeat_idx": repeat_idx})

    plotters = {
        "populations_go": _RecordingPlotter(),
        "populations_eo": _RecordingPlotter(),
        "current_g": _RecordingPlotter(),
        "current_e": _RecordingPlotter(),
    }

    n_iters, n_times = 2, 4
    # update_plotter reads lengths = results["lengths"][0], i.e. a 1D times row.
    lengths = np.linspace(0.0, 3.0, n_times, dtype=np.float64)[np.newaxis, :]
    # populations: (iters, 2, times, 2) -> calc_populations -> (..., 3)
    populations = np.random.default_rng(0).random((n_iters, 2, n_times, 2))
    results = {"lengths": lengths, "populations": populations}

    mixin.update_plotter(plotters, ctx, results)

    # All four plotters get a tick; current_* use repeat_idx from ctx.env (must
    # not raise IndexError, proving repeat_idx came from env not env_dict).
    for p in plotters.values():
        assert len(p.calls) == 1
