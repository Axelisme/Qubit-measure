"""ProgressBar (worker) + ProgressBarModel (SSOT) via a synchronous transport.

No Qt here — the worker bar emits ``ProgressEvent``s through a
``DirectProgressTransport`` into a ``ProgressService``; the live
``ProgressBarModel`` is the single source of truth and computes format / percent
/ timing on query.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.services.progress import ProgressService

from ._progress_fakes import DirectProgressTransport


def _service_and_factory(operation_id: int = 1, owner_id: str = "tab-1"):
    svc = ProgressService(DirectProgressTransport())
    factory = svc.make_factory(operation_id, owner_id=owner_id)
    return svc, factory


def test_service_retains_progress_until_close():
    svc, factory = _service_and_factory()
    pbar = factory(desc="Ramp value", total=2.0, leave=False)

    pbar.update(1.0)
    pbar.refresh()  # force flush past throttle

    ((_, live),) = svc.bars_for_owner("tab-1")
    assert live.qt_maximum() == 10000  # float total scaled to _FLOAT_SCALE
    assert live.qt_value() == 5000  # 1.0 / 2.0 → 50%
    assert "Ramp value" in live.format()

    pbar.close()
    assert svc.bars_for_owner("tab-1") == ()


def test_live_model_computes_format_and_percent_on_query():
    svc, factory = _service_and_factory()
    pbar = factory(desc="Rounds", total=100, leave=False)
    pbar.update(23)
    pbar.refresh()

    ((_, live),) = svc.bars_for_owner("tab-1")
    assert live.n == 23
    assert live.total == 100
    assert live.percent() == 23.0
    assert live.qt_maximum() == 100
    assert live.qt_value() == 23
    fmt = live.format()
    assert "Rounds" in fmt
    assert "%v/%m" in fmt  # int total → Qt %v/%m placeholders


def test_live_model_elapsed_advances_with_wall_clock(monkeypatch):
    """elapsed()/format() use wall-clock at read time, not a frozen value."""
    import zcu_tools.gui.app.main.services.progress as prog

    clock = {"t": 1000.0}
    monkeypatch.setattr(prog.time, "monotonic", lambda: clock["t"])

    svc, factory = _service_and_factory()
    factory(desc="x", total=10, leave=False)  # created at t=1000

    ((_, live),) = svc.bars_for_owner("tab-1")
    assert live.elapsed() == 0.0

    clock["t"] = 1042.0  # 42s later
    assert live.elapsed() == 42.0  # recomputed live, no update needed
