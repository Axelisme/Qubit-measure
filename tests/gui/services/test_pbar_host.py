from __future__ import annotations

from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.pbar_host import ProgressFactory, ProgressModel


def test_progress_model_retains_progress_without_widget(qapp):
    model = ProgressModel()
    factory = ProgressFactory(model)
    pbar = factory(desc="Ramp value", total=2.0, leave=False)
    QCoreApplication.processEvents()

    pbar.update(1.0)
    pbar.refresh()  # force flush past throttle
    QCoreApplication.processEvents()

    # The model retains a live ProgressBarModel even with no widget attached.
    (live,) = model.models()
    assert live.qt_maximum() == 10000  # float total scaled to _FLOAT_SCALE
    assert live.qt_value() == 5000  # 1.0 / 2.0 → 50%
    assert "Ramp value" in live.format()

    pbar.close()
    QCoreApplication.processEvents()
    assert model.models() == ()


def test_live_model_computes_format_and_percent_on_query(qapp):
    """The live ProgressBarModel is the SSOT: format/percent are methods
    computed on query (worker forwards only raw n; main thread + this model
    do all derivation)."""
    model = ProgressModel()
    factory = ProgressFactory(model)
    pbar = factory(desc="Rounds", total=100, leave=False)
    QCoreApplication.processEvents()
    pbar.update(23)
    pbar.refresh()
    QCoreApplication.processEvents()

    (live,) = model.models()
    assert live.n == 23
    assert live.total == 100
    assert live.percent() == 23.0
    assert live.qt_maximum() == 100
    assert live.qt_value() == 23
    fmt = live.format()
    assert "Rounds" in fmt
    assert "%v/%m" in fmt  # int total → Qt %v/%m placeholders


def test_live_model_elapsed_advances_with_wall_clock(qapp, monkeypatch):
    """elapsed()/format() use wall-clock at read time, not a frozen value."""
    import zcu_tools.gui.pbar_host as ph

    clock = {"t": 1000.0}
    monkeypatch.setattr(ph.time, "monotonic", lambda: clock["t"])

    model = ProgressModel()
    factory = ProgressFactory(model)
    factory(desc="x", total=10, leave=False)  # pushed at t=1000
    QCoreApplication.processEvents()
    (live,) = model.models()
    assert live.elapsed() == 0.0

    clock["t"] = 1042.0  # 42s later
    assert live.elapsed() == 42.0  # recomputed live, no update needed
