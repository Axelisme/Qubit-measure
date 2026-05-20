"""Tests for QtProgressBar / QtProgressBarFactory and FakeFreqAdapter pbar integration."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    app = QApplication.instance() or QApplication([])
    return app


# ---------------------------------------------------------------------------
# _ProgressStack helpers
# ---------------------------------------------------------------------------


def _make_stack(qapp):  # noqa: ARG001
    from zcu_tools.gui.ui.main_window import _ProgressStack

    return _ProgressStack()


# ---------------------------------------------------------------------------
# QtProgressBarFactory — main-thread smoke test
# ---------------------------------------------------------------------------


def test_factory_push_and_pop_leave_false(qapp):
    """leave=False: close() pops the bar immediately."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="test", total=10, leave=False)
    QApplication.processEvents()
    assert len(stack._active) == 1  # bar was pushed

    pbar.update(3)
    QApplication.processEvents()
    assert pbar.n == 3

    pbar.close()
    QApplication.processEvents()
    assert len(stack._active) == 0  # bar was popped


def test_factory_push_and_pop_leave_true(qapp):
    """leave=True: close() leaves bar visible; reset_all() clears it."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="test", total=10, leave=True)
    QApplication.processEvents()
    assert len(stack._active) == 1

    pbar.close()
    QApplication.processEvents()
    assert len(stack._active) == 1  # still active (leave=True)

    stack.reset_all()
    assert len(stack._active) == 0


def test_factory_two_layers(qapp):
    """Two nested pbars (leave=False): inner pops on close, outer stays until reset."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    outer = factory(desc="outer", total=5, leave=True)
    QApplication.processEvents()
    inner = factory(desc="inner", total=10, leave=False)
    QApplication.processEvents()
    assert len(stack._active) == 2

    inner.close()
    QApplication.processEvents()
    assert len(stack._active) == 1  # inner popped (leave=False)

    outer.close()
    QApplication.processEvents()
    assert len(stack._active) == 1  # outer stays (leave=True)

    stack.reset_all()
    assert len(stack._active) == 0


def test_total_setter(qapp):
    """Setting total updates the QProgressBar maximum."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="t", total=5, leave=False)
    QApplication.processEvents()
    pbar.total = 20
    QApplication.processEvents()
    assert stack._active[0].maximum() == 20

    pbar.close()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# FakeFreqAdapter pbar integration — use_pbar_factory with QtProgressBarFactory
# ---------------------------------------------------------------------------


def test_fake_freq_adapter_run_with_qt_pbar(qapp):
    """FakeFreqAdapter.run() completes; leave=True outer bar stays, reset_all clears."""
    from unittest.mock import MagicMock

    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.experiment.v2_gui.adapters.onetone.freq import FakeFreqAdapter
    from zcu_tools.gui.adapter import ExpContext
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory
    from zcu_tools.progress_bar.interface import use_pbar_factory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    ctx = ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)
    adapter = FakeFreqAdapter()
    schema = adapter.make_default_cfg(ctx)

    # Override to small values so the test is fast
    from zcu_tools.gui.adapter import ScalarField, SweepField

    schema.root.fields["rounds"] = ScalarField(value=2, label="Rounds", type=int)
    schema.root.fields["freq"] = SweepField(
        start=5800.0, stop=5808.0, expts=5, label="Freq (MHz)"
    )

    with use_pbar_factory(factory):
        freqs, signals = adapter.run(ctx, schema)
        QApplication.processEvents()

    assert len(freqs) == 5
    assert len(signals) == 5
    # run_task uses leave=True pbar; reset_all clears it
    stack.reset_all()
    assert len(stack._active) == 0
