"""Tests for QtProgressBar / QtProgressBarFactory and FakeFreqAdapter pbar integration."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


# ---------------------------------------------------------------------------
# _ProgressStack helpers
# ---------------------------------------------------------------------------


def _make_stack(qapp):  # noqa: ARG001
    from zcu_tools.gui.ui.progress_stack import ProgressStack

    return ProgressStack()


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


def test_reset_all_keeps_removed_bar_as_hidden_child(qapp):
    """Clearing a visible bar must not turn it into a top-level window."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    stack.show()
    factory = QtProgressBarFactory(stack)
    factory(desc="test", total=10, leave=True)
    QApplication.processEvents()
    bar = stack._active[0]
    assert bar.isVisible() is True

    stack.reset_all()
    QApplication.processEvents()

    assert bar.parent() is stack
    assert bar.isWindow() is False
    assert bar.isVisible() is False


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
    """Integer totals use raw value as max; float totals use _FLOAT_SCALE for proportional fill."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import _FLOAT_SCALE, QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    # integer total → max == raw value
    pbar = factory(desc="t", total=5, leave=False)
    QApplication.processEvents()
    assert stack._active[0].maximum() == 5

    # changing to another integer → max updates
    pbar.total = 20
    QApplication.processEvents()
    assert stack._active[0].maximum() == 20
    assert pbar.total == 20

    # changing to float → scaled max, bar can fill proportionally
    pbar.total = 3.14
    QApplication.processEvents()
    assert stack._active[0].maximum() == _FLOAT_SCALE

    pbar.close()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# FakeFreqAdapter pbar integration — use_pbar_factory with QtProgressBarFactory
# ---------------------------------------------------------------------------


def test_fake_freq_adapter_run_with_qt_pbar(qapp):
    """FakeFreqAdapter.run() completes; leave=True outer bar stays, reset_all clears."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter
    from zcu_tools.gui.adapter import ExpContext, RunRequest
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory
    from zcu_tools.progress_bar.interface import use_pbar_factory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)

    # Override to small values so the test is fast
    from zcu_tools.gui.adapter import DirectValue, SweepValue

    schema.value.fields["rounds"] = DirectValue(2)
    from zcu_tools.gui.adapter import CfgSectionValue

    schema.value.fields["sweep"] = CfgSectionValue(
        fields={"freq": SweepValue(start=5800.0, stop=5808.0, expts=5)}
    )

    with use_pbar_factory(factory):
        run_result = adapter.run(
            RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg), schema
        )
        QApplication.processEvents()

    assert len(run_result.freqs) == 5
    assert len(run_result.signals) == 5
    # run_task uses leave=True pbar; reset_all clears it
    stack.reset_all()
    assert len(stack._active) == 0


# ---------------------------------------------------------------------------
# disable=True behaviour
# ---------------------------------------------------------------------------


def test_disabled_pbar_does_not_push_to_stack(qapp):
    """factory(disable=True) must not add any bar to stack._active."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    factory(desc="hidden", total=10, leave=False, disable=True)
    QApplication.processEvents()
    assert len(stack._active) == 0


def test_disabled_pbar_methods_are_noop_no_exception(qapp):
    """All mutating methods on a disabled pbar must not raise and must not touch the stack."""
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="hidden", total=10, leave=False, disable=True)
    pbar.update(3)
    pbar.reset()
    pbar.refresh()
    pbar.set_description("new label")
    pbar.total = 20
    pbar.close()

    assert len(stack._active) == 0


def test_disabled_pbar_internal_state_updates(qapp):
    """Mutating methods on a disabled pbar still update internal state (no Qt signals)."""
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="hidden", total=10, disable=True)
    pbar.update(3)
    pbar.update(2)
    assert pbar.n == 5

    pbar.total = 20
    assert pbar.total == 20

    pbar.set_description("updated")
    assert pbar.desc == "updated"

    assert len(stack._active) == 0


def test_disabled_pbar_close_does_not_pop_stack(qapp):
    """close() on a disabled pbar (leave=False) must not affect the stack."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="hidden", total=5, leave=False, disable=True)
    pbar.close()
    QApplication.processEvents()
    assert len(stack._active) == 0


def test_explicit_disable_false_behaves_normally(qapp):
    """Explicitly passing disable=False must behave identically to the default."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    pbar = factory(desc="normal", total=5, leave=False, disable=False)
    QApplication.processEvents()
    assert len(stack._active) == 1

    pbar.update(2)
    assert pbar.n == 2

    pbar.close()
    QApplication.processEvents()
    assert len(stack._active) == 0


def test_disabled_and_enabled_pbars_coexist(qapp):
    """A disabled pbar alongside an enabled one must not interfere with the stack."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
    from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

    stack = _make_stack(qapp)
    factory = QtProgressBarFactory(stack)

    enabled = factory(desc="active", total=10, leave=True, disable=False)
    QApplication.processEvents()
    disabled = factory(desc="hidden", total=10, leave=False, disable=True)
    QApplication.processEvents()

    assert len(stack._active) == 1

    disabled.update(5)
    disabled.close()
    QApplication.processEvents()
    assert len(stack._active) == 1  # enabled still in stack; disabled close is no-op

    stack.reset_all()
    assert len(stack._active) == 0
    del enabled
