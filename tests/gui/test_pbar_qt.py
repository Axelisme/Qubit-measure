"""Tests for the pbar_host ProgressFactory/ProgressBar + context-local factory integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar
    from zcu_tools.gui.session.ui.progress_stack import ProgressStack

pytest.importorskip("qtpy")


# ---------------------------------------------------------------------------
# ProgressStack helpers
# ---------------------------------------------------------------------------


def _make_stack(qapp):
    from zcu_tools.gui.session.ui.progress_stack import ProgressStack

    return ProgressStack()


def _layout_bars(stack: ProgressStack) -> tuple[LightweightProgressBar, ...]:
    from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar

    layout = stack.layout()
    assert layout is not None
    bars: list[LightweightProgressBar] = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        assert item is not None
        widget = item.widget()
        assert isinstance(widget, LightweightProgressBar)
        bars.append(widget)
    return tuple(bars)


def _make_factory(stack, *, operation_id: int = 1, owner_id: str = "owner"):
    """Build a worker progress factory whose live bars render into ``stack``.

    Wires the real Qt marshal (QtProgressTransport) → ProgressService → an
    owner-attached listener that re-renders the stack on every change, mirroring
    how a real View attaches. Returns the worker-side factory.
    """
    from zcu_tools.gui.session.adapters.qt_progress_transport import (
        QtProgressTransport,
    )
    from zcu_tools.gui.session.services.progress import ProgressService

    svc = ProgressService(QtProgressTransport())
    svc.attach_by_owner(
        owner_id,
        lambda: stack.render_models(tuple(m for _, m in svc.bars_for_owner(owner_id))),
    )
    return svc.make_factory(operation_id, owner_id=owner_id)


# ---------------------------------------------------------------------------
# ProgressFactory — main-thread smoke test
# ---------------------------------------------------------------------------


def test_lightweight_progress_bar_format_and_value(
    qapp, monkeypatch: pytest.MonkeyPatch
):
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QPainter, QPixmap
    from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar

    bar = LightweightProgressBar()
    try:
        bar.setMaximum(10)
        bar.setValue(3)
        bar.setFormat("rounds %v/%m %p%")

        assert bar.maximum() == 10
        assert bar.value() == 3
        assert bar.format() == "rounds %v/%m %p%"

        bar.resize(bar.fontMetrics().horizontalAdvance("rounds 3/10 30%") + 100, 30)
        pixmap = QPixmap(bar.size())
        captured_texts: list[str | None] = []
        real_draw_text = QPainter.drawText

        def observe_draw_text(
            painter: QPainter, rectangle: QRect, flags: int, text: str | None
        ) -> QRect | None:
            """Record Qt's text while preserving the real drawing call and result."""
            captured_texts.append(text)
            return real_draw_text(painter, rectangle, flags, text)

        with monkeypatch.context() as patch:
            patch.setattr(QPainter, "drawText", observe_draw_text)
            bar.render(pixmap)
        assert captured_texts == ["rounds 3/10 30%"]

        bar.setValue(99)
        assert bar.value() == 10
    finally:
        bar.deleteLater()
        qapp.processEvents()


def test_factory_push_and_pop_leave_false(qapp):
    """leave=False: close() pops the bar immediately."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="test", total=10, leave=False)
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1  # bar was pushed

    pbar.update(3)
    QApplication.processEvents()
    assert pbar.n == 3

    pbar.set_progress(7)
    QApplication.processEvents()
    assert pbar.n == 7

    pbar.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 0  # bar was popped


def test_reset_all_keeps_removed_bar_as_hidden_child(qapp):
    """Clearing a visible bar must not turn it into a top-level window."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    stack.show()
    factory = _make_factory(stack)
    factory(desc="test", total=10, leave=True)
    QApplication.processEvents()
    bar = _layout_bars(stack)[0]
    assert bar.isVisible() is True

    stack.reset_all()
    QApplication.processEvents()

    assert bar.parent() is stack
    assert bar.isWindow() is False
    assert bar.isVisible() is False


def test_factory_push_and_pop_leave_true(qapp):
    """leave=True: close() leaves bar visible; reset_all() clears it."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="test", total=10, leave=True)
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1

    pbar.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1  # still active (leave=True)

    stack.reset_all()
    assert len(_layout_bars(stack)) == 0


def test_factory_two_layers(qapp):
    """Two nested pbars (leave=False): inner pops on close, outer stays until reset."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    outer = factory(desc="outer", total=5, leave=True)
    QApplication.processEvents()
    inner = factory(desc="inner", total=10, leave=False)
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 2

    inner.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1  # inner popped (leave=False)

    outer.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1  # outer stays (leave=True)

    stack.reset_all()
    assert len(_layout_bars(stack)) == 0


def test_total_setter(qapp):
    """Integer totals set the raw maximum; float progress fills proportionally."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    # integer total → max == raw value
    pbar = factory(desc="t", total=5, leave=False)
    try:
        QApplication.processEvents()
        assert _layout_bars(stack)[0].maximum() == 5

        # changing to another integer → max updates
        pbar.total = 20
        QApplication.processEvents()
        assert _layout_bars(stack)[0].maximum() == 20
        assert pbar.total == 20

        # changing to float → bar fills proportionally from zero to completion
        pbar.total = 3.14
        pbar.set_progress(0)
        pbar.refresh()
        QApplication.processEvents()
        widget = _layout_bars(stack)[0]
        assert widget.maximum() > 0
        assert widget.value() == 0

        pbar.set_progress(1.57)
        pbar.refresh()
        QApplication.processEvents()
        assert widget.value() / widget.maximum() == pytest.approx(0.5)

        pbar.set_progress(3.14)
        pbar.refresh()
        QApplication.processEvents()
        assert widget.value() == widget.maximum()
        assert pbar.total == 3.14
    finally:
        pbar.close()
        QApplication.processEvents()


def test_total_change_updates_widget_in_place_no_flicker(qapp):
    """Changing a live bar's ``total`` between sweep points must update the existing
    progress widget in place, never remove+re-add
    it. Regression: the total setter used to emit CREATE, which rebuilt the model
    and made render_models reset_all() the stack — the bar flickered out of view
    on every sweep point.
    """
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="rounds", total=10, leave=True)
    QApplication.processEvents()
    widget = _layout_bars(stack)[0]  # the specific progress widget instance shown

    # Mutating total many times (simulating per-sweep-point set_pbar_n) keeps the
    # *same* widget object in place — it is never popped back to the pool.
    for new_total in (20, 30, 40):
        pbar.total = new_total
        pbar.refresh()
        QApplication.processEvents()
        assert len(_layout_bars(stack)) == 1
        assert _layout_bars(stack)[0] is widget  # same widget, not rebuilt
        assert widget.maximum() == new_total

    pbar.close()
    QApplication.processEvents()


def test_progress_stack_skips_unchanged_bar_properties(
    qapp, monkeypatch: pytest.MonkeyPatch
):
    from zcu_tools.gui.session.pbar_host import ProgressBarModel

    stack = _make_stack(qapp)
    profile_calls: list[tuple[str, float, str]] = []
    stack.set_profile_callback(
        lambda label, duration_ms, detail: profile_calls.append(
            (label, duration_ms, detail)
        )
    )

    class _FixedFormatModel(ProgressBarModel):
        """Isolate live clock formatting across the four property-cache cases.

        ``display_format`` is the exact Qt format string returned by ``format``;
        all numeric state and Qt value/maximum derivation use the real model.
        """

        display_format: str = "rounds %v/%m"

        def format(self) -> str:
            """Return the controlled display string without reading the clock."""
            return self.display_format

    model = _FixedFormatModel(label="rounds", total=10, start_time=0.0)
    model.set_n(2)
    bar = stack.push()
    calls: list[tuple[str, int | str]] = []
    set_maximum = bar.setMaximum
    set_format = bar.setFormat
    set_value = bar.setValue

    def observe_maximum(value: int) -> None:
        """Record the maximum write, delegating its value, result and errors."""
        calls.append(("maximum", value))
        return set_maximum(value)

    def observe_format(value: str) -> None:
        """Record the format write, delegating its value, result and errors."""
        calls.append(("format", value))
        return set_format(value)

    def observe_value(value: int) -> None:
        """Record the value write, delegating its value, result and errors."""
        calls.append(("value", value))
        return set_value(value)

    monkeypatch.setattr(bar, "setMaximum", observe_maximum)
    monkeypatch.setattr(bar, "setFormat", observe_format)
    monkeypatch.setattr(bar, "setValue", observe_value)

    stack.render_models((model,))
    assert calls == [
        ("maximum", 10),
        ("format", "rounds %v/%m"),
        ("value", 2),
    ]

    calls.clear()
    stack.render_models((model,))
    assert calls == []
    assert profile_calls[-1][0] == "snapshot_noop"

    model.set_n(3)
    stack.render_models((model,))
    assert calls == [("value", 3)]

    calls.clear()
    model.set_total(20)
    model.display_format = "rounds %v/%m [0:01]"
    stack.render_models((model,))
    assert calls == [
        ("maximum", 20),
        ("format", "rounds %v/%m [0:01]"),
    ]
    assert any(label == "set_format" for label, _duration_ms, _detail in profile_calls)


# ---------------------------------------------------------------------------
# Caller progress integration — context-local factory with nested bars
# ---------------------------------------------------------------------------


def test_context_local_factory_routes_nested_progress(qapp):
    """A synthetic nested acquisition leaves only its retained outer bar."""
    from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar
    from zcu_tools.progress_bar.interface import make_pbar, use_pbar_factory

    stack = _make_stack(qapp)
    factory = _make_factory(stack)
    try:
        with use_pbar_factory(factory):
            outer = make_pbar(total=2, leave=True)
            try:
                for _ in range(2):
                    inner = make_pbar(total=5, leave=False)
                    try:
                        for _ in range(5):
                            inner.update()
                        assert inner.n == 5
                    finally:
                        inner.close()
                    outer.update()
                assert outer.n == 2
            finally:
                outer.close()
            qapp.processEvents()
        assert (
            sum(
                not bar.isHidden() for bar in stack.findChildren(LightweightProgressBar)
            )
            == 1
        )
        stack.reset_all()
        assert all(bar.isHidden() for bar in stack.findChildren(LightweightProgressBar))
    finally:
        stack.close()
        stack.deleteLater()


# ---------------------------------------------------------------------------
# disable=True behaviour
# ---------------------------------------------------------------------------


def test_disabled_pbar_does_not_push_to_stack(qapp):
    """factory(disable=True) must not add any bar to the stack layout."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    factory(desc="hidden", total=10, leave=False, disable=True)
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 0


def test_disabled_pbar_methods_are_noop_no_exception(qapp):
    """All mutating methods on a disabled pbar must not raise and must not touch the stack."""

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="hidden", total=10, leave=False, disable=True)
    pbar.update(3)
    pbar.reset()
    pbar.refresh()
    pbar.set_description("new label")
    pbar.total = 20
    pbar.close()

    assert len(_layout_bars(stack)) == 0


def test_disabled_pbar_internal_state_updates(qapp):
    """Mutating methods on a disabled pbar still update internal state (no Qt signals)."""

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="hidden", total=10, disable=True)
    pbar.update(3)
    pbar.update(2)
    assert pbar.n == 5
    pbar.set_progress(8)
    assert pbar.n == 8

    pbar.total = 20
    assert pbar.total == 20

    pbar.set_description("updated")
    assert pbar.desc == "updated"

    assert len(_layout_bars(stack)) == 0


def test_disabled_pbar_close_does_not_pop_stack(qapp):
    """close() on a disabled pbar (leave=False) must not affect the stack."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="hidden", total=5, leave=False, disable=True)
    pbar.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 0


def test_explicit_disable_false_behaves_normally(qapp):
    """Explicitly passing disable=False must behave identically to the default."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    pbar = factory(desc="normal", total=5, leave=False, disable=False)
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 1

    pbar.update(2)
    assert pbar.n == 2

    pbar.close()
    QApplication.processEvents()
    assert len(_layout_bars(stack)) == 0


def test_disabled_and_enabled_pbars_coexist(qapp):
    """A disabled pbar alongside an enabled one must not interfere with the stack."""
    from qtpy.QtWidgets import QApplication

    stack = _make_stack(qapp)
    factory = _make_factory(stack)

    enabled = factory(desc="active", total=10, leave=True, disable=False)
    QApplication.processEvents()
    disabled = factory(desc="hidden", total=10, leave=False, disable=True)
    QApplication.processEvents()

    assert len(_layout_bars(stack)) == 1

    disabled.update(5)
    disabled.close()
    QApplication.processEvents()
    assert (
        len(_layout_bars(stack)) == 1
    )  # enabled still in stack; disabled close is no-op

    stack.reset_all()
    assert len(_layout_bars(stack)) == 0
    del enabled
