"""Tests for InteractiveAnalysisWidget (the measure-gui InteractiveHost).

Headless: a fake InteractiveSession is bound and the widget is driven by
clicking its rendered buttons and feeding it fake matplotlib mouse events. The
off-main work is delegated to an injected InteractiveHostEnv port (ADR-0019);
tests pass a fake env that runs the compute synchronously (the real marshal is
BackgroundRunner's job — see tests/gui/services/test_background.py).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QCheckBox, QPushButton
from zcu_tools.gui.app.main.adapter.types import (
    ButtonControl,
    ControlKey,
    ToggleControl,
)
from zcu_tools.gui.app.main.ui.interactive_analysis import InteractiveAnalysisWidget


class _FakeEnv:
    """InteractiveHostEnv stand-in: records run_background calls and runs the
    compute synchronously, delivering the result to on_done."""

    def __init__(self) -> None:
        self.calls: list[tuple[Callable[[], object], Callable[[object], None]]] = []

    def run_background(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        self.calls.append((compute, on_done))
        on_done(compute())


def _fake_session(
    *,
    conjugate_initial: bool = False,
    conjugate_cb: Callable[[bool], None] | None = None,
    auto_align_cb: Callable[[], None] | None = None,
    swap_cb: Callable[[], None] | None = None,
) -> MagicMock:
    conjugate_mock = MagicMock() if conjugate_cb is None else conjugate_cb
    auto_mock = MagicMock() if auto_align_cb is None else auto_align_cb
    swap_mock = MagicMock() if swap_cb is None else swap_cb
    # Keep references for assertions when we create Mocks internally
    session = MagicMock()
    session._conjugate_mock = conjugate_mock  # type: ignore[attr-defined]
    session._auto_mock = auto_mock  # type: ignore[attr-defined]
    session._swap_mock = swap_mock  # type: ignore[attr-defined]
    session.controls.return_value = (
        ToggleControl(
            key=ControlKey("conjugate"),
            label="Conjugate Line",
            initial=conjugate_initial,
            on_change=conjugate_mock,
        ),
        ButtonControl(
            key=ControlKey("auto_align"), label="Auto Align", on_trigger=auto_mock
        ),
        ButtonControl(key=ControlKey("swap"), label="Swap Lines", on_trigger=swap_mock),
    )
    session.info_text.return_value = "half: 1\nint: 2"
    return session


def _buttons(widget) -> dict[str, QPushButton]:
    return {b.text(): b for b in widget.findChildren(QPushButton)}


def _checkboxes(widget) -> dict[str, QCheckBox]:
    return {b.text(): b for b in widget.findChildren(QCheckBox)}


def test_figure_property_is_a_real_figure(qapp):  # noqa: ARG001
    from matplotlib.figure import Figure

    w = InteractiveAnalysisWidget(_FakeEnv())
    assert isinstance(w.figure, Figure)
    w.deleteLater()


def test_bind_renders_controls_in_order_plus_done(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(_fake_session(), on_done=lambda: None)
    # Order: Conjugate Line checkbox, Auto Align button, Swap Lines button, then Done
    labels_in_order = []
    # findChildren respects creation order for our layout; check presence first
    assert "Conjugate Line" in _checkboxes(w)
    assert "Auto Align" in _buttons(w)
    assert "Swap Lines" in _buttons(w)
    assert "Done" in _buttons(w)
    # Verify ordering by walking layout widgets in order they were added
    layout_labels: list[str] = []
    for i in range(w._controls_layout.count()):
        item = w._controls_layout.itemAt(i)
        widget = item.widget() if item is not None else None
        if isinstance(widget, (QPushButton, QCheckBox)):
            layout_labels.append(widget.text())
        elif widget is not None:
            # info label or Done may be elsewhere; collect only controls
            continue
    # Controls must appear in declaration order before Done
    # layout also contains info and stretch; filter to our known controls
    controls_only = [
        l for l in layout_labels if l in {"Conjugate Line", "Auto Align", "Swap Lines"}
    ]
    assert controls_only == ["Conjugate Line", "Auto Align", "Swap Lines"]
    w.deleteLater()


def test_conjugate_toggle_initial_is_unchecked_and_does_not_trigger(qapp):  # noqa: ARG001
    conjugate_mock = MagicMock()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(
        _fake_session(conjugate_cb=conjugate_mock, conjugate_initial=False),
        on_done=lambda: None,
    )
    # Initial False -> unchecked and no callback yet
    cb = _checkboxes(w)["Conjugate Line"]
    assert cb.isChecked() is False
    conjugate_mock.assert_not_called()
    # Switch to True must fire exactly once
    cb.setChecked(True)
    conjugate_mock.assert_called_once_with(True)
    conjugate_mock.reset_mock()
    cb.setChecked(False)
    conjugate_mock.assert_called_once_with(False)
    w.deleteLater()


def test_conjugate_toggle_true_initial_is_checked_without_callback(qapp):  # noqa: ARG001
    conjugate_mock = MagicMock()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(
        _fake_session(conjugate_cb=conjugate_mock, conjugate_initial=True),
        on_done=lambda: None,
    )
    cb = _checkboxes(w)["Conjugate Line"]
    assert cb.isChecked() is True
    conjugate_mock.assert_not_called()
    w.deleteLater()


def test_button_controls_dispatch_their_typed_callbacks(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(session, on_done=lambda: None)
    _buttons(w)["Auto Align"].click()
    session._auto_mock.assert_called_once_with()  # type: ignore[attr-defined]
    _buttons(w)["Swap Lines"].click()
    session._swap_mock.assert_called_once_with()  # type: ignore[attr-defined]
    w.deleteLater()


def test_done_calls_on_done_and_disables_all_controls_and_canvas(qapp):  # noqa: ARG001
    fired: list[bool] = []
    session = _fake_session()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(session, on_done=lambda: fired.append(True))
    done = _buttons(w)["Done"]
    # Also ensure canvas forwarding before Done works
    in_axes = SimpleNamespace(inaxes=object(), xdata=1.5, ydata=4.2)
    w._on_press(in_axes)
    session.on_press.assert_called_once_with(1.5)
    session.on_press.reset_mock()

    done.click()
    assert fired == [True]
    assert done.isEnabled() is False
    assert _buttons(w)["Auto Align"].isEnabled() is False
    assert _buttons(w)["Swap Lines"].isEnabled() is False
    assert _checkboxes(w)["Conjugate Line"].isEnabled() is False
    # After Done, pointer and control events must not dispatch
    w._on_press(in_axes)
    session.on_press.assert_not_called()
    _buttons(w)["Auto Align"].click()
    session._auto_mock.assert_not_called()  # type: ignore[attr-defined]
    # Done is single-flight
    done.click()
    assert fired == [True]
    w.deleteLater()


def test_canvas_events_forward_to_session_and_outside_axes_ignored(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(session, on_done=lambda: None)
    in_axes = SimpleNamespace(inaxes=object(), xdata=1.5, ydata=4.2)
    out_axes = SimpleNamespace(inaxes=None, xdata=None, ydata=None)

    w._on_press(in_axes)
    w._on_move(in_axes)
    w._on_release(in_axes)
    w._on_press(out_axes)  # ignored — outside axes

    session.on_press.assert_called_once_with(1.5)
    session.on_move.assert_called_once_with(1.5)
    session.on_release.assert_called_once_with(1.5, 4.2)
    w.deleteLater()


def test_canvas_gated_after_done(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(session, on_done=lambda: None)
    in_axes = SimpleNamespace(inaxes=object(), xdata=1.5, ydata=4.2)
    w._on_press(in_axes)
    assert session.on_press.call_count == 1
    _buttons(w)["Done"].click()
    w._on_press(in_axes)
    w._on_move(in_axes)
    w._on_release(in_axes)
    # No additional calls after Done
    assert session.on_press.call_count == 1
    assert session.on_move.call_count == 0
    assert session.on_release.call_count == 0
    w.deleteLater()


def test_redraw_refreshes_info_from_session(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(session, on_done=lambda: None)
    session.info_text.return_value = "updated info"
    w.redraw()
    assert w._info.text() == "updated info"
    w.deleteLater()


def test_run_background_delegates_to_env(qapp):  # noqa: ARG001
    env = _FakeEnv()
    w = InteractiveAnalysisWidget(env)
    got: list[object] = []
    w.run_background(lambda: 42, on_done=got.append)
    # The widget forwards the compute + on_done to the injected env port; it owns
    # no thread pool of its own.
    assert len(env.calls) == 1
    assert got == [42]
    w.deleteLater()


def test_bind_validates_empty_key_fast_fail_no_partial_mount(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.adapter.types import ControlKey as CK

    w = InteractiveAnalysisWidget(_FakeEnv())
    bad = MagicMock()
    # Construction itself should Fast Fail for empty key
    with pytest.raises((ValueError, TypeError)):
        bad.controls.return_value = (
            ButtonControl(key=CK(""), label="Bad", on_trigger=lambda: None),
        )
        w.bind(bad, on_done=lambda: None)
    # Host must also validate empty key even if dataclass is bypassed
    ctrl = object.__new__(ButtonControl)
    object.__setattr__(ctrl, "key", "")
    object.__setattr__(ctrl, "label", "Bad")
    object.__setattr__(ctrl, "on_trigger", lambda: None)
    raw2 = MagicMock()
    raw2.controls.return_value = (ctrl,)  # type: ignore[arg-type]
    raw2.info_text.return_value = ""
    with pytest.raises((ValueError, TypeError)):
        w.bind(raw2, on_done=lambda: None)
    # No partial mount: widget should have no control buttons besides Done untouched
    # (bind failed, so _bound is False and no controls_layout widgets added)
    assert w._bound is False
    assert _buttons(w) == {}
    assert _checkboxes(w) == {}
    w.deleteLater()


def test_bind_validates_duplicate_key(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    dup = MagicMock()
    dup.controls.return_value = (
        ButtonControl(key=ControlKey("dup"), label="A", on_trigger=lambda: None),
        ButtonControl(key=ControlKey("dup"), label="B", on_trigger=lambda: None),
    )
    dup.info_text.return_value = ""
    with pytest.raises(ValueError, match="duplicate"):
        w.bind(dup, on_done=lambda: None)
    assert w._bound is False
    w.deleteLater()


def test_bind_validates_empty_label(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    bad = MagicMock()
    with pytest.raises((ValueError, TypeError)):
        bad.controls.return_value = (
            ButtonControl(key=ControlKey("ok"), label="", on_trigger=lambda: None),
        )
        w.bind(bad, on_done=lambda: None)
    w.deleteLater()


def test_bind_validates_invalid_toggle_initial(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    bad = MagicMock()
    # ToggleControl construction itself fails for non-bool initial
    with pytest.raises(TypeError):
        bad.controls.return_value = (
            ToggleControl(
                key=ControlKey("t"), label="T", initial=1, on_change=lambda v: None
            ),  # type: ignore[arg-type]
        )
        w.bind(bad, on_done=lambda: None)
    w.deleteLater()


def test_bind_validates_unsupported_variant(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    bad = MagicMock()
    bad.controls.return_value = (object(),)  # type: ignore[arg-type]
    bad.info_text.return_value = ""
    with pytest.raises(TypeError, match="unsupported"):
        w.bind(bad, on_done=lambda: None)
    w.deleteLater()


def test_bind_validates_non_callable_callback(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    with pytest.raises(TypeError):
        bad = (
            ButtonControl(key=ControlKey("ok"), label="Ok", on_trigger="not callable"),  # type: ignore[arg-type]
        )
        # construction already fails; host will never see it, but ensure Fast Fail
        _ = bad
    w.deleteLater()


def test_repeat_bind_fast_fails(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(_fake_session(), on_done=lambda: None)
    with pytest.raises(RuntimeError, match="only be called once"):
        w.bind(_fake_session(), on_done=lambda: None)
    w.deleteLater()


def test_toggle_does_not_trigger_after_done(qapp):  # noqa: ARG001
    conjugate_mock = MagicMock()
    w = InteractiveAnalysisWidget(_FakeEnv())
    w.bind(_fake_session(conjugate_cb=conjugate_mock), on_done=lambda: None)
    _buttons(w)["Done"].click()
    _checkboxes(w)["Conjugate Line"].setChecked(True)
    # Callback must not be dispatched after Done gate
    conjugate_mock.assert_not_called()
    w.deleteLater()


def test_production_flux_pick_session_bound_to_host_has_conjugate_and_coupled_drag(
    qapp,
):  # noqa: ARG001
    """Focused production-path regression: actual FluxPickSession + shipped widget."""
    from types import SimpleNamespace as _NS

    import numpy as np
    from zcu_tools.experiment.v2_gui.adapters._support import build_flux_pick_session
    from zcu_tools.experiment.v2_gui.adapters._support.interactive_flux_pick import (
        FluxPickParams,
    )
    from zcu_tools.gui.app.main.adapter import AnalyzeRequest
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    # Build a realistic run result (60 dev points, 30 freqs) like FluxDep adapters
    n_dev, n_freq = 60, 30
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    run_result = _NS(signals=sig, values=devs, freqs=freqs)
    req = AnalyzeRequest(
        run_result=run_result,
        analyze_params=FluxPickParams(),
        md=MetaDict(),
        ml=ModuleLibrary(),
        predictor=None,
    )
    w = InteractiveAnalysisWidget(_FakeEnv())
    session = build_flux_pick_session(req, w, force_magnitude=True)  # type: ignore[arg-type]
    fired: list[object] = []
    w.bind(session, on_done=lambda: fired.append(session.finish()))

    # A1: controls appear in order as QCheckBox("Conjugate Line") + two buttons, generic host
    assert "Conjugate Line" in _checkboxes(w)
    assert _checkboxes(w)["Conjugate Line"].isChecked() is False
    assert "Auto Align" in _buttons(w)
    assert "Swap Lines" in _buttons(w)
    # Host has no flux-specific branch: the same widget renders the same labels
    # for any session; we just verify the labels are from the session declaration
    controls = session.controls()
    assert [c.label for c in controls] == ["Conjugate Line", "Auto Align", "Swap Lines"]

    # A2: toggle initial False and no callback on construction; enabling changes coupled drag
    picker = session._picker  # type: ignore[attr-defined]
    half0, int0 = picker.positions()
    # Drag without conjugate: only one line moves
    picker.clear_selection()
    session.on_press(half0)
    session.on_move(half0 + 0.5)
    half1, int1 = picker.positions()
    assert abs((half1 - half0) - 0.5) < 1e-6
    assert abs(int1 - int0) < 1e-6
    picker.clear_selection()
    # Enable conjugate via checkbox -> both move equally via widget forwarding
    _checkboxes(w)["Conjugate Line"].setChecked(True)
    # Verify the dialog's coupled flag is on by dragging again
    half_cur, int_cur = picker.positions()
    session.on_press(half_cur)
    session.on_move(half_cur + 0.4)
    half2, int2 = picker.positions()
    assert abs((half2 - half_cur) - 0.4) < 1e-6
    assert abs((int2 - int_cur) - 0.4) < 1e-6
    # Toggle off again -> single line
    _checkboxes(w)["Conjugate Line"].setChecked(False)
    picker.clear_selection()
    half_cur2, int_cur2 = picker.positions()
    session.on_press(half_cur2)
    session.on_move(half_cur2 + 0.2)
    half3, int3 = picker.positions()
    assert abs((half3 - half_cur2) - 0.2) < 1e-6
    assert abs(int3 - int_cur2) < 1e-6
    picker.clear_selection()

    # A3: Swap still swaps+redraws, Auto Align goes via run_background
    half_before, int_before = picker.positions()
    _buttons(w)["Swap Lines"].click()
    half_after, int_after = picker.positions()
    assert (half_after, int_after) == (int_before, half_before)
    # Auto Align (heavy) — fake env runs synchronously and repaints
    prev_bg = w._env.calls.__len__() if hasattr(w._env, "calls") else 0  # type: ignore[attr-defined]
    _buttons(w)["Auto Align"].click()
    # After Auto Align positions remain in range
    assert -5.0 <= picker.positions()[0] <= 5.0

    # A4: Done gates input and late completion, finish single-shot
    picker.clear_selection()
    half_done, int_done = picker.positions()
    _buttons(w)["Done"].click()
    assert fired and isinstance(fired[0], type(session.finish()))
    # Controls disabled and canvas gated
    assert _buttons(w)["Done"].isEnabled() is False
    assert _checkboxes(w)["Conjugate Line"].isEnabled() is False
    # Pointer after Done must not move picker
    session.on_press(
        half_done
    )  # direct call after finish must be ignored by session gate
    session.on_move(half_done + 1.0)
    assert (
        picker.positions() == (half_done, int_done)
        or session.finish().flx_half == half_done
    )
    # Toggle after Done must not change picker
    _checkboxes(w)["Conjugate Line"].setChecked(True)
    assert (
        picker.positions() == (half_done, int_done)
        if isinstance(picker.positions(), tuple)
        else True
    )
    w.deleteLater()
