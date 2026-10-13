"""Headless tests for AnalyzePanelWidget — tabs, bounds preset, guards.

The search worker / embedded-figure path is covered elsewhere; here we pin the
panel-local UI behaviour: the three tabs, the bounds preset and its spin boxes,
frequency fields marked by the transitions that need them, Search being
blocked when the database path is empty / missing, and each tab reporting on
its own status line. Controls are located by their public objectNames.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
)
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.state import FitState, FluxDepState
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.transitions_form import TransitionsForm
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.app.fluxdep._search_result import (
    perfect_fit_result,
    synthetic_search_input,
)
from tests.gui.app.fluxdep._window import named


@pytest.fixture
def panel(qapp):
    ctrl = Controller(FluxDepState())
    w = AnalyzePanelWidget(ctrl)
    yield w, ctrl
    w.quiesce()
    w.dispose()
    w.deleteLater()
    ctrl.interactive.dispose()


@pytest.mark.parametrize("publish_before_panel", [False, True])
@pytest.mark.parametrize("frequencies", [(None, None), (7.0, 3.0)])
def test_published_fit_inputs_reach_search_and_show(
    qapp, publish_before_panel, frequencies, search_database_hdf5
):
    ctrl = Controller(FluxDepState())
    widget = None
    if not publish_before_panel:
        widget = AnalyzePanelWidget(ctrl)
    try:
        ctrl.set_fit_params(
            search_database_hdf5,
            (3.0, 12.0),
            (0.3, 1.5),
            (0.2, 1.2),
            TransitionDict({"transitions": [(0, 1)]}),
            *frequencies,
        )
        if widget is None:
            widget = AnalyzePanelWidget(ctrl)
        widget.show_tab("search")
        assert (
            named(widget, QLineEdit, "fluxdep.fit.search_db").text()
            == ctrl.state.fit.search_db
        ), "Search must display the currently published database"
        assert _bound_values(widget) == [3.0, 12.0, 0.3, 1.5, 0.2, 1.2]
        assert [
            named(widget, QLineEdit, f"fluxdep.fit.{field}").text()
            for field in ("r_f", "sample_f")
        ] == ["" if value is None else f"{value:g}" for value in frequencies]
        for form in widget.findChildren(TransitionsForm):
            transitions = form.get_transitions()
            assert transitions.get("transitions") == [(0, 1)]
            assert not transitions.get("mirror"), (
                "Search and Show must not retain default mirror transitions"
            )
    finally:
        if widget is not None:
            widget.quiesce()
            widget.dispose()
            widget.deleteLater()
        ctrl.interactive.dispose()


@pytest.mark.parametrize("display_edit", ["none", "subset", "missing_frequency"])
def test_fit_publication_preserves_display_edits_and_renders_owner_inputs(
    panel, monkeypatch, display_edit, search_database_hdf5
):
    from zcu_tools.gui.app.fluxdep.services import viz

    widget, ctrl = panel
    ctrl.set_fit_params(
        search_database_hdf5,
        (3.0, 12.0),
        (0.3, 1.5),
        (0.2, 1.2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    show_form = next(
        group for group in widget.findChildren(QGroupBox) if group.title() == "Display"
    ).findChildren(TransitionsForm)[0]
    search_form = next(
        group
        for group in widget.findChildren(QGroupBox)
        if group.title() == "Search parameters"
    ).findChildren(TransitionsForm)[0]
    if display_edit == "subset":
        show_form.set_transitions(TransitionDict({"transitions": [(0, 2)]}))
    elif display_edit == "missing_frequency":
        show_form.set_transitions(TransitionDict({"mirror": [(0, 1)]}))
        # An unsubmitted Search edit is not the visualization's fit input.
        search_form.set_transitions(TransitionDict({"mirror": [(0, 1)]}))
    display_before_result = show_form.get_transitions()
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    fluxs, freqs = np.array([0.0, 0.1]), np.array([1.0, 1.0])
    ctrl.record_search_result(
        perfect_fit_result(fluxs, freqs, bounds),
        synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
    )
    assert show_form.get_transitions() == display_before_result
    monkeypatch.setattr(
        viz,
        "calculate_energy_vs_flux",
        lambda params, fluxs, **kwargs: (
            None,
            np.tile(np.arange(15, dtype=float), (len(fluxs), 1)),
        ),
    )
    search_status = named(widget, QLabel, "fluxdep.fit.status").text()
    widget.show_tab("show")
    named(widget, QPushButton, "fluxdep.show.apply").click()
    # Show reports its own drawing problems; the Search status line is untouched.
    assert named(widget, QLabel, "fluxdep.fit.status").text() == search_status
    show_status = named(widget, QLabel, "fluxdep.show.status").text()
    assert bool(show_status) == (display_edit == "missing_frequency")
    canvas = widget.findChildren(FigureCanvasQTAgg)[0]
    assert len(canvas.figure.axes) == 1
    lines = canvas.figure.axes[0].lines
    assert len(lines) == 1
    expected_frequency = 2.0 if display_edit == "subset" else 1.0
    np.testing.assert_allclose(lines[0].get_ydata(), expected_frequency)
    assert ctrl.state.fit.transitions == {"transitions": [(0, 1)]}
    assert ctrl.state.fit.params == (5.0, 1.0, 0.5)


def _show_title(widget: AnalyzePanelWidget) -> str:
    return widget.findChildren(FigureCanvasQTAgg)[0].figure.axes[0].get_title()


def _freshness(widget: AnalyzePanelWidget) -> QLabel:
    return named(widget, QLabel, "fluxdep.fit.freshness")


def _export_button(widget: AnalyzePanelWidget) -> QPushButton:
    return named(widget, QPushButton, "fluxdep.fit.export_params")


def _search_group(widget: AnalyzePanelWidget) -> QGroupBox:
    return next(
        group
        for group in widget.findChildren(QGroupBox)
        if group.title() == "Search parameters"
    )


def _bound_values(widget: AnalyzePanelWidget) -> list[float]:
    """EJ/EC/EL lower and upper bounds as the Search form shows them."""
    return [box.value() for box in _search_group(widget).findChildren(QDoubleSpinBox)]


def _preset_names(widget: AnalyzePanelWidget) -> list[str]:
    """The named bounds presets the combo offers, without the custom entry."""
    combo = named(widget, QComboBox, "fluxdep.fit.bounds_preset")
    names = [combo.itemText(index) for index in range(combo.count())]
    return [name for name in names if name != "custom"]


def _search_without_frequencies(widget: AnalyzePanelWidget) -> None:
    """Enter transitions that need neither r_f nor sample_f, enabling Search."""
    form = _search_group(widget).findChildren(TransitionsForm)[0]
    form.set_transitions(TransitionDict({"transitions": [(0, 1)]}))


def _choose_preset(widget: AnalyzePanelWidget, name: str) -> None:
    """Pick a bounds preset the way a user activates a combo entry."""
    combo = named(widget, QComboBox, "fluxdep.fit.bounds_preset")
    index = combo.findText(name)
    assert index >= 0, f"missing preset {name!r}"
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def test_upstream_change_marks_search_and_show_stale_until_a_new_result(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    ctrl.set_fit_params(
        search_database_hdf5,
        (3.0, 12.0),
        (0.3, 1.5),
        (0.2, 1.2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    widget = AnalyzePanelWidget(ctrl)
    try:
        fluxs, freqs = ctrl.selected_pointcloud()
        ctrl.record_search_result(
            perfect_fit_result(fluxs, freqs, bounds),
            synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
        )
        widget.show_tab("show")
        assert _freshness(widget).text() == ""
        assert "Outdated" not in _show_title(widget)
        assert _export_button(widget).isEnabled()

        ctrl.set_selection(np.array([True, False, True, True]))
        assert ctrl.state.fit_stale_reasons == ["filter changed"]
        assert _freshness(widget).text() == "Outdated: filter changed"
        assert "Outdated: filter changed" in _show_title(widget)
        assert not _export_button(widget).isEnabled()
        # Show keeps the recorded search cloud, not the newer Filter selection.
        scatter = next(
            artist
            for artist in widget.findChildren(FigureCanvasQTAgg)[0]
            .figure.axes[0]
            .collections
            if artist.get_label() == "_points"
        )
        np.testing.assert_allclose(
            np.asarray(scatter.get_offsets(), dtype=np.float64),
            np.column_stack([fluxs, freqs]),
        )

        fluxs, freqs = ctrl.selected_pointcloud()
        ctrl.record_search_result(
            perfect_fit_result(fluxs, freqs, bounds),
            synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
        )
        assert ctrl.state.fit_stale_reasons == []
        assert _freshness(widget).text() == ""
        assert "Outdated" not in _show_title(widget)
        assert _export_button(widget).isEnabled()
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


def _set_joint_fit_inputs(ctrl: Controller, search_db: str) -> ParamBounds:
    """Publish 0-1 inputs over the joint cloud and return their bounds."""
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    ctrl.set_fit_params(
        search_db,
        bounds.EJ,
        bounds.EC,
        bounds.EL,
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    return bounds


def _record_perfect_fit(ctrl: Controller, bounds: ParamBounds) -> None:
    """Publish a zero-residual result over the currently selected cloud."""
    fluxs, freqs = ctrl.selected_pointcloud()
    ctrl.record_search_result(
        perfect_fit_result(fluxs, freqs, bounds),
        synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
    )


def _background_grids(figure: Figure) -> list[np.ndarray]:
    """Every drawn heatmap's normalized-flux/GHz corner grid, in draw order."""
    return [
        np.asarray(mesh.get_coordinates(), dtype=np.float64)
        for mesh in figure.axes[0].collections
        if isinstance(mesh, QuadMesh)
    ]


def _assert_same_grids(drawn: list[np.ndarray], expected: list[np.ndarray]) -> None:
    assert expected
    assert len(drawn) == len(expected)
    for grid, reference in zip(drawn, expected, strict=True):
        np.testing.assert_allclose(grid, reference)


def _assert_native_background(shown: Figure, ctrl: Controller) -> list[np.ndarray]:
    """Show's heatmaps equal the shared Controller renderer's; return them."""
    native = Figure()
    ctrl.render_fit(native)
    expected = _background_grids(native)
    _assert_same_grids(_background_grids(shown), expected)
    return expected


def test_repeated_stale_alignment_keeps_captured_show_background(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    bounds = _set_joint_fit_inputs(ctrl, search_database_hdf5)
    widget = AnalyzePanelWidget(ctrl)
    try:
        _record_perfect_fit(ctrl, bounds)
        widget.show_tab("show")
        shown = widget.findChildren(FigureCanvasQTAgg)[0].figure
        captured = _assert_native_background(shown, ctrl)

        for flux_half, flux_int in ((0.55, 1.05), (0.60, 1.10)):
            ctrl.set_alignment("a", flux_half, flux_int)
            assert ctrl.state.fit_stale_reasons == ["a re-aligned"]
            assert "Outdated: a re-aligned" in shown.axes[0].get_title()
            _assert_same_grids(_background_grids(shown), captured)
            _assert_native_background(shown, ctrl)

        _record_perfect_fit(ctrl, bounds)
        assert ctrl.state.fit_stale_reasons == []
        assert "Outdated" not in shown.axes[0].get_title()
        fresh = _assert_native_background(shown, ctrl)
        assert any(
            grid.shape != reference.shape or not np.allclose(grid, reference)
            for grid, reference in zip(fresh, captured, strict=True)
        ), "a fresh search must draw the newly aligned source background"
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


Limits = tuple[tuple[float, float], tuple[float, float]]


def _display_group(widget: AnalyzePanelWidget) -> QGroupBox:
    return next(
        group for group in widget.findChildren(QGroupBox) if group.title() == "Display"
    )


def _click_auto_limits(widget: AnalyzePanelWidget) -> Limits:
    """Click Show's Auto limits and read the x/y limit boxes it filled."""
    display = _display_group(widget)
    named(display, QPushButton, "fluxdep.show.auto_limits").click()
    x_lo, x_hi, y_lo, y_hi = display.findChildren(QDoubleSpinBox)[:4]
    return (x_lo.value(), x_hi.value()), (y_lo.value(), y_hi.value())


def _as_displayed(widget: AnalyzePanelWidget, limits: Limits) -> Limits:
    """Round normalized-flux/GHz limits to the limit boxes' display precision."""
    decimals = _display_group(widget).findChildren(QDoubleSpinBox)[0].decimals()
    (x_lo, x_hi), (y_lo, y_hi) = limits
    return (
        (round(x_lo, decimals), round(x_hi, decimals)),
        (round(y_lo, decimals), round(y_hi, decimals)),
    )


def test_auto_limits_keep_the_recorded_search_frame_until_a_fresh_search(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    bounds = _set_joint_fit_inputs(ctrl, search_database_hdf5)
    widget = AnalyzePanelWidget(ctrl)
    try:
        widget.show_tab("show")
        assert _click_auto_limits(widget) == _as_displayed(
            widget, ctrl.fit_auto_limits()
        ), "before search, Auto limits use the published sources"

        _record_perfect_fit(ctrl, bounds)
        captured = _click_auto_limits(widget)
        assert captured == _as_displayed(widget, ctrl.fit_auto_limits())

        # A non-integer shift of one source; the recorded search keeps its frame.
        ctrl.set_alignment("a", 0.685, 1.185)
        assert ctrl.state.fit_stale_reasons == ["a re-aligned"]
        assert _click_auto_limits(widget) == captured

        _record_perfect_fit(ctrl, bounds)
        fresh = _click_auto_limits(widget)
        assert fresh == _as_displayed(widget, ctrl.fit_auto_limits())
        assert fresh[0] != captured[0], "a fresh search adopts the new frame"
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


def test_params_only_auto_limits_follow_current_sources(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    _set_joint_fit_inputs(ctrl, search_database_hdf5)
    ctrl.state.set_fit_result((5.0, 1.0, 0.5))
    widget = AnalyzePanelWidget(ctrl)
    try:
        widget.show_tab("show")
        before = _click_auto_limits(widget)
        ctrl.set_alignment("a", 0.685, 1.185)
        moved = _click_auto_limits(widget)
        assert moved == _as_displayed(widget, ctrl.fit_auto_limits())
        assert moved[0] != before[0], "params-only display has no recorded frame"
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


def test_upstream_refresh_keeps_unapplied_display_edits_pending(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    bounds = _set_joint_fit_inputs(ctrl, search_database_hdf5)
    widget = AnalyzePanelWidget(ctrl)
    try:
        _record_perfect_fit(ctrl, bounds)
        widget.show_tab("show")
        display = next(
            group
            for group in widget.findChildren(QGroupBox)
            if group.title() == "Display"
        )
        show_form = display.findChildren(TransitionsForm)[0]
        x_hi = display.findChildren(QDoubleSpinBox)[1]
        shown = widget.findChildren(FigureCanvasQTAgg)[0].figure

        def labels() -> list[str]:
            return [str(line.get_label()) for line in shown.axes[0].lines]

        assert labels() == ["0 -> 1"]
        xlim_applied = shown.axes[0].get_xlim()
        assert x_hi.value() != pytest.approx(0.9)

        show_form.set_transitions(TransitionDict({"transitions": [(0, 2)]}))
        x_hi.setValue(0.9)
        assert labels() == ["0 -> 1"], "editing a field is not Apply"

        ctrl.set_alignment("a", 0.55, 1.05)
        assert ctrl.state.fit_stale_reasons == ["a re-aligned"]
        assert "Outdated: a re-aligned" in shown.axes[0].get_title()
        assert labels() == ["0 -> 1"], "an upstream refresh must not apply edits"
        assert shown.axes[0].get_xlim() == pytest.approx(xlim_applied)
        assert show_form.get_transitions() == {"transitions": [(0, 2)]}
        assert x_hi.value() == pytest.approx(0.9)

        named(widget, QPushButton, "fluxdep.show.apply").click()
        assert labels() == ["0 -> 2"]
        assert shown.axes[0].get_xlim()[1] == pytest.approx(0.9)
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


def test_three_tabs_filter_search_show(panel):
    w, _ = panel
    tabs = w.findChildren(QTabWidget)
    assert len(tabs) == 1
    labels = [tabs[0].tabText(i) for i in range(tabs[0].count())]
    assert labels == ["Filter", "Search", "Show"]


def test_each_preset_fills_its_own_bounds(panel):
    w, _ = panel
    preset = named(w, QComboBox, "fluxdep.fit.bounds_preset")
    filled = {}
    for name in _preset_names(w):
        _choose_preset(w, name)
        assert preset.currentText() == name
        filled[name] = tuple(_bound_values(w))
    assert len(filled) > 1
    assert len(set(filled.values())) == len(filled), "presets must differ"


def _publish_bounds(ctrl: Controller, search_db: str, values: tuple[float, ...]):
    ej_lo, ej_hi, ec_lo, ec_hi, el_lo, el_hi = values
    ctrl.set_fit_params(
        search_db,
        (ej_lo, ej_hi),
        (ec_lo, ec_hi),
        (el_lo, el_hi),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )


def test_bounds_preset_names_the_preset_the_bounds_match(panel, search_database_hdf5):
    w, ctrl = panel
    preset = named(w, QComboBox, "fluxdep.fit.bounds_preset")
    default = FitState()
    fresh = preset.currentText()
    assert fresh != "custom", "a fresh form names the preset its default bounds are"
    _choose_preset(w, fresh)
    assert _bound_values(w) == [*default.EJb, *default.ECb, *default.ELb]
    filled = {}
    for name in _preset_names(w):
        _choose_preset(w, name)
        filled[name] = tuple(_bound_values(w))
    for name, values in filled.items():
        _publish_bounds(ctrl, search_database_hdf5, values)
        assert preset.currentText() == name
    ctrl.set_fit_params(
        search_database_hdf5,
        (2.5, 15.0),
        (0.2, 2.0),
        (0.1, 2.0),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    assert preset.currentText() == "custom"

    _choose_preset(w, "integer")
    assert preset.currentText() == "integer"
    ej_lo = _search_group(w).findChildren(QDoubleSpinBox)[0]
    ej_lo.setValue(ej_lo.value() + 0.5)
    assert preset.currentText() == "custom", "an edited bound is no longer the preset"
    ej_lo.setValue(filled["integer"][0])
    assert preset.currentText() == "integer"


def test_search_blocked_on_empty_database_path(panel):
    w, ctrl = panel
    _search_without_frequencies(w)
    named(w, QLineEdit, "fluxdep.fit.search_db").setText("")
    named(w, QPushButton, "fluxdep.fit.search").click()
    assert named(w, QLabel, "fluxdep.fit.status").text()
    assert ctrl.state.fit.search_db == ""
    assert ctrl.search.current is None


def test_search_blocked_on_missing_database_file(panel):
    w, ctrl = panel
    _search_without_frequencies(w)
    named(w, QLineEdit, "fluxdep.fit.search_db").setText("/nonexistent/path/to/db.h5")
    named(w, QPushButton, "fluxdep.fit.search").click()
    assert named(w, QLabel, "fluxdep.fit.status").text()
    assert ctrl.state.fit.search_db == ""
    assert ctrl.search.current is None


def test_search_status_ignored_in_horizontal_minimum_width(panel):
    w, _ = panel
    status = named(w, QLabel, "fluxdep.fit.status")
    assert status.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored


def test_show_tab_has_display_tools(panel):
    w, _ = panel
    display = _display_group(w)
    # x/y limit boxes + const-freq toggle + a transitions-to-show form exist
    assert len(display.findChildren(QDoubleSpinBox)) == 4
    toggles = display.findChildren(QCheckBox)
    assert len(toggles) == 1 and toggles[0].isChecked()  # default on
    assert len(display.findChildren(TransitionsForm)) == 1


def test_auto_limits_match_visualizer(panel, qapp):
    import numpy as np
    from zcu_tools.analysis.fluxdep.models import PointsData
    from zcu_tools.analysis.spectrum import SpectrumData
    from zcu_tools.gui.app.fluxdep.state import SpectrumEntry

    w, ctrl = panel
    fx = np.linspace(0.0, 0.5, 6)
    fr = np.linspace(4.0, 6.0, 5)
    raw = SpectrumData(
        dev_values=fx, fluxs=fx, freqs=fr, signals=np.ones((6, 5), complex)
    )
    pts = PointsData(dev_values=fx, fluxs=fx, freqs=np.full(6, 5.0))
    ctrl.state.put_spectrum(
        SpectrumEntry(name="a", spec_type="TwoTone", raw=raw, points=pts, aligned=True)
    )
    _x, y = _click_auto_limits(w)
    # y spans the spectrum's freq range [4, 6]
    assert y == pytest.approx((4.0, 6.0))


@pytest.mark.parametrize(
    ("needing", "needed", "other"),
    [
        (TransitionDict({"mirror": [(0, 1)]}), "sample_f", "r_f"),
        (TransitionDict({"red side": [(0, 1)]}), "r_f", "sample_f"),
    ],
    ids=["mirror", "red side"],
)
def test_frequency_field_is_marked_while_a_transition_needs_it(
    panel, needing, needed, other
):
    w, _ = panel
    form = _search_group(w).findChildren(TransitionsForm)[0]
    required = named(w, QLabel, f"fluxdep.fit.{needed}.required")
    field = named(w, QLineEdit, f"fluxdep.fit.{needed}")
    form.set_transitions(TransitionDict({"transitions": [(0, 1)]}))
    assert required.isHidden()
    form.set_transitions(needing)
    assert not required.isHidden()
    assert named(w, QLabel, f"fluxdep.fit.{other}.required").isHidden()
    field.setText("6.0")
    assert required.isHidden()
    field.setText("")
    assert not required.isHidden()
    form.set_transitions(TransitionDict({"transitions": [(0, 1)]}))
    assert required.isHidden()


@pytest.mark.parametrize(
    ("needing", "field"),
    [
        (TransitionDict({"transitions": [(0, 1)], "mirror": [(0, 1)]}), "sample_f"),
        (TransitionDict({"red side": [(0, 1)]}), "r_f"),
    ],
    ids=["mirror", "red side"],
)
def test_search_waits_for_a_needed_frequency_and_keeps_the_result(
    qapp, cross_controller, search_database_hdf5, needing, field
):
    ctrl = cross_controller
    _record_perfect_fit(ctrl, _set_joint_fit_inputs(ctrl, search_database_hdf5))
    recorded = ctrl.state.fit.params
    widget = AnalyzePanelWidget(ctrl)
    try:
        widget.show_tab("search")
        search = named(widget, QPushButton, "fluxdep.fit.search")
        required = named(widget, QLabel, f"fluxdep.fit.{field}.required")
        assert search.isEnabled()
        form = _search_group(widget).findChildren(TransitionsForm)[0]
        form.set_transitions(needing)
        assert not required.isHidden()
        assert not search.isEnabled(), "a search bound to fail must not start"
        search.click()
        assert ctrl.search.current is None
        assert ctrl.state.fit.params == recorded, "the recorded result stays"
        named(widget, QLineEdit, f"fluxdep.fit.{field}").setText("6.0")
        assert required.isHidden() and search.isEnabled()
        named(widget, QLineEdit, f"fluxdep.fit.{field}").setText("")
        assert not search.isEnabled()
        form.set_transitions(TransitionDict({"transitions": [(0, 1)]}))
        assert search.isEnabled()
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()


def test_freq_fields_are_blank_by_default(panel):
    w, _ = panel
    for field in ("r_f", "sample_f"):
        assert named(w, QLineEdit, f"fluxdep.fit.{field}").text() == ""


@pytest.mark.parametrize("usable", [False, True])
def test_filter_skips_completed_empty_spectra(qapp, spectrum_hdf5, usable):
    import numpy as np
    from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget

    filepath, *_ = spectrum_hdf5
    ctrl = Controller(FluxDepState())
    name = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(name, 0.0, 1.0)
    ctrl.set_points(name, np.empty(0), np.empty(0))
    if usable:
        from dataclasses import replace

        ctrl.state.put_spectrum(replace(ctrl.state.spectrums[name], name="signal"))
        other = "signal"
        ctrl.set_alignment(other, 0.0, 1.0)
        ctrl.set_points(other, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    w = AnalyzePanelWidget(ctrl)
    try:
        w.activate()
        selectors = w.findChildren(SelectorWidget)
        assert len(selectors) == (1 if usable else 0)
        if usable:
            result = selectors[0].get_result()
            np.testing.assert_array_equal(result.freqs, [5.0, 5.1])
            assert result.selected.all() and result.selected.size == 2
    finally:
        w.quiesce()
        w.dispose()
        w.deleteLater()
        qapp.processEvents()


def test_activate_attaches_selector_when_points_exist(qapp):
    import numpy as np
    from zcu_tools.analysis.fluxdep.models import PointsData
    from zcu_tools.analysis.spectrum import SpectrumData
    from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
    from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget

    st = FluxDepState()
    fx = np.array([0.0, 0.1, 0.2, 0.3])
    raw = SpectrumData(
        dev_values=fx,
        fluxs=fx,
        freqs=np.linspace(4, 6, 5),
        signals=np.ones((4, 5), complex),
    )
    pts = PointsData(dev_values=fx, fluxs=fx, freqs=np.full(4, 5.0))
    st.put_spectrum(
        SpectrumEntry(
            name="s1",
            spec_type="TwoTone",
            raw=raw,
            points=pts,
            aligned=True,
            points_completed=True,
        )
    )
    ctrl = Controller(st)
    w = AnalyzePanelWidget(ctrl)
    try:
        w.activate()
        selector = w.findChild(SelectorWidget)
        assert selector is not None
        assert selector.get_result().selected.all()
    finally:
        w.quiesce()
        w.dispose()
        ctrl.interactive.dispose()
        w.deleteLater()
        qapp.processEvents()


def _ranked_database(tmp_path) -> str:
    """Two-candidate DB whose first entry matches the points below exactly."""
    import h5py

    path = tmp_path / "ranked.h5"
    grid = np.linspace(0.0, 0.5, 9)
    energies = np.zeros((2, 9, 3))
    for i in range(2):
        energies[i, :, 1] = (i + 1) * (1 + grid)
        energies[i, :, 2] = (i + 1) * (2 + grid)
    with h5py.File(path, "w") as database:
        database.create_dataset("fluxs", data=grid)
        database.create_dataset("params", data=[[3.0, 0.8, 0.5], [4.0, 1.1, 0.6]])
        database.create_dataset("energies", data=energies)
    return str(path)


def test_search_diagnostic_axis_labels_clear_the_next_subplot_after_resize(
    qapp, tmp_path
):
    from zcu_tools.gui.app.fluxdep.state import SpectrumEntry

    runner = BackgroundRunner()
    ctrl = Controller(
        FluxDepState(),
        interactive_owner=QtOwnerScheduler(),
        search_runtime=FluxDepSearchRuntime(
            runner, ProgressService(QtProgressTransport())
        ),
    )
    axis = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
    points = np.array([0.0, 0.2, 0.4])
    ctrl.state.put_spectrum(
        SpectrumEntry(
            name="a",
            spec_type="TwoTone",
            raw={
                "dev_values": axis.copy(),
                "fluxs": axis,
                "freqs": np.array([1.0, 1.2, 1.4]),
                "signals": np.ones((5, 3), dtype=np.complex128),
            },
            points={
                "dev_values": points.copy(),
                "fluxs": points,
                "freqs": np.array([1.0, 1.2, 1.4]),
            },
            flux_half=0.5,
            flux_int=1.0,
            flux_period=1.0,
            aligned=True,
            points_completed=True,
        )
    )
    ctrl.set_fit_params(
        _ranked_database(tmp_path),
        (0.1, 10.0),
        (0.1, 10.0),
        (0.1, 10.0),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    widget = AnalyzePanelWidget(ctrl)
    widget.resize(1600, 1000)
    widget.show()
    try:
        widget.show_tab("search")
        named(widget, QPushButton, "fluxdep.fit.search").click()
        assert runner.quiesce()
        qapp.processEvents()
        figure = widget.figures["diagnostic"]
        qapp.processEvents()
        widget.resize(1200, 800)
        qapp.processEvents()
        canvas = figure.canvas
        assert isinstance(canvas, FigureCanvasQTAgg)
        canvas.draw()
        renderer = canvas.get_renderer()
        by_label = {axes.get_xlabel(): axes for axes in figure.axes}
        for label, below in (("EJ", "EC"), ("EC", "EL")):
            box = by_label[label].xaxis.label.get_window_extent(renderer)
            neighbour = by_label[below].get_tightbbox(renderer)
            assert neighbour is not None
            assert not box.overlaps(neighbour), (
                f"the {label} axis label is covered by the {below} subplot"
            )
        last = by_label["EL"].xaxis.label.get_window_extent(renderer)
        assert last.y0 >= figure.bbox.y0, "the EL axis label is clipped"
    finally:
        ctrl.search.begin_close()
        assert runner.quiesce()
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()
        ctrl.interactive.dispose()
