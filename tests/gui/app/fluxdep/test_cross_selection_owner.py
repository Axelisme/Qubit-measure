"""Joint-cloud owner invalidation and nonterminal selection publication."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.cross_selection import (
    analyze_cross_selection,
    make_cross_selection_state,
)
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    SelectionChangedPayload,
    SpectrumChangedPayload,
)
from zcu_tools.gui.app.fluxdep.interactive import (
    FluxDepInteractiveOwner,
    FluxDepInteractivePorts,
)
from zcu_tools.gui.app.fluxdep.state import (
    SELECTION_VERSION_KEY,
    FluxDepState,
    SpectrumEntry,
    spectrum_version_key,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


def test_capture_full_cloud_sources_and_seed_published_mask_and_distance(
    cross_controller,
):
    ctrl = cross_controller
    ctrl.set_selection(np.array([False, True, False, False]), 0.08)
    context = ctrl.interactive.begin_cross_selection()
    assert ctrl.interactive.begin_cross_selection() is context
    assert ctrl.interactive.current_cross_selection() is context
    assert context.source_versions == tuple(
        (name, ctrl.state.version.get(spectrum_version_key(name)))
        for name in ("a", "empty", "b")
    )
    inputs = context.plugin.inputs
    assert tuple(item.name for item in inputs.backgrounds) == ("a", "b")
    np.testing.assert_array_equal(inputs.fluxs, [0.0, 0.5, 1.0, 0.5])
    assert inputs.flux_bound == (0.0, 1.0)
    assert inputs.freq_bound == (4.0, 5.0)
    np.testing.assert_array_equal(
        context.session.snapshot().selected, [False, True, False, False]
    )
    assert context.session.snapshot().min_distance == 0.08
    assert not context.session.can_undo()
    context.plugin.clear.execute(context.session, None)
    ctrl.interactive.cancel()
    new = ctrl.interactive.begin_cross_selection()
    # A cancelled draft publishes nothing, so the next one starts from the
    # same published mask, not from the cancelled draft.
    assert new is not context
    np.testing.assert_array_equal(
        new.session.snapshot().selected, [False, True, False, False]
    )
    assert new.session.snapshot().min_distance == 0.08


def test_new_context_starts_all_selected_without_a_published_mask(cross_controller):
    ctrl = cross_controller
    assert ctrl.state.selection_mask() is None
    context = ctrl.interactive.begin_cross_selection()
    assert context.session.snapshot().selected.all()
    assert context.session.snapshot().selected.size == 4


def test_new_context_seeds_unmasked_spectra_all_selected(cross_controller):
    ctrl = cross_controller
    ctrl.set_selection(np.array([True, False, False, False]), 0.0)
    # New points for b drop only b's mask; a keeps its published exclusions.
    ctrl.set_points("b", np.array([0.2, 0.8]), np.array([4.2, 4.8]))
    context = ctrl.interactive.begin_cross_selection()
    np.testing.assert_array_equal(
        context.session.snapshot().selected, [True, False, False, True, True]
    )


def test_finished_mask_seeds_the_next_context(cross_controller):
    ctrl = cross_controller
    first = ctrl.interactive.begin_cross_selection()
    first.plugin.stroke.execute(
        first.session, BrushStroke((BrushPoint(0.0, 4.0),), 0.0, "erase")
    )
    ctrl.interactive.finish_cross_selection()
    assert ctrl.interactive.current_cross_selection() is None
    reopened = ctrl.interactive.begin_cross_selection()
    np.testing.assert_array_equal(
        reopened.session.snapshot().selected, [False, True, True, True]
    )


def test_apply_publishes_exact_snapshot_once_retaining_input_and_undo(cross_controller):
    ctrl = cross_controller
    context = ctrl.interactive.begin_cross_selection()
    notifications = []
    unsubscribe = ctrl.bus.subscribe(
        SelectionChangedPayload, lambda event: notifications.append(event)
    ).unsubscribe
    try:
        before_version = ctrl.state.version.get(SELECTION_VERSION_KEY)
        context.plugin.stroke.execute(
            context.session, BrushStroke((BrushPoint(0.5, 4.5),), 0.0, "erase")
        )
        result = ctrl.interactive.apply_cross_selection()
        np.testing.assert_array_equal(result.selected, [True, False, True, False])
        np.testing.assert_array_equal(ctrl.state.selection_mask(), result.selected)
        assert len(notifications) == 1
        assert ctrl.state.version.get(SELECTION_VERSION_KEY) == before_version + 1
        assert context.selection_version == before_version + 1
        assert ctrl.interactive.current_cross_selection() is context
        assert context.session.can_undo()
        assert context.session.undo().selected.all()
        context.plugin.clear.execute(context.session, None)
        empty = ctrl.interactive.apply_cross_selection()
        assert empty.selected.size == 4 and not empty.selected.any()
        assert len(notifications) == 2
        context.plugin.perform_on_all.execute(context.session, None)
    finally:
        unsubscribe()


def test_apply_result_mutation_cannot_change_published_or_committed_mask(
    cross_controller,
):
    ctrl = cross_controller
    context = ctrl.interactive.begin_cross_selection()
    notifications = []
    subscription = ctrl.bus.subscribe(
        SelectionChangedPayload, lambda event: notifications.append(event)
    )
    try:
        context.plugin.stroke.execute(
            context.session, BrushStroke((BrushPoint(0.5, 4.5),), 0.0, "erase")
        )
        before_version = ctrl.state.version.get(SELECTION_VERSION_KEY)
        result = ctrl.interactive.apply_cross_selection()
        result.selected[:] = False
        np.testing.assert_array_equal(
            ctrl.state.selection_mask(), [True, False, True, False]
        )
        np.testing.assert_array_equal(
            context.session.snapshot().selected, [True, False, True, False]
        )
        assert ctrl.state.version.get(SELECTION_VERSION_KEY) == before_version + 1
        assert len(notifications) == 1
        assert ctrl.interactive.current_cross_selection() is context
        assert context.session.can_undo()
    finally:
        subscription.unsubscribe()


@pytest.mark.parametrize(
    "change",
    [
        "inactive_points",
        "same_length_reload",
        "empty_source",
        "add",
        "remove",
        "external_selection",
    ],
)
def test_source_or_owner_switch_invalidates_old_full_cloud(cross_controller, change):
    ctrl = cross_controller
    context = ctrl.interactive.begin_cross_selection()
    if change == "inactive_points":
        ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
    elif change == "same_length_reload":
        entry = ctrl.state.spectrums["b"]
        ctrl.state.put_spectrum(
            replace(
                entry,
                points={
                    "dev_values": np.array([0.9]),
                    "fluxs": np.array([0.9]),
                    "freqs": np.array([4.9]),
                },
            )
        )
    elif change == "empty_source":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums["empty"]))
        ctrl.bus.emit(SpectrumChangedPayload(name="empty"))
    elif change == "add":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums["empty"], name="new"))
    elif change == "remove":
        ctrl.remove_spectrum("b")
    elif change == "external_selection":
        ctrl.set_selection(np.ones(4, dtype=bool), 0.01)
    assert ctrl.interactive.current_cross_selection() is None
    with pytest.raises(FailedPreconditionError):
        context.plugin.clear.execute(context.session, None)
    with pytest.raises(FailedPreconditionError):
        ctrl.interactive.apply_cross_selection()


def test_no_cloud_and_disposed_owner_fail_without_session(cross_controller):
    ctrl = cross_controller
    for name in list(ctrl.state.spectrums):
        ctrl.set_points(name, np.empty(0), np.empty(0))
    with pytest.raises(FailedPreconditionError):
        ctrl.interactive.begin_cross_selection()
    assert ctrl.interactive.current_cross_selection() is None
    ctrl.interactive.dispose()
    with pytest.raises(FailedPreconditionError):
        ctrl.interactive.begin_cross_selection()


@pytest.mark.parametrize("publish_before_failure", [False, True])
def test_apply_failure_keeps_input_and_releases_self_publication_guard(
    cross_controller, publish_before_failure
):
    ctrl = cross_controller

    def fail_selection(selected, distance):
        if publish_before_failure:
            ctrl.set_selection(selected, distance)
        raise OSError("selection publication unavailable")

    owner = FluxDepInteractiveOwner(
        ctrl.state,
        ctrl.bus,
        ManualOwnerScheduler(),
        ports=FluxDepInteractivePorts(
            background=None,
            publish_alignment=ctrl.set_alignment,
            publish_points=ctrl.set_points,
            derive_pointcloud=ctrl.derive_pointcloud,
            publish_selection=fail_selection,
        ),
    )
    try:
        context = owner.begin_cross_selection()
        context.plugin.clear.execute(context.session, None)
        with pytest.raises(OSError, match="publication unavailable"):
            owner.apply_cross_selection()
        assert owner.current_cross_selection() is context
        assert context.session.undo().selected.all()
        if publish_before_failure:
            assert not ctrl.state.selection_mask().any()
        ctrl.set_selection(np.ones(4, dtype=bool), 0.0)
        assert owner.current_cross_selection() is None
        with pytest.raises(FailedPreconditionError):
            context.plugin.clear.execute(context.session, None)
    finally:
        owner.dispose()


def test_reopened_cloud_backgrounds_follow_replaced_spectrum_signals(
    cross_controller,
):
    from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm

    ctrl = cross_controller
    first = ctrl.interactive.begin_cross_selection()
    before = {bg.name: bg.real_signals for bg in first.plugin.inputs.backgrounds}
    ctrl.interactive.cancel()
    reopened = ctrl.interactive.begin_cross_selection()
    for background in reopened.plugin.inputs.backgrounds:
        np.testing.assert_array_equal(background.real_signals, before[background.name])
    ctrl.interactive.cancel()

    entry = ctrl.state.spectrums["a"]
    signals = np.arange(15, dtype=np.float64).reshape(5, 3) * (1 + 1j)
    ctrl.state.put_spectrum(replace(entry, raw={**entry.raw, "signals": signals}))
    replaced = ctrl.interactive.begin_cross_selection()
    background = next(bg for bg in replaced.plugin.inputs.backgrounds if bg.name == "a")
    np.testing.assert_array_equal(
        background.real_signals, cast2real_and_norm(signals, use_phase=True)
    )


@pytest.fixture
def column_cloud_controller():
    """One aligned spectrum: same-flux columns of three, singles between them.

    Raw axes span flux [0, 1] and 4-5 GHz, so normalized distances equal the
    point offsets below. At min_distance 0.1 a second downsample of the kept
    points drops more of them, so a reopen seeded from kept would not round-trip.
    """
    columns = np.linspace(0.0, 1.0, 6)
    fluxs = np.concatenate([np.repeat(columns, 3), columns[:-1] + 0.02])
    offsets = np.concatenate([np.tile([0.0, 0.1, 0.2], 6), np.full(5, 0.05)])
    freqs = 4.0 + offsets
    state = FluxDepState()
    state.put_spectrum(
        SpectrumEntry(
            name="a",
            spec_type="TwoTone",
            raw={
                "dev_values": np.array([0.0, 0.5, 1.0]),
                "fluxs": np.array([0.0, 0.5, 1.0]),
                "freqs": np.array([4.0, 4.5, 5.0]),
                "signals": np.ones((3, 3), dtype=np.complex128),
            },
            points={"dev_values": fluxs.copy(), "fluxs": fluxs, "freqs": freqs},
            flux_half=0.5,
            flux_int=1.0,
            flux_period=1.0,
            aligned=True,
            points_completed=True,
        )
    )
    ctrl = Controller(state)
    yield ctrl
    ctrl.interactive.dispose()


def _publish_brushed_draft(ctrl: Controller, min_distance: float) -> np.ndarray:
    """Erase the first column, finish at min_distance, return the draft's brush."""
    context = ctrl.interactive.begin_cross_selection()
    context.plugin.stroke.execute(
        context.session, BrushStroke((BrushPoint(0.0, 4.1),), 0.1, "erase")
    )
    context.plugin.set_min_distance.execute(context.session, min_distance)
    brush = context.session.snapshot().selected.copy()
    ctrl.interactive.finish_cross_selection()
    return brush


def test_unchanged_reopened_draft_finishes_to_the_published_kept_mask(
    column_cloud_controller,
):
    ctrl = column_cloud_controller
    brush = _publish_brushed_draft(ctrl, 0.1)
    kept = ctrl.state.selection_mask()
    assert kept is not None and kept.sum() < brush.sum()

    reopened = ctrl.interactive.begin_cross_selection()
    np.testing.assert_array_equal(reopened.session.snapshot().selected, brush)
    assert reopened.session.snapshot().min_distance == 0.1
    ctrl.interactive.finish_cross_selection()

    np.testing.assert_array_equal(ctrl.state.selection_mask(), kept)
    np.testing.assert_array_equal(ctrl.state.brush_mask(), brush)


def test_reopened_draft_recomputes_a_changed_distance_from_the_brush(
    column_cloud_controller,
):
    ctrl = column_cloud_controller
    brush = _publish_brushed_draft(ctrl, 0.1)

    reopened = ctrl.interactive.begin_cross_selection()
    reopened.plugin.set_min_distance.execute(reopened.session, 0.05)
    expected = analyze_cross_selection(
        reopened.plugin.inputs,
        make_cross_selection_state(
            reopened.plugin.inputs, selected=brush, min_distance=0.05
        ),
    ).selected
    ctrl.interactive.finish_cross_selection()

    np.testing.assert_array_equal(ctrl.state.selection_mask(), expected)
    np.testing.assert_array_equal(ctrl.state.brush_mask(), brush)
    assert ctrl.state.selection.min_distance == 0.05


def test_discarding_a_reopened_draft_keeps_published_kept_and_brush(
    column_cloud_controller,
):
    ctrl = column_cloud_controller
    brush = _publish_brushed_draft(ctrl, 0.1)
    kept = ctrl.state.selection_mask()

    reopened = ctrl.interactive.begin_cross_selection()
    reopened.plugin.clear.execute(reopened.session, None)
    reopened.plugin.set_min_distance.execute(reopened.session, 0.0)
    ctrl.interactive.cancel_selection()

    np.testing.assert_array_equal(ctrl.state.selection_mask(), kept)
    np.testing.assert_array_equal(ctrl.state.brush_mask(), brush)
    assert ctrl.state.selection.min_distance == 0.1


def test_restored_spectrum_reopens_all_selected_with_the_published_distance(
    column_cloud_controller, tmp_path
):
    ctrl = column_cloud_controller
    _publish_brushed_draft(ctrl, 0.1)
    filepath = ctrl.export_spectrums(str(tmp_path / "checkpoint.hdf5"))
    ctrl.remove_spectrum("a")
    ctrl.load_processed_spectrums(filepath)

    reopened = ctrl.interactive.begin_cross_selection()

    assert reopened.session.snapshot().selected.all()
    assert reopened.session.snapshot().selected.size == 23
    assert reopened.session.snapshot().min_distance == 0.1
