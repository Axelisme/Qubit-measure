"""Joint-cloud owner invalidation and nonterminal selection publication."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke
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
