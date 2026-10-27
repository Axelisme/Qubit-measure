"""Public numeric plugin/session contracts, independent of lab definitions."""

from collections.abc import Iterator

import numpy as np
import pytest
from zcu_tools.analysis.dips import DipFinderOptions
from zcu_tools.gui.app.measure.range_find_plugin import RangeFindPlugin, RangeFindState
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive import Session
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@pytest.fixture
def picker() -> Iterator[
    tuple[RangeFindPlugin[RangeFindState], Session[RangeFindState]]
]:
    freqs = np.arange(101, dtype=np.float64) + 5000
    magnitude = 1 + 0.001 * np.sin(np.arange(101, dtype=np.float64))
    magnitude[20], magnitude[70] = 0.3, 0.5
    plugin = RangeFindPlugin(
        freqs,
        magnitude,
        components=("west", "east"),
        options=DipFinderOptions(max_dips=2),
        build_result=lambda state: state,
    )
    session = plugin.open(ManualOwnerScheduler())
    try:
        yield plugin, session
    finally:
        session.dispose()


def test_initial_guess_selects_narrow_dips_and_pairs_caller_labels(picker):
    plugin, session = picker
    selected = session.snapshot().selected
    assert [(dip.freq, dip.component) for dip in selected] == [
        (5020, "west"),
        (5070, "east"),
    ]
    assert plugin.finish(session).selected == selected


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"max_dips": 1}, [5020]),
        ({"min_separation": 60.0}, [5020]),
        ({"max_fwhm": 0.5}, []),
        ({"prominence_factor": 1e8}, []),
        ({"smoothing_sigma": 4.0, "max_fwhm": 2.0}, []),
    ],
)
def test_parameter_commands_recompute_the_shared_automatic_guess(
    picker, changed, expected
):
    plugin, session = picker
    args = dict(
        smoothing_sigma=0.0,
        prominence_factor=5.0,
        max_fwhm=25.0,
        min_separation=1.0,
        max_dips=2,
    )
    args.update(changed)
    plugin.execute_command(session, "set_parameters", args)
    state = session.snapshot()
    assert [dip.freq for dip in state.selected] == expected
    for name, value in changed.items():
        assert getattr(state.options, name) == value
    session.undo()
    assert [dip.freq for dip in session.snapshot().selected] == [5020, 5070]


def test_manual_selection_and_arbitrary_assignments_survive_finish(picker):
    plugin, session = picker
    plugin.execute_command(session, "remove_dip", {"index": 0})
    assert [dip.freq for dip in session.snapshot().selected] == [5070]
    plugin.execute_command(session, "add_dip", {"frequency": 5040.2})
    assert [dip.freq for dip in session.snapshot().selected] == [5040, 5070]
    plugin.execute_command(session, "assign_dip", {"index": 0, "component": "east"})
    assert [dip.component for dip in session.snapshot().selected] == ["east", None]
    plugin.execute_command(session, "assign_dip", {"index": 1, "component": "west"})
    result = plugin.finish(session)
    assert [(dip.freq, dip.component) for dip in result.selected] == [
        (5040, "east"),
        (5070, "west"),
    ]
    assert result.selected[0].estimate is None
    assert result.selected[1].estimate is not None
    with pytest.raises(FailedPreconditionError):
        plugin.execute_command(session, "add_dip", {"frequency": 5020.0})


def test_invalid_edits_preserve_state_and_empty_selection_cannot_finish(picker):
    plugin, session = picker
    original = session.snapshot()
    for command, args in [
        ("add_dip", {"frequency": float("nan")}),
        ("add_dip", {"frequency": 4000}),
        ("remove_dip", {"index": -1}),
        ("assign_dip", {"index": 0, "component": "unknown"}),
        ("set_parameters", {"smoothing_sigma": True}),
    ]:
        with pytest.raises(InvalidInputError):
            plugin.execute_command(session, command, args)
        assert session.snapshot() == original
    plugin.execute_command(session, "remove_dip", {"index": 1})
    plugin.execute_command(session, "remove_dip", {"index": 0})
    with pytest.raises(FailedPreconditionError, match="Select at least one"):
        plugin.finish(session)
    plugin.execute_command(session, "add_dip", {"frequency": 5020})
    assert plugin.finish(session).selected[0].freq == 5020
