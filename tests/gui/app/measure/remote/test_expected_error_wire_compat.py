from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.remote.handlers.run_save import (
    h_tab_load_data,
    h_tab_run_start,
)
from zcu_tools.gui.app.measure.remote.handlers.view import (
    h_tab_get_figure,
)
from zcu_tools.gui.app.measure.services.guard import GuardError
from zcu_tools.gui.app.measure.services.load import LoadDataError
from zcu_tools.gui.expected_error import (
    ExpectedError,
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    RemoteError,
    remote_error_from_expected,
)

WireTuple = tuple[ErrorCode, str, str, dict[str, object] | None]


class _ExpectedWithIncidentalData(InvalidInputError):
    def __init__(self) -> None:
        super().__init__("bad input", reason_code="bad_field")
        self.data = {"must_not": "cross the generic boundary"}


class _StructuralExpectedDuck(Exception):
    category = ExpectedErrorCategory.INVALID_INPUT
    reason_code = "duck_reason"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            InvalidInputError("bad input", reason_code="bad_field"),
            (ErrorCode.INVALID_PARAMS, "bad input", "bad_field", None),
        ),
        (
            FailedPreconditionError("not ready", reason_code="no_context"),
            (
                ErrorCode.PRECONDITION_FAILED,
                "not ready",
                "no_context",
                None,
            ),
        ),
        (
            _ExpectedWithIncidentalData(),
            (ErrorCode.INVALID_PARAMS, "bad input", "bad_field", None),
        ),
    ],
)
def test_generic_expected_error_translation_has_exact_wire_contract(
    exc: ExpectedError,
    expected: WireTuple,
) -> None:
    translated = remote_error_from_expected(exc)

    assert (
        translated.code,
        translated.message,
        translated.reason,
        translated.data,
    ) == expected


def test_generic_translation_rejects_structural_expected_error_duck() -> None:
    # tolerated(cast): A non-nominal duck must reach the ExpectedError runtime rejection.
    duck = cast(ExpectedError, _StructuralExpectedDuck("structural"))

    with pytest.raises(TypeError, match="nominal ExpectedError"):
        remote_error_from_expected(duck)


def _remote_error(call: Callable[[], object]) -> WireTuple:
    try:
        call()
    except RemoteError as exc:
        return exc.code, exc.message, exc.reason, exc.data
    except ExpectedError as expected:
        translated = remote_error_from_expected(expected)
        return (
            translated.code,
            translated.message,
            translated.reason,
            translated.data,
        )
    raise AssertionError("call did not raise a remote or expected error")


def test_existing_handler_error_projection_is_wire_equivalent() -> None:
    run_control = MagicMock()
    run_control.has_tab.return_value = True
    run_control.start_run.side_effect = GuardError(
        "No run result", reason_code="no_run_result"
    )
    run_adapter = SimpleNamespace(run_analyze_control=run_control, render_view=None)

    load_control = MagicMock()
    load_control.has_tab.return_value = True
    load_control.load_tab_result.side_effect = LoadDataError(
        "Bad data file", reason_code="invalid_data_file"
    )
    load_adapter = SimpleNamespace(run_analyze_control=load_control)

    render_view = MagicMock()
    render_view.take_figure_screenshot_for_subtab.side_effect = FailedPreconditionError(
        "tab 't1' analysis has no figure yet"
    )
    tab_control = MagicMock()
    tab_control.has_tab.return_value = True
    view_adapter = SimpleNamespace(render_view=render_view, tab_control=tab_control)

    cases: list[tuple[Callable[[], object], WireTuple]] = [
        (
            lambda: h_tab_run_start(
                # tolerated(cast): The run-control namespace is not a RemoteControlAdapter.
                cast(Any, run_adapter),
                {"tab_id": "t1", "expected": {"cfg_id": "cfg-t1", "revision": "0"}},
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "No run result",
                "no_run_result",
                None,
            ),
        ),
        (
            lambda: h_tab_load_data(
                # tolerated(cast): The load-control namespace is not a RemoteControlAdapter.
                cast(Any, load_adapter),
                {"tab_id": "t1", "data_path": "bad.h5"},
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "Bad data file",
                "invalid_data_file",
                None,
            ),
        ),
        (
            lambda: h_tab_get_figure(
                # tolerated(cast): The render/tab namespace is not a RemoteControlAdapter.
                cast(Any, view_adapter),
                {"tab_id": "t1", "subtab_id": "analysis"},
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "tab 't1' analysis has no figure yet",
                "",
                None,
            ),
        ),
    ]

    for call, expected in cases:
        assert _remote_error(call) == expected
