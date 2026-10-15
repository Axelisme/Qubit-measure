from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.remote.handlers.arb_waveform import (
    h_arb_waveform_list,
)
from zcu_tools.gui.app.measure.remote.handlers.connection_device import (
    h_device_connect,
)
from zcu_tools.gui.app.measure.remote.handlers.run_save import (
    h_tab_load_data,
    h_tab_run_start,
)
from zcu_tools.gui.app.measure.remote.handlers.view import (
    h_dialog_screenshot,
    h_tab_get_figure,
)
from zcu_tools.gui.app.measure.remote.handlers.writeback import (
    h_tab_writeback_set,
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

from ._point_application import PointApplication

WireTuple = tuple[ErrorCode, str, str, dict | None]


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("context.ml_del_module", {"name": "Q1.pulses.seed"}),
        ("context.ml_del_waveform", {"name": "Q1.waveforms.seed"}),
        (
            "context.ml_rename_module",
            {"old": "Q1.pulses.seed", "new": "Q1.pulses.moved"},
        ),
        (
            "context.ml_rename_waveform",
            {"old": "Q1.waveforms.seed", "new": "Q1.waveforms.moved"},
        ),
    ],
)
def test_library_structure_empty_context_uses_owner_error_projection(
    application: PointApplication, method: str, params: dict[str, str]
) -> None:
    reply = application.rpc(method, params)
    assert reply["ok"] is False, reply
    assert reply["error"] == {
        "code": ErrorCode.PRECONDITION_FAILED,
        "message": "Select or create a working point first",
        "reason": "no_context",
    }
    assert application.controller.get_session_env().parameters is None


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

    device_control = MagicMock()
    device_control.start_connect_device.side_effect = FailedPreconditionError(
        "Device busy"
    )
    device_adapter = SimpleNamespace(device_control=device_control)

    arb_ctrl = MagicMock()
    arb_ctrl.arb_waveforms.list_data_keys.side_effect = FailedPreconditionError(
        "No project database_path is configured.", reason_code="no_project"
    )
    arb_adapter = SimpleNamespace(ctrl=arb_ctrl)

    render_view = MagicMock()
    render_view.take_figure_screenshot_for_subtab.side_effect = FailedPreconditionError(
        "tab 't1' analysis has no figure yet"
    )
    tab_control = MagicMock()
    tab_control.has_tab.return_value = True
    view_adapter = SimpleNamespace(render_view=render_view, tab_control=tab_control)

    writeback_control = MagicMock()
    writeback_control.has_tab.return_value = True
    writeback_control.set_writeback_item_for_pane.side_effect = InvalidInputError(
        "bad facet"
    )
    writeback_adapter = SimpleNamespace(writeback_control=writeback_control)

    cases: list[tuple[Callable[[], object], WireTuple]] = [
        (
            lambda: h_tab_run_start(
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
                cast(Any, load_adapter), {"tab_id": "t1", "data_path": "bad.h5"}
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "Bad data file",
                "invalid_data_file",
                None,
            ),
        ),
        (
            lambda: h_device_connect(
                cast(Any, device_adapter),
                {"type_name": "fake", "name": "flux", "address": "mock"},
            ),
            (ErrorCode.PRECONDITION_FAILED, "Device busy", "", None),
        ),
        (
            lambda: h_arb_waveform_list(cast(Any, arb_adapter), {}),
            (
                ErrorCode.PRECONDITION_FAILED,
                "No project database_path is configured.",
                "no_project",
                None,
            ),
        ),
        (
            lambda: h_tab_get_figure(
                cast(Any, view_adapter), {"tab_id": "t1", "subtab_id": "analysis"}
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "tab 't1' analysis has no figure yet",
                "",
                None,
            ),
        ),
        (
            lambda: h_tab_writeback_set(
                cast(Any, writeback_adapter),
                {
                    "tab_id": "t1",
                    "subtab_id": "analysis",
                    "id": "param-0",
                    "selected": True,
                },
            ),
            (ErrorCode.INVALID_PARAMS, "bad facet", "", None),
        ),
    ]

    for call, expected in cases:
        assert _remote_error(call) == expected


def test_dialog_screenshot_expected_errors_use_producer_taxonomy() -> None:
    render_view = MagicMock()
    adapter = cast(Any, SimpleNamespace(render_view=render_view))

    assert _remote_error(lambda: h_dialog_screenshot(adapter, {"name": "unknown"})) == (
        ErrorCode.INVALID_PARAMS,
        "unknown dialog name: 'unknown'",
        "",
        None,
    )
    # The retired startup dialog identity is not an alias for setup.
    assert _remote_error(lambda: h_dialog_screenshot(adapter, {"name": "startup"})) == (
        ErrorCode.INVALID_PARAMS,
        "unknown dialog name: 'startup'",
        "",
        None,
    )
    render_view.take_dialog_screenshot.assert_not_called()

    render_view.take_dialog_screenshot.side_effect = FailedPreconditionError(
        "dialog 'setup' is not open"
    )
    assert _remote_error(lambda: h_dialog_screenshot(adapter, {"name": "setup"})) == (
        ErrorCode.PRECONDITION_FAILED,
        "dialog 'setup' is not open",
        "",
        None,
    )
