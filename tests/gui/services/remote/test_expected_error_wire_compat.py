from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError
from zcu_tools.gui.app.main.services.guard import GuardError
from zcu_tools.gui.app.main.services.load import LoadDataError
from zcu_tools.gui.app.main.services.remote.handlers.arb_waveform import (
    _h_arb_waveform_list,
)
from zcu_tools.gui.app.main.services.remote.handlers.connection_device import (
    _h_device_connect,
    _h_startup_apply,
)
from zcu_tools.gui.app.main.services.remote.handlers.context import (
    _h_context_md_del_attr,
    _h_context_md_set_attr,
)
from zcu_tools.gui.app.main.services.remote.handlers.editor import _h_editor_new
from zcu_tools.gui.app.main.services.remote.handlers.run_save import (
    _h_tab_load_data,
    _h_tab_run_start,
)
from zcu_tools.gui.app.main.services.remote.handlers.view import (
    _h_dialog_screenshot,
    _h_tab_get_current_figure,
)
from zcu_tools.gui.app.main.services.remote.handlers.writeback import (
    _h_tab_writeback_set,
)
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
from zcu_tools.gui.result_scope import ResultScopeError

WireTuple = tuple[ErrorCode, str, str, dict | None]


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
    run_adapter = SimpleNamespace(run_analyze_control=run_control)

    load_control = MagicMock()
    load_control.has_tab.return_value = True
    load_control.load_tab_result.side_effect = LoadDataError(
        "Bad data file", reason_code="invalid_data_file"
    )
    load_adapter = SimpleNamespace(run_analyze_control=load_control)

    editor_ctrl = MagicMock()
    editor_ctrl.open_cfg_editor.side_effect = CfgEditorError("unknown module")
    editor_adapter = SimpleNamespace(ctrl=editor_ctrl)

    context_control = MagicMock()
    context_control.set_md_attr.side_effect = FailedPreconditionError("No context")
    context_control.del_md_attr.side_effect = FailedPreconditionError(
        "MetaDict has no attribute missing"
    )
    context_adapter = SimpleNamespace(context_control=context_control)

    device_control = MagicMock()
    device_control.start_connect_device.side_effect = FailedPreconditionError(
        "Device busy"
    )
    device_adapter = SimpleNamespace(device_control=device_control)

    arb_ctrl = MagicMock()
    arb_ctrl.list_arb_waveforms.side_effect = FailedPreconditionError(
        "No project database_path is configured.", reason_code="no_project"
    )
    arb_adapter = SimpleNamespace(ctrl=arb_ctrl)

    render_view = MagicMock()
    render_view.take_figure_screenshot.side_effect = FailedPreconditionError(
        "tab 't1' has no figure yet"
    )
    view_adapter = SimpleNamespace(render_view=render_view)

    writeback_control = MagicMock()
    writeback_control.has_tab.return_value = True
    writeback_control.set_writeback_item.side_effect = InvalidInputError("bad facet")
    writeback_adapter = SimpleNamespace(writeback_control=writeback_control)

    cases: list[tuple[Callable[[], object], WireTuple]] = [
        (
            lambda: _h_tab_run_start(cast(Any, run_adapter), {"tab_id": "t1"}),
            (
                ErrorCode.PRECONDITION_FAILED,
                "No run result",
                "no_run_result",
                None,
            ),
        ),
        (
            lambda: _h_tab_load_data(
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
            lambda: _h_editor_new(
                cast(Any, editor_adapter),
                {"item_kind": "module", "from_name": "missing"},
            ),
            (ErrorCode.INVALID_PARAMS, "unknown module", "", None),
        ),
        (
            lambda: _h_context_md_set_attr(
                cast(Any, context_adapter), {"key": "x", "value": 1}
            ),
            (ErrorCode.PRECONDITION_FAILED, "No context", "", None),
        ),
        (
            lambda: _h_context_md_del_attr(
                cast(Any, context_adapter), {"key": "missing"}
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "MetaDict has no attribute missing",
                "",
                None,
            ),
        ),
        (
            lambda: _h_device_connect(
                cast(Any, device_adapter),
                {"type_name": "fake", "name": "flux", "address": "mock"},
            ),
            (ErrorCode.PRECONDITION_FAILED, "Device busy", "", None),
        ),
        (
            lambda: _h_arb_waveform_list(cast(Any, arb_adapter), {}),
            (
                ErrorCode.PRECONDITION_FAILED,
                "No project database_path is configured.",
                "no_project",
                None,
            ),
        ),
        (
            lambda: _h_tab_get_current_figure(
                cast(Any, view_adapter), {"tab_id": "t1"}
            ),
            (
                ErrorCode.PRECONDITION_FAILED,
                "tab 't1' has no figure yet",
                "",
                None,
            ),
        ),
        (
            lambda: _h_tab_writeback_set(
                cast(Any, writeback_adapter),
                {"tab_id": "t1", "id": "md-0", "selected": True},
            ),
            (ErrorCode.INVALID_PARAMS, "bad facet", "", None),
        ),
    ]

    for call, expected in cases:
        assert _remote_error(call) == expected


def test_dialog_screenshot_expected_errors_use_producer_taxonomy() -> None:
    render_view = MagicMock()
    adapter = SimpleNamespace(render_view=render_view)

    assert _remote_error(
        lambda: _h_dialog_screenshot(cast(Any, adapter), {"name": "unknown"})
    ) == (ErrorCode.INVALID_PARAMS, "unknown dialog name: 'unknown'", "", None)
    render_view.take_dialog_screenshot.assert_not_called()

    render_view.take_dialog_screenshot.side_effect = FailedPreconditionError(
        "dialog 'setup' is not open"
    )
    assert _remote_error(
        lambda: _h_dialog_screenshot(cast(Any, adapter), {"name": "setup"})
    ) == (
        ErrorCode.PRECONDITION_FAILED,
        "dialog 'setup' is not open",
        "",
        None,
    )


@pytest.mark.parametrize(
    ("category", "reason", "code"),
    [
        (
            ExpectedErrorCategory.INVALID_INPUT,
            "params_read_failed",
            ErrorCode.INVALID_PARAMS,
        ),
        (
            ExpectedErrorCategory.FAILED_PRECONDITION,
            "scope_prefix_is_not_semantics",
            ErrorCode.PRECONDITION_FAILED,
        ),
    ],
)
def test_result_scope_projection_uses_category_not_reason_prefix(
    category: ExpectedErrorCategory,
    reason: str,
    code: ErrorCode,
) -> None:
    ctrl = MagicMock()
    ctrl.apply_startup_project.side_effect = ResultScopeError(
        "scope failure", category=category, reason_code=reason
    )
    adapter = SimpleNamespace(ctrl=ctrl)

    actual = _remote_error(
        lambda: _h_startup_apply(
            cast(Any, adapter),
            {
                "chip_name": "chip",
                "qub_name": "qubit",
                "res_name": "resonator",
            },
        )
    )

    assert actual == (code, "scope failure", reason, None)
