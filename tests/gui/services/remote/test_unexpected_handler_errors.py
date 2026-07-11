from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote.handlers.analysis import _h_tab_analyze
from zcu_tools.gui.app.main.services.remote.handlers.connection_device import (
    _h_device_connect,
    _h_soc_connect,
)
from zcu_tools.gui.app.main.services.remote.handlers.context import _h_value_read
from zcu_tools.gui.app.main.services.remote.handlers.editor import (
    _h_editor_commit,
    _h_editor_set_field,
)
from zcu_tools.gui.app.main.services.remote.handlers.run_save import (
    _h_tab_load_data,
    _h_tab_run_start,
    _h_tab_save_data,
)
from zcu_tools.gui.app.main.services.remote.handlers.view import (
    _h_tab_get_current_figure,
)
from zcu_tools.gui.app.main.services.remote.handlers.writeback import (
    _h_tab_writeback_apply,
    _h_tab_writeback_set,
)
from zcu_tools.gui.session.value_lookup import ProviderError


@dataclass(frozen=True)
class _AnalyzeParams:
    repetitions: int = 1


def _assert_escapes_unchanged(call: Any, error: BaseException) -> None:
    with pytest.raises(type(error)) as exc_info:
        call()
    assert exc_info.value is error


def test_run_and_analyze_runtime_errors_escape_handlers_unchanged() -> None:
    run_error = RuntimeError("run programmer bug")
    run_control = MagicMock()
    run_control.has_tab.return_value = True
    run_control.start_run.side_effect = run_error
    _assert_escapes_unchanged(
        lambda: _h_tab_run_start(
            cast(Any, SimpleNamespace(run_analyze_control=run_control)),
            {"tab_id": "t1"},
        ),
        run_error,
    )

    analyze_error = RuntimeError("analyze programmer bug")
    analyze_control = MagicMock()
    analyze_control.has_tab.return_value = True
    analyze_control.get_tab_snapshot.return_value = SimpleNamespace(
        interaction=SimpleNamespace(has_run_result=True),
        analyze_params=_AnalyzeParams(),
    )
    analyze_control.analyze.side_effect = analyze_error
    _assert_escapes_unchanged(
        lambda: _h_tab_analyze(
            cast(Any, SimpleNamespace(run_analyze_control=analyze_control)),
            {"tab_id": "t1", "updates": {}},
        ),
        analyze_error,
    )


def test_device_runtime_error_escapes_handler_unchanged() -> None:
    error = RuntimeError("device programmer bug")
    control = MagicMock()
    control.start_connect_device.side_effect = error
    _assert_escapes_unchanged(
        lambda: _h_device_connect(
            cast(Any, SimpleNamespace(device_control=control)),
            {
                "type_name": "fake",
                "name": "flux",
                "address": "mock",
                "remember": True,
            },
        ),
        error,
    )


def test_provider_error_escapes_value_handler_unchanged() -> None:
    error = ProviderError("ctx.value", "provider", RuntimeError("provider bug"))
    control = MagicMock()
    control.read_value_source.side_effect = error
    _assert_escapes_unchanged(
        lambda: _h_value_read(
            cast(Any, SimpleNamespace(context_control=control)),
            {"key": "ctx.value", "type": None},
        ),
        error,
    )


@pytest.mark.parametrize("handler_name", ["set", "commit"])
def test_editor_runtime_errors_escape_handlers_unchanged(handler_name: str) -> None:
    error = RuntimeError(f"editor {handler_name} programmer bug")
    ctrl = MagicMock()
    if handler_name == "set":
        ctrl.owner_of_editor.return_value = None
        ctrl.cfg_editor_set_field.side_effect = error
        call = lambda: _h_editor_set_field(
            cast(Any, SimpleNamespace(ctrl=ctrl)),
            {"editor_id": "e1", "path": "pulse.freq", "value": 1.0},
        )
    else:
        ctrl.commit_cfg_editor.side_effect = error
        call = lambda: _h_editor_commit(
            cast(Any, SimpleNamespace(ctrl=ctrl)),
            {"editor_id": "e1", "name": "module"},
        )
    _assert_escapes_unchanged(call, error)


def test_load_and_save_runtime_errors_escape_handlers_unchanged() -> None:
    load_error = RuntimeError("load programmer bug")
    load_control = MagicMock()
    load_control.has_tab.return_value = True
    load_control.load_tab_result.side_effect = load_error
    _assert_escapes_unchanged(
        lambda: _h_tab_load_data(
            cast(Any, SimpleNamespace(run_analyze_control=load_control)),
            {"tab_id": "t1", "data_path": "result.hdf5"},
        ),
        load_error,
    )

    save_error = OSError("save disk failed")
    save_control = MagicMock()
    save_control.save_data.side_effect = save_error
    _assert_escapes_unchanged(
        lambda: _h_tab_save_data(
            cast(Any, SimpleNamespace(save_control=save_control)),
            {"tab_id": "t1", "data_path": None, "comment": ""},
        ),
        save_error,
    )


def test_figure_runtime_error_escapes_handler_unchanged() -> None:
    error = RuntimeError("canvas invariant")
    render_view = MagicMock()
    render_view.take_figure_screenshot.side_effect = error
    _assert_escapes_unchanged(
        lambda: _h_tab_get_current_figure(
            cast(Any, SimpleNamespace(render_view=render_view)),
            {"tab_id": "t1", "out_path": None},
        ),
        error,
    )


@pytest.mark.parametrize("handler_name", ["set", "apply"])
def test_writeback_runtime_errors_escape_handlers_unchanged(
    handler_name: str,
) -> None:
    error = RuntimeError(f"writeback {handler_name} programmer bug")
    control = MagicMock()
    control.has_tab.return_value = True
    if handler_name == "set":
        control.set_writeback_item.side_effect = error
        call = lambda: _h_tab_writeback_set(
            cast(Any, SimpleNamespace(writeback_control=control)),
            {
                "tab_id": "t1",
                "id": "item-1",
                "selected": True,
                "target_name": None,
                "proposed_value": None,
                "edits": None,
            },
        )
    else:
        control.apply_writeback.side_effect = error
        call = lambda: _h_tab_writeback_apply(
            cast(Any, SimpleNamespace(writeback_control=control)),
            {"tab_id": "t1"},
        )
    _assert_escapes_unchanged(call, error)


def test_soc_connect_runtime_error_escapes_handler_unchanged() -> None:
    error = RuntimeError("SoC connection failed")
    ctrl = MagicMock()
    ctrl.connect_sync.side_effect = error
    _assert_escapes_unchanged(
        lambda: _h_soc_connect(
            cast(Any, SimpleNamespace(ctrl=ctrl)),
            {"kind": "mock"},
        ),
        error,
    )
