"""MCP tab_interact and historical projection over the real synthetic plugin.

The transport calls registered GUI handlers in process; it opens no port and
launches no process. Session/control transport policy has its own owner tests.
"""

import json
from pathlib import Path

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.remote.param_spec import validate_params

from tests.gui.app.measure._range_find_adapter import (
    PickerParams,
    range_application,
)
from tests.gui.app.measure.remote._helpers import call_mcp_with_qt
from tests.mcp.measure._support import make_client


def test_mcp_edits_and_finish_preserve_final_controls_selection_and_mapping(
    tmp_path: Path, qapp: QApplication
):
    with range_application(tmp_path, qapp) as app:
        remote = RemoteControlAdapter(
            app.ctrl,
            ControlOptions(port=0),
            app.ctrl.owner_scheduler,
            render_view=app.window,
        )
        app.ctrl.run_analyze_control.analyze(app.tab_id, PickerParams())

        def respond(method, params):
            entry = METHOD_REGISTRY[method]

            def invoke():
                return dict(
                    entry.handler(remote, validate_params(entry.params, params))
                )

            return (
                invoke()
                if entry.off_main_thread
                else app.ctrl.owner_scheduler.call(invoke)
            )

        client = make_client(tmp_path, respond)
        try:

            def interact(command=None, args=None):
                payload = (
                    {}
                    if command is None
                    else {"payload": {"command": command, "args": args or {}}}
                )
                return call_mcp_with_qt(
                    client.tools, "tab_interact", {"tab": app.tab_id, **payload}
                )

            assert interact()["state"]["selected"][0]["component"] == "west"
            changed = interact(
                "set_parameters",
                dict(
                    smoothing_sigma=0.0,
                    prominence_factor=5.0,
                    max_fwhm=20.0,
                    min_separation=2.0,
                    max_dips=1,
                ),
            )
            assert len(changed["state"]["selected"]) == 1
            interact("remove_dip", {"index": 0})
            interact("add_dip", {"frequency": 5040})
            changed = interact("assign_dip", {"index": 0, "component": "east"})
            assert changed["state"]["selected"][0]["freq"] == 5040
            assert changed["state"]["selected"][0]["component"] == "east"
            done = interact("done")
            assert done["status"] == "finished", str(done)
            assert done["steps"]["analysis_save"]["primary"]["status"] == "saved"
            artifact = done["artifacts"]["analysis"]["results"]["members"][
                "analysis:analysis"
            ][0]
            summary = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))[
                "summary"
            ]
            assert summary["options"] == dict(
                smoothing_sigma=0.0,
                prominence_factor=5.0,
                max_fwhm=20.0,
                min_separation=2.0,
                max_dips=1,
            )
            assert summary["selected"] == [
                {"freq": 5040.0, "component": "east", "estimate": None}
            ]
        finally:
            client.context.session.close()
