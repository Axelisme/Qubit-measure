"""Recipe promises through the shipped tool table and GUI transport seam."""

import base64
from copy import deepcopy
from typing import Any

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from ._support import make_client

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _scalar(value: object) -> dict[str, Any]:
    return {
        "kind": "scalar",
        "valid": True,
        "input": {
            "mode": "direct",
            "raw": value,
            "resolved": value,
            "error": None,
            "validation_error": None,
        },
    }


def _section(**children: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "section", "valid": True, "children": children}


class LookbackGui:
    """GUI collaborator with distinct Run, save and analysis wire identities."""

    def __init__(self):
        self.publication: dict[str, Any] = {
            "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
            "status": "Valid",
            "source_basis": [],
            "diagnostics": [],
            "tree": _section(
                rounds=_scalar(3),
                modules=_section(
                    reset={"kind": "reference", "valid": True, "ref": None},
                    init_pulse={"kind": "reference", "valid": True, "ref": None},
                    readout={
                        "kind": "reference",
                        "valid": True,
                        "ref": None,
                        "error": None,
                        "children": {
                            "pulse_cfg": _section(freq=_scalar(5000.0)),
                            "ro_cfg": _section(
                                ro_freq=_scalar(5000.0),
                                ro_length=_scalar(2.0),
                                trig_offset=_scalar(0.1),
                            ),
                        },
                    },
                ),
            ),
        }
        self.ran = False
        self.raw_saved = False

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            "context.snapshot": {
                "label": "sample",
                "md": {},
                "ml": {"modules": {}, "waveforms": {}},
            },
            "tab.new": {"tab_id": "t"},
            "soc.info": {"connected": True, "cfg": {}},
            "device.list": {"devices": [{"name": "bias"}]},
            "device.snapshot": {"snapshot": {"name": "bias", "info": {"value": 0.0}}},
            "tab.snapshot": {
                "tabs": [
                    {
                        "tab_id": "t",
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                        "result_state": {
                            "available": self.ran,
                            "revision": 1,
                            "source_operation_id": 71 if self.ran else None,
                        },
                    }
                ]
            },
        }

    def _edit(self, params: dict[str, Any]) -> None:
        assert params["expected"] == self.publication["cfg_ref"]
        for edit in params["edits"]:
            node = self.publication["tree"]
            for part in edit["path"]:
                node = node["children"][part]
            if node["kind"] == "reference":
                node["ref"] = edit["value"].get("__ref") if edit["value"] else None
            else:
                node.update(_scalar(edit["value"]))
        revision = int(self.publication["cfg_ref"]["revision"]) + 1
        self.publication["cfg_ref"]["revision"] = str(revision)

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        observations = self._observations()
        if method in observations:
            expected = {
                "tab.new": {"adapter_name": "lookback"},
                "soc.info": {"include_cfg": True},
                "device.snapshot": {"name": "bias"},
            }
            if method in expected:
                assert params == expected[method]
            return deepcopy(observations[method])
        if method in ("tab.get_cfg", "tab.edit_cfg"):
            if method == "tab.edit_cfg":
                self._edit(params)
            return deepcopy(self.publication)
        if method == "tab.run_start":
            assert params == {"tab_id": "t", "expected": self.publication["cfg_ref"]}
            assert not self.ran
            self.ran = True
            return {"operation_id": 71}
        if method == "tab.save_data":
            assert params == {"tab_id": "t", "run_operation_id": 71}
            return {"operation_id": 82, "data_path": "/actual/raw.h5"}
        if method == "operation.await":
            assert 0 < params["timeout"] <= 0.25
            assert params["operation_id"] in (71, 82, 93)
            if params["operation_id"] == 82:
                self.raw_saved = True
            return {"reason": "completed", "status": "finished"}
        return self._analysis(method, params)

    def _analysis(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.analyze":
            assert self.raw_saved
            assert params == {"tab_id": "t", "updates": {}, "run_operation_id": 71}
            return {
                "operation_id": 93,
                "interactive": False,
                "params": {"threshold": 0.5},
                "invalidated_on_success": [],
            }
        if method == "tab.get_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 93}
            return {
                "summary": {"offset": 0.24},
                "params": {"threshold": 0.5},
                "operation_state": {"analysis_state": {"figure_names": ["trace"]}},
            }
        if method == "tab.save_image":
            assert params["operation_id"] == 93
            assert params["figure_name"] == "trace"
            return {"image_path": "/actual/trace.png"}
        if method == "tab.get_figure":
            assert params["operation_id"] == 93
            return {"png_b64": base64.b64encode(_PNG).decode()}
        if method == "tab.writeback_preview":
            assert params == {
                "tab_id": "t",
                "subtab_id": "analysis",
                "operation_id": 93,
            }
            return {
                "has_draft": True,
                "items": [{"id": "md-1", "proposed": 0.24}],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)


def test_lookback_saves_original_run_then_analysis_and_delivers_complete_reply(
    tmp_path,
):
    gui = LookbackGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call(
            "lookback",
            {
                "frequency_mhz": 6020.0,
                "readout_length_us": 4.0,
                "trigger_offset_us": 0.2,
                "rounds": 7,
            },
        )
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert not reply.is_error
        assert data["tab"] == "t"
        assert data["run_op"] != 71
        assert data["run_outcome"]["status"] == "finished"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["writeback"]["items"] == [{"id": "md-1", "proposed": 0.24}]
        assert reply.images[0].data == _PNG
        assert data["elapsed_s"] >= 0
        actual = data["actual"]
        assert actual["cfg_ref"] == gui.publication["cfg_ref"]
        assert actual["fields"]["modules.readout.pulse_cfg.freq"]["value"] == 6020.0
        assert actual["fields"]["modules.readout.ro_cfg.ro_freq"]["value"] == 6020.0
        assert actual["fields"]["modules.readout.ro_cfg.ro_length"]["value"] == 4.0
        assert actual["fields"]["modules.readout.ro_cfg.trig_offset"]["value"] == 0.2
        assert actual["fields"]["rounds"]["value"] == 7
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.index("device.snapshot") < methods.index("tab.run_start")
        assert methods.index("soc.info") < methods.index("tab.run_start")
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
        assert methods.index("tab.save_image") < methods.index("tab.writeback_preview")
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
        ]
        by_path = {tuple(edit["path"]): edit["value"] for edit in edits}
        assert by_path["modules", "reset"] is None
        assert by_path["modules", "init_pulse"] is None
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse_tab_id", [None, "kept"])
def test_lookback_missing_frequency_does_not_run_a_blind_default(
    tmp_path, reuse_tab_id
):
    tab = reuse_tab_id or "new"
    publication = {
        "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
        "status": "Valid",
        "source_basis": [],
        "diagnostics": [],
        "tree": {
            "kind": "section",
            "valid": True,
            "children": {
                "modules": {
                    "kind": "section",
                    "valid": True,
                    "children": {
                        "readout": {
                            "kind": "reference",
                            "ref": None,
                            "valid": True,
                            "children": {},
                        },
                    },
                },
            },
        },
    }

    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.snapshot":
            return {"label": "sample", "md": {}, "ml": {"modules": {}, "waveforms": {}}}
        if method == "tab.new":
            return {"tab_id": tab}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": tab,
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                    }
                ]
            }
        if method in ("tab.get_cfg", "tab.reset_cfg", "tab.edit_cfg"):
            return deepcopy(publication)
        raise AssertionError(f"Missing frequency must not start work: {method}")

    client = make_client(tmp_path, respond)
    try:
        arguments = {} if reuse_tab_id is None else {"reuse_tab_id": reuse_tab_id}
        reply = client.call("lookback", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert reply.data["tab"] == tab
        assert [item["parameter"] for item in reply.data["missing"]] == [
            "frequency_mhz"
        ]
        assert reply.is_error is False
        methods = [method for method, _ in client.transport.sent]
        assert "tab.run_start" not in methods
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
    finally:
        client.context.session.close()
