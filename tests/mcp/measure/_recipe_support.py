"""Recording GUI collaborator for public recipe contract tests."""

import base64
from copy import deepcopy
from typing import Any

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def scalar(value: object) -> dict[str, Any]:
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


def section(**children: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "section", "valid": True, "children": children}


class LookbackGui:
    """GUI collaborator with distinct Run, save and analysis wire identities."""

    def __init__(self):
        self.publication: dict[str, Any] = {
            "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
            "status": "Valid",
            "source_basis": [],
            "diagnostics": [],
            "tree": section(
                rounds=scalar(3),
                modules=section(
                    reset={"kind": "reference", "valid": True, "ref": None},
                    init_pulse={"kind": "reference", "valid": True, "ref": None},
                    readout={
                        "kind": "reference",
                        "valid": True,
                        "ref": None,
                        "error": None,
                        "children": {
                            "pulse_cfg": section(freq=scalar(5000.0)),
                            "ro_cfg": section(
                                ro_freq=scalar(5000.0),
                                ro_length=scalar(2.0),
                                trig_offset=scalar(0.1),
                            ),
                        },
                    },
                ),
            ),
        }
        self.ran = False
        self.raw_saved = False
        self.md: dict[str, Any] = {}

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            "context.snapshot": {
                "label": "sample",
                "md": self.md,
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
                node["ref"] = edit["value"]["__ref"]
            else:
                value = edit["value"]
                if isinstance(value, dict) and "__expr" in value:
                    node.update(scalar(self.md.get(value["__expr"])))
                    node["input"].update(mode="expression", raw=value["__expr"])
                else:
                    node.update(scalar(value))
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
        if method in ("tab.get_cfg", "tab.edit_cfg", "tab.reset_cfg"):
            if method != "tab.get_cfg":
                self._edit(
                    params if method == "tab.edit_cfg" else {**params, "edits": []}
                )
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
                "invalid": [],
                "params": {"threshold": 0.5},
                "operation_state": {"analysis_state": {"figure_names": ["trace"]}},
            }
        if method == "tab.save_image":
            assert params["operation_id"] == 93
            assert params["figure_name"] == "trace"
            return {"image_path": "/actual/trace.png"}
        if method == "tab.get_figure":
            assert params["operation_id"] == 93
            return {"png_b64": base64.b64encode(PNG).decode()}
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
