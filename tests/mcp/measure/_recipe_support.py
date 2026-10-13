"""Recording GUI collaborator for public recipe contract tests."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from zcu_tools.gui.cfg.reference_key import is_custom_reference_key
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.expression import evaluate_scalar_expr
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import (
    ComponentSchema,
    PointDocument,
    PointGeneral,
)

from ._support import make_client


class FormulaComponent(ComponentSchema):
    freq: float | None = None
    kappa: float | None = None
    flux_half: float | None = None
    flux_int: float | None = None


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


def scalar(value: object) -> dict[str, Any]:
    return {
        "kind": "scalar",
        "valid": True,
        "readonly": False,
        "input": {
            "mode": "direct",
            "raw": value,
            "resolved": value,
            "error": None,
            "validation_error": None,
        },
    }


def section(**children: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "section", "valid": True, "readonly": False, "children": children}


def publish_reference_locks(node: object, *, readonly: bool = False) -> None:
    """Update modes and locks in a recording cfg subtree, in place.

    ``node`` is an observation dict with a boolean ``readonly`` and optional
    ``children`` dict of observations. Reference nodes carry a string or None
    ``ref``: library references use ref mode, custom references use inline mode,
    and None disables the reference. ``readonly`` carries an ancestor lock.
    Descendants of library references, literals and existing static locks stay
    readonly. These recording trees do not simulate conversion back to inline.
    Raise TypeError for malformed nodes, references or children.
    """
    if not isinstance(node, dict):
        raise TypeError("Recording cfg nodes must be mappings")
    locked = node.get("readonly")
    if not isinstance(locked, bool):
        raise TypeError("Recording cfg nodes must declare readonly")
    child_readonly = readonly or locked or node.get("kind") == "literal"
    node["readonly"] = child_readonly
    if node.get("kind") == "reference":
        reference = node.get("ref")
        if reference is not None and not isinstance(reference, str):
            raise TypeError("Recording references must be strings or None")
        mode = (
            ("inline" if is_custom_reference_key(reference) else "ref")
            if reference is not None
            else None
        )
        node["mode"] = mode
        child_readonly = child_readonly or mode == "ref"
    children = node.get("children", {})
    if not isinstance(children, dict):
        raise TypeError("Recording cfg children must be mappings")
    for child in children.values():
        publish_reference_locks(child, readonly=child_readonly)


class LookbackGui:
    """GUI collaborator with distinct Run, save and analysis wire identities."""

    def __init__(self):
        self.publication: dict[str, Any] = {
            "status": "Valid",
            "source_basis": [],
            "diagnostics": [],
            "tree": section(
                rounds=scalar(3),
                modules=section(
                    reset={
                        "kind": "reference",
                        "valid": True,
                        "readonly": False,
                        "ref": None,
                    },
                    init_pulse={
                        "kind": "reference",
                        "valid": True,
                        "readonly": False,
                        "ref": None,
                    },
                    readout={
                        "kind": "reference",
                        "valid": True,
                        "readonly": False,
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
        self.library: dict[str, Any] = {}

    def evaluate_expression(self, expression: str) -> int | float | complex | str:
        """Evaluate GUI formulas with native models and a fixed public context.

        The recording parameter-read wire below deliberately retains malformed
        values for caller rejection coverage; it does not claim native admission.
        Formula evaluation, unlike that wire, uses the real scalar expression owner.
        Bare aliases return resolved names; arithmetic returns numeric values.
        Direct native model construction requires no process-global registry.
        """
        resonator = FormulaComponent.model_validate(
            {
                "kind": "resonator",
                **{
                    name: self.md[f"$resonator.{name}"]
                    for name in ("freq", "kappa")
                    if f"$resonator.{name}" in self.md
                },
            }
        )
        qubit = FormulaComponent.model_validate(
            {
                "kind": "qubit/transmon",
                **{
                    name: self.md[f"$qubit.{name}"]
                    for name in ("freq", "kappa", "flux_half", "flux_int")
                    if f"$qubit.{name}" in self.md
                },
            }
        )
        parameters = ParameterContext(
            point=PointDocument.model_construct(
                format="zcu.parameter-container",
                format_version="2.0",
                general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
                components={"R1": resonator, "Q1": qubit},
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="1.0",
                )
            ),
            target=SavedPoint(entry_id="recipe-entry", point="working"),
            roles=None,
        ).bind_aliases((AliasRow("resonator", "R1"), AliasRow("qubit", "Q1")), {})
        return evaluate_scalar_expr(expression, parameters)

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
        for edit in params["edits"]:
            node = self.publication["tree"]
            for part in edit["path"]:
                node = node["children"][part]
            if node["kind"] == "reference":
                node["ref"] = edit["value"]["__ref"]
                publish_reference_locks(node)
            else:
                value = edit["value"]
                if isinstance(value, dict) and "__expr" in value:
                    node.update(scalar(self.md.get(value["__expr"])))
                    node["input"].update(mode="expression", raw=value["__expr"])
                else:
                    node.update(scalar(value))

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        observations = self._observations()
        if method == "tab.parameter_read":
            observations[method] = {
                "values": {
                    path: deepcopy(self.md.get(path)) for path in params["paths"]
                },
                "modules": {
                    reference: reference if reference in self.library else None
                    for reference in params["module_refs"]
                },
            }
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
            assert params == {
                "tab_id": "t",
            }
            assert not self.ran
            self.ran = True
            return {"operation_id": 71}
        if method == "tab.save_data":
            assert params == {"tab_id": "t", "run_operation_id": 71}
            self.raw_saved = True
            return {
                "run_id": "run-a",
                "artifacts": {
                    "raw": {
                        "data": {
                            "status": "Saved",
                            "members": {
                                "data": [
                                    {
                                        "format": "data_h5",
                                        "path": "/runs/run-a/data.h5",
                                        "status": "saved",
                                        "error": None,
                                    },
                                    {
                                        "format": "labber",
                                        "path": "/labber/raw.hdf5",
                                        "status": "saved",
                                        "error": None,
                                    },
                                ]
                            },
                        }
                    }
                },
            }
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
        if method == "tab.save_artifacts":
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": "analysis", "name": "results"}],
                "operation_id": 93,
            }
            return {
                "run_id": "run-a",
                "artifacts": {
                    "analysis": {
                        "results": {
                            "status": "Saved",
                            "members": {
                                "analysis:primary": [
                                    {
                                        "format": "json",
                                        "path": "/runs/run-a/analysis/results.json",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                                "figure:trace": [
                                    {
                                        "format": "png",
                                        "path": "/runs/run-a/analysis/trace.png",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
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
                "items": [
                    {
                        "id": "param-1",
                        "kind": "parameter",
                        "target_name": "trigger_offset",
                        "selected": False,
                        "proposed": 0.24,
                        "current": 0.1,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)
