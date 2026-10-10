"""Recipe target admission and fixed role-source reads through public tools."""

from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession

from ._recipe_execution_support import definition
from ._recipe_support import LookbackGui, scalar
from ._support import make_client


def targeted(
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
) -> RecipeGenerator:
    """Prepare current/temporary readout, then yield one fake acquisition."""
    tab = session.open_tab(
        "lookback",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.set_frequency(
        "modules.readout", calibration="resonator", required="readout_ref"
    )
    yield tab.run()


TARGETED = definition(
    targeted,
    {
        "type": "object",
        "properties": {
            "target": {"type": ["string", "null"]},
            "reuse_tab_id": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
)


class TargetGui(LookbackGui):
    """Record current/temporary source queries through the public recipe wire."""

    def __init__(self) -> None:
        super().__init__()
        self.md = {"$resonator.freq": 6200.0}
        self.created: list[dict[str, object]] = []
        self.reads: list[dict[str, object]] = []
        self.modules: dict[str, str] = {}
        self.overrides: list[dict[str, str] | None] = []
        self.queried_paths: list[str] = []
        self.queried_modules: list[str] = []

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "context.snapshot":
            raise AssertionError("Recipe sources must come from the tab capture")
        if method == "tab.new":
            self.created.append(dict(params))
            return {"tab_id": "t"}
        if method == "tab.get_cfg" and params.get("aliases") is not None:
            assert params["expected"] == self.publication["cfg_ref"]
            projected = deepcopy(self.publication)
            projected["status"] = "Valid"
            readout = projected["tree"]["children"]["modules"]["children"]["readout"]
            for node in readout["children"].values():
                for child in node.get("children", {}).values():
                    if child.get("input", {}).get("raw") == "$resonator.freq":
                        child["input"]["resolved"] = 8200.0
            projected["source_basis"] = [{"source_id": "target", "revision": "1"}]
            return projected
        if method == "tab.run_start":
            aliases = params.get("aliases")
            assert aliases is None or isinstance(aliases, dict)
            self.overrides.append(aliases)
            return super().__call__(
                method,
                {key: value for key, value in params.items() if key != "aliases"},
            )
        if method == "tab.parameter_read":
            assert params["tab_id"] == "t"
            self.reads.append(dict(params))
            paths = params["paths"]
            references = params["module_refs"]
            assert isinstance(paths, list) and isinstance(references, list)
            self.queried_paths.extend(path for path in paths if isinstance(path, str))
            self.queried_modules.extend(
                ref for ref in references if isinstance(ref, str)
            )
            return {
                "values": {
                    path: self.md[path]
                    for path in paths
                    if isinstance(path, str) and path in self.md
                },
                "modules": {
                    reference: self.modules.get(reference)
                    for reference in references
                    if isinstance(reference, str)
                },
            }
        return super().__call__(method, params)


class NamedAliasesGui(TargetGui):
    """Record the public aliases wire contract without changing recipe logic."""

    def __init__(self) -> None:
        super().__init__()
        self.alias_requests: list[tuple[str, dict[str, object]]] = []
        self.modules = {"$resonator.module.readout": "R2.readout.main"}
        self.publication["tree"]["children"]["modules"]["children"]["readout"][
            "ref"
        ] = "$resonator.module.readout"

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if params.get("aliases"):
            self.alias_requests.append((method, dict(params)))
            plain = {key: value for key, value in params.items() if key != "aliases"}
            reply = super().__call__(method, plain)
            if method == "tab.get_cfg":
                reply["source_basis"] = [{"source_id": "aliases", "revision": "1"}]
            return reply
        return super().__call__(method, params)


@pytest.mark.parametrize("reuse", [None, "t"], ids=["new", "reuse"])
def test_open_tab_preserves_named_aliases_for_new_and_reused_recipe_runs(
    tmp_path: Path, reuse: str | None
) -> None:
    gui = NamedAliasesGui()
    aliases = {"custom": "Q1", "resonator": "R2"}

    def recipe(session: RecipeSession) -> RecipeGenerator:
        tab = session.open_tab("lookback", aliases=aliases, reuse=reuse)
        aliases["resonator"] = "missing"
        tab.use_library("modules.readout", None, required="readout_ref")
        tab.set_frequency(
            "modules.readout", calibration="resonator", required="readout_ref"
        )
        yield tab.run()

    registered = definition(
        recipe, {"type": "object", "properties": {}, "additionalProperties": False}
    )
    client = make_client(tmp_path, gui, recipes=(registered,))
    with closing(client.context.session):
        reply = client.call("sample", {})
        assert reply.data["status"] == "finished"
        assert gui.ran
        assert {method for method, _ in gui.alias_requests} == {
            "tab.get_cfg",
            "tab.parameter_read",
            "tab.run_start",
        }
        assert all(
            params["aliases"] == {"custom": "Q1", "resonator": "R2"}
            for _, params in gui.alias_requests
        )
        run = next(
            params for method, params in gui.alias_requests if method == "tab.run_start"
        )
        preview = [
            params for method, params in gui.alias_requests if method == "tab.get_cfg"
        ][-1]
        assert run["expected"] == preview["expected"] == gui.publication["cfg_ref"]
        assert all(
            "aliases" not in params
            for method, params in client.transport.sent
            if method in ("tab.new", "tab.reset_cfg")
        )
        assert gui.created == (
            [] if reuse is not None else [{"adapter_name": "lookback"}]
        )
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["actual"]["source_basis"] == [
            {"source_id": "aliases", "revision": "1"}
        ]


def frequency_sweep(
    session: RecipeSession,
    *,
    calibration: Literal["resonator", "qubit"],
) -> RecipeGenerator:
    """Retain the captured GUI linewidth expression for a frequency sweep."""
    tab = session.open_tab("lookback")
    tab.set_frequency_sweep("sweep", calibration=calibration)
    yield tab.run()


def flux_sweep(
    session: RecipeSession, *, start: float | None = None, stop: float | None = None
) -> RecipeGenerator:
    """Retain or explicitly replace captured flux endpoints."""
    tab = session.open_tab("lookback")
    tab.set_flux_sweep("sweep", start=start, stop=stop)
    yield tab.run()


def required_library(
    session: RecipeSession, *, reference: str | None = None
) -> RecipeGenerator:
    """Require a readout from the captured role/slot library lookup."""
    tab = session.open_tab("lookback")
    tab.use_library("modules.readout", reference, required="readout_ref")
    yield tab.run()


class SweepGui(TargetGui):
    """Publish known GUI-resolved sweep inputs, without mocking recipe decisions."""

    def __init__(
        self, start_expr: str, stop_expr: str, start: float, stop: float
    ) -> None:
        super().__init__()
        inputs = {}
        for key, expression, value in (
            ("start", start_expr, start),
            ("stop", stop_expr, stop),
        ):
            state = scalar(value)["input"]
            state.update(mode="expression", raw=expression)
            inputs[key] = state
        inputs["expts"] = scalar(51)["input"]
        self.publication["tree"]["children"]["sweep"] = {
            "kind": "sweep",
            "valid": True,
            "inputs": inputs,
        }
        self.resolved: dict[str, float] = {}

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.edit_cfg":
            edits = params["edits"]
            assert isinstance(edits, list)
            if len(edits) == 1 and edits[0]["path"] == ["sweep"]:
                assert params["expected"] == self.publication["cfg_ref"]
                inputs = self.publication["tree"]["children"]["sweep"]["inputs"]
                for key, value in edits[0]["value"].items():
                    if isinstance(value, dict):
                        expression = value["__expr"]
                        state = scalar(self.resolved[expression])["input"]
                        state.update(mode="expression", raw=expression)
                    else:
                        state = scalar(value)["input"]
                    inputs[key] = state
                ref = self.publication["cfg_ref"]
                ref["revision"] = str(int(ref["revision"]) + 1)
                return self.publication
        return super().__call__(method, params)


@pytest.mark.parametrize("calibration", ["resonator", "qubit"])
def test_frequency_sweep_preserves_dotted_linewidth_expression(
    tmp_path: Path, calibration: Literal["resonator", "qubit"]
) -> None:
    center = f"${calibration}.freq"
    width = f"${calibration}.kappa"
    gui = SweepGui(f"{center} - 3 * {width}", f"{center} + 3 * {width}", 970.0, 1030.0)
    gui.md = {center: 1000.0, width: 10.0}
    span = f"(0 + 3 * {width}) - (0 - 3 * {width})"
    gui.resolved = {
        f"({center}) - ({span}) / 2": 970.0,
        f"({center}) + ({span}) / 2": 1030.0,
    }
    recipe = definition(
        frequency_sweep,
        {
            "type": "object",
            "properties": {"calibration": {"type": "string"}},
            "required": ["calibration"],
            "additionalProperties": False,
        },
    )
    client = make_client(tmp_path, gui, recipes=(recipe,))
    with closing(client.context.session):
        reply = client.call("sample", {"calibration": calibration})
        assert reply.data["status"] == "finished"
        assert gui.ran
        inputs = gui.publication["tree"]["children"]["sweep"]["inputs"]
        assert inputs["start"]["raw"] == f"({center}) - ({span}) / 2"
        assert inputs["stop"]["raw"] == f"({center}) + ({span}) / 2"
        assert inputs["expts"]["resolved"] == 51
        assert {center, width} <= set(gui.queried_paths)


@pytest.mark.parametrize(
    "start,stop,status",
    [(None, None, "finished"), (0.1, None, "failed"), (0.1, 0.1, "failed")],
)
def test_flux_sweep_uses_role_expressions_and_rejects_invalid_explicit_endpoints(
    tmp_path: Path, start: float | None, stop: float | None, status: str
) -> None:
    low = "$qubit.flux_half - ($qubit.flux_int - $qubit.flux_half) / 2"
    high = "$qubit.flux_half + ($qubit.flux_int - $qubit.flux_half) / 2"
    gui = SweepGui(low, high, 0.1, 0.5)
    gui.md = {"$qubit.flux_half": 0.3, "$qubit.flux_int": 0.7}
    recipe = definition(
        flux_sweep,
        {
            "type": "object",
            "properties": {
                "start": {"type": ["number", "null"]},
                "stop": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
    )
    client = make_client(tmp_path, gui, recipes=(recipe,))
    with closing(client.context.session):
        reply = client.call("sample", {"start": start, "stop": stop})
        assert reply.data["status"] == status
        assert gui.ran is (status == "finished")
        inputs = gui.publication["tree"]["children"]["sweep"]["inputs"]
        assert inputs["start"]["raw"] == low
        assert inputs["stop"]["raw"] == high
        if status == "finished":
            assert {"$qubit.flux_half", "$qubit.flux_int"} <= set(gui.queried_paths)


@pytest.mark.parametrize(
    "reference", [None, "$resonator.module.readout", "<Custom:readout>"]
)
def test_required_library_uses_captured_role_slot_membership(
    tmp_path: Path, reference: str | None
) -> None:
    gui = TargetGui()
    gui.modules = {"$resonator.module.readout": "R2.readouts.main"}
    gui.publication["tree"]["children"]["modules"]["children"]["readout"]["ref"] = (
        "$resonator.module.readout"
    )
    recipe = definition(
        required_library,
        {
            "type": "object",
            "properties": {"reference": {"type": ["string", "null"]}},
            "additionalProperties": False,
        },
    )
    client = make_client(tmp_path, gui, recipes=(recipe,))
    with closing(client.context.session):
        reply = client.call("sample", {"reference": reference})
        valid = reference != "<Custom:readout>"
        assert reply.data["status"] == ("finished" if valid else "failed")
        assert gui.ran is valid
        expected = "$resonator.module.readout" if valid else "<Custom:readout>"
        assert expected in gui.queried_modules


@pytest.mark.parametrize("target", [None, "Q2"])
def test_recipe_run_uses_temporary_target_without_tab_birth_binding(
    tmp_path: Path, target: str | None
) -> None:
    gui = TargetGui()
    if target is not None:
        gui.publication["status"] = "Invalid"
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    with closing(client.context.session):
        reply = client.call("sample", {} if target is None else {"target": target})
        assert reply.data["status"] == "finished"
        assert gui.created == [{"adapter_name": "lookback"}]
        assert gui.ran
        assert gui.overrides == [None if target is None else {"qubit": target}]
        assert gui.reads
        assert all(
            read.get("aliases") == (None if target is None else {"qubit": target})
            for read in gui.reads
        )
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["actual"]["fields"]["modules.readout.pulse_cfg.freq"]["value"] == (
            6200.0 if target is None else 8200.0
        )
        assert full["actual"]["source_basis"] == (
            [] if target is None else [{"source_id": "target", "revision": "1"}]
        )
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
        ]
        expressions = [
            edit["value"]
            for params in edits
            for edit in params["edits"]
            if edit["path"][-1] in ("freq", "ro_freq")
        ]
        assert expressions == [
            {"__expr": "$resonator.freq"},
            {"__expr": "$resonator.freq"},
        ]
        assert (
            gui.publication["tree"]["children"]["modules"]["children"]["readout"][
                "children"
            ]["pulse_cfg"]["children"]["freq"]["input"]["resolved"]
            == 6200.0
        )


def test_recipe_stale_run_ref_is_not_refreshed_or_retried(tmp_path: Path) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    client.transport.replies["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_revision",
            "message": "Source publication changed after preview",
        },
    }
    with closing(client.context.session):
        reply = client.call("sample", {"target": "Q2"})
        assert reply.data["status"] == "failed"
        assert not gui.ran
        starts = [
            params
            for method, params in client.transport.sent
            if method == "tab.run_start"
        ]
        assert starts == [
            {
                "tab_id": "t",
                "expected": gui.publication["cfg_ref"],
                "aliases": {"qubit": "Q2"},
            }
        ]
        previews = [
            params
            for method, params in client.transport.sent
            if method == "tab.get_cfg" and params.get("aliases") is not None
        ]
        assert previews[-1]["expected"] == starts[0]["expected"]
        start_index = next(
            index
            for index, (method, _) in enumerate(client.transport.sent)
            if method == "tab.run_start"
        )
        assert not any(
            method == "tab.get_cfg"
            for method, _ in client.transport.sent[start_index + 1 :]
        )


def test_recipe_target_can_be_used_with_explicit_reuse(tmp_path: Path) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    with closing(client.context.session):
        reply = client.call("sample", {"target": "Q2", "reuse_tab_id": "t"})
        assert reply.data["status"] == "finished"
        assert gui.created == []
        assert gui.overrides == [{"qubit": "Q2"}]
        assert any(method == "tab.reset_cfg" for method, _ in client.transport.sent)


def test_role_resolution_rejection_is_not_replaced_by_another_target(
    tmp_path: Path,
) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    client.transport.replies["tab.parameter_read"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "role_resolution_failed",
            "message": "No driver for target",
            "data": {
                "role": "driver",
                "focus": "missing",
                "required_kind": "fake/drive/*",
                "detail": "Unknown component",
            },
        },
    }
    with closing(client.context.session):
        reply = client.call("sample", {"target": "missing"})
        assert reply.data["status"] == "failed"
        assert "role_resolution_failed" in str(reply.data)
        assert not gui.ran
        creations = [
            params for method, params in client.transport.sent if method == "tab.new"
        ]
        assert creations == [{"adapter_name": "lookback"}]
        reads = [
            params
            for method, params in client.transport.sent
            if method == "tab.parameter_read"
        ]
        assert reads[0]["aliases"] == {"qubit": "missing"}
