"""Recipe target admission and fixed role-source reads through public tools."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession

from ._recipe_execution_support import definition
from ._recipe_parameter_support import TargetGui
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
    tab.read_value("$resonator.freq")
    for path in (
        "modules.readout.module.pulse_cfg.module.freq",
        "modules.readout.module.ro_cfg.ro_freq",
    ):
        tab.set(path, {"expr": "$resonator.freq"}, source="$resonator.freq")
    yield tab.run()


_TARGETED_DEFINITION = definition(
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


TARGETED = replace(
    _TARGETED_DEFINITION,
    summary_parameters=(
        SummaryParameter(
            "frequency_mhz", "modules.readout.module.pulse_cfg.module.freq", "MHz"
        ),
    ),
)


class NamedAliasesGui(TargetGui):
    """Record the public aliases wire contract without changing recipe logic."""

    def __init__(self) -> None:
        super().__init__()
        self.alias_requests: list[tuple[str, dict[str, object]]] = []
        self.modules = {"$resonator.module.readout": "R2.readout.main"}

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if params.get("aliases"):
            self.alias_requests.append((method, dict(params)))
            return super().__call__(method, params)
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
        tab.read_value("$resonator.freq")
        yield tab.run()

    registered = definition(
        recipe, {"type": "object", "properties": {}, "additionalProperties": False}
    )
    client = make_client(tmp_path, gui, recipes=(registered,))
    with closing(client.context.session):
        reply = client.call("sample", {})
        assert reply.data["status"] == "finished"
        assert gui.ran
        assert gui.alias_requests == [
            (
                "tab.reset_cfg",
                {"tab_id": "t", "aliases": {"custom": "Q1", "resonator": "R2"}},
            )
        ]
        assert gui.created == (
            [] if reuse is not None else [{"adapter_name": "lookback"}]
        )


@pytest.mark.parametrize("reuse", [None, "t"], ids=["new", "reuse"])
def test_recipe_aliases_are_captured_only_by_reset(
    tmp_path: Path, reuse: str | None
) -> None:
    gui = NamedAliasesGui()
    client = make_client(tmp_path, gui)
    with closing(client.context.session):
        aliases = {"resonator": "R2"}
        tab = RecipeSession(client.context).open_tab(
            "lookback", aliases=aliases, reuse=reuse
        )
        aliases["resonator"] = "missing"
        tab.read_value("$resonator.freq")
        tab.run()
        assert gui.alias_requests == [
            ("tab.reset_cfg", {"tab_id": "t", "aliases": {"resonator": "R2"}})
        ]


def required_library(
    session: RecipeSession, *, reference: str | None = None
) -> RecipeGenerator:
    """Require a readout from the captured role/slot library lookup."""
    tab = session.open_tab("lookback")
    tab.use_library("modules.readout", "R2.readout.main")
    tab.use_library("modules.readout", reference, required="readout_ref")
    yield tab.run()


@pytest.mark.parametrize("reference", [None, "R2.readout.main", "missing.readout.main"])
def test_required_library_uses_captured_role_slot_membership(
    tmp_path: Path, reference: str | None
) -> None:
    gui = TargetGui()
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
        valid = reference != "missing.readout.main"
        assert reply.data["status"] == ("finished" if valid else "failed")
        assert gui.ran is valid
        assert (
            gui.publication["input"]["modules"]["readout"]["name"] == "R2.readout.main"
        )


@pytest.mark.parametrize("target", [None, "Q2"])
def test_recipe_run_uses_temporary_target_without_tab_birth_binding(
    tmp_path: Path, target: str | None
) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    with closing(client.context.session):
        reply = client.call("sample", {} if target is None else {"target": target})
        assert reply.data["status"] == "finished"
        assert gui.created == [{"adapter_name": "lookback"}]
        assert gui.ran
        assert gui.overrides == [None if target is None else {"qubit": target}]
        assert gui.reads
        assert all(read.get("aliases") is None for read in gui.reads)
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["actual"]["frequency_mhz"]["value"] == (
            6200.0 if target is None else 8200.0
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
            {"expr": "$resonator.freq"},
            {"expr": "$resonator.freq"},
        ]
        assert gui.publication["input"]["modules"]["readout"]["module"]["pulse_cfg"][
            "module"
        ]["freq"] == {"expr": "$resonator.freq"}


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
    client.transport.replies["tab.reset_cfg"] = {
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
        resets = [
            params
            for method, params in client.transport.sent
            if method == "tab.reset_cfg"
        ]
        assert resets[0]["aliases"] == {"qubit": "missing"}
