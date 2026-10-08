"""Recipe target admission and fixed role-source reads through public tools."""

from contextlib import closing
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession

from ._recipe_execution_support import definition
from ._recipe_support import LookbackGui
from ._support import make_client


def targeted(
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
) -> RecipeGenerator:
    """Prepare readout from the tab capture, then yield one fake acquisition."""
    tab = session.open_tab("lookback", target=target, reuse=reuse_tab_id)
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
    """Expose role calibration only through the new fixed-tab source wire."""

    def __init__(self) -> None:
        super().__init__()
        self.md = {"resonator.freq": 6200.0}
        self.created: list[dict[str, object]] = []
        self.reads: list[dict[str, object]] = []

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "context.snapshot":
            raise AssertionError("Recipe sources must come from the tab capture")
        if method == "tab.new":
            self.created.append(dict(params))
            return {"tab_id": "t"}
        if method == "tab.parameter_read":
            assert params["tab_id"] == "t"
            self.reads.append(dict(params))
            paths = params["paths"]
            references = params["module_refs"]
            assert isinstance(paths, list) and isinstance(references, list)
            return {
                "values": {
                    path: self.md[path]
                    for path in paths
                    if isinstance(path, str) and path in self.md
                },
                "modules": {
                    reference: None
                    for reference in references
                    if isinstance(reference, str)
                },
            }
        return super().__call__(method, params)


@pytest.mark.parametrize("target", [None, "Q2"])
def test_recipe_opens_target_and_uses_role_calibration_without_live_context(
    tmp_path: Path, target: str | None
) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    with closing(client.context.session):
        reply = client.call("sample", {} if target is None else {"target": target})
        assert reply.data["status"] == "finished"
        assert gui.created == [
            {"adapter_name": "lookback"}
            if target is None
            else {"adapter_name": "lookback", "target": "Q2"}
        ]
        assert gui.ran
        assert gui.reads
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
            {"__expr": "resonator.freq"},
            {"__expr": "resonator.freq"},
        ]
        assert (
            gui.publication["tree"]["children"]["modules"]["children"]["readout"][
                "children"
            ]["pulse_cfg"]["children"]["freq"]["input"]["resolved"]
            == 6200.0
        )


def test_recipe_target_and_reuse_are_rejected_before_gui_requests(
    tmp_path: Path,
) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    with closing(client.context.session):
        reply = client.call("sample", {"target": "Q2", "reuse_tab_id": "t"})
        assert reply.data["status"] == "failed"
        assert "重用 tab 時不要傳 target" in str(reply.data)
        assert gui.created == []
        assert not gui.ran
        assert not any(
            method.startswith("tab.") or method == "context.snapshot"
            for method, _ in client.transport.sent
        )


def test_role_resolution_rejection_is_not_replaced_by_another_target(
    tmp_path: Path,
) -> None:
    gui = TargetGui()
    client = make_client(tmp_path, gui, recipes=(TARGETED,))
    client.transport.replies["tab.new"] = {
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
        assert creations == [{"adapter_name": "lookback", "target": "missing"}]
