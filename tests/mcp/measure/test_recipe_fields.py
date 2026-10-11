"""Generic batch and expression author edits retain the GUI cfg protocol."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure import recipe
from zcu_tools.mcp.measure.recipe import RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, recipe_client, scalar


class FieldGui(LookbackGui):
    """Recording scalar/sweep edit owner; formulas use supplied logical values."""

    def __init__(self) -> None:
        super().__init__()
        self.md.update({"$custom.low": 10.0, "$custom.high": 20.0})
        self.publication["tree"]["children"]["range"] = {
            "kind": "sweep",
            "valid": True,
            "inputs": {
                "start": scalar(1.0)["input"],
                "stop": scalar(2.0)["input"],
                "expts": scalar(11)["input"],
            },
        }

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.edit_cfg":
            edits = params["edits"]
            if (
                isinstance(edits, list)
                and len(edits) == 1
                and edits[0]["path"] == ["range"]
            ):
                assert params["expected"] == self.publication["cfg_ref"]
                for key, value in edits[0]["value"].items():
                    state = scalar(value)["input"]
                    if isinstance(value, dict):
                        state.update(
                            mode="expression",
                            raw=value["__expr"],
                            resolved=self.md[value["__expr"]],
                        )
                    self.publication["tree"]["children"]["range"]["inputs"][key] = state
                self.publication["cfg_ref"]["revision"] = str(
                    int(self.publication["cfg_ref"]["revision"]) + 1
                )
                return self.publication
        return super().__call__(method, params)


def test_set_fields_writes_one_batch_revision_and_keeps_none_default(
    tmp_path: Path,
) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        tab.set_fields(
            [
                recipe.RecipeFieldEdit("rounds", None, "ignored"),
                recipe.RecipeFieldEdit(
                    "modules.readout.pulse_cfg.freq",
                    recipe.RecipeExpression(__expr="$custom.low"),
                    "caller_low",
                ),
                recipe.RecipeFieldEdit(
                    "modules.readout.ro_cfg.ro_freq", 22.0, "caller_high"
                ),
            ]
        )
        assert client.transport.sent[before:] == [
            (
                "tab.edit_cfg",
                {
                    "tab_id": "t",
                    "expected": {"cfg_id": "cfg", "revision": "2"},
                    "edits": [
                        {
                            "path": ["modules", "readout", "pulse_cfg", "freq"],
                            "value": {"__expr": "$custom.low"},
                        },
                        {
                            "path": ["modules", "readout", "ro_cfg", "ro_freq"],
                            "value": 22.0,
                        },
                    ],
                },
            )
        ]
        assert gui.publication["cfg_ref"]["revision"] == "3"
        assert gui.publication["tree"]["children"]["rounds"]["input"]["resolved"] == 3
        pulse = gui.publication["tree"]["children"]["modules"]["children"]["readout"][
            "children"
        ]["pulse_cfg"]["children"]["freq"]["input"]
        assert pulse["mode"] == "expression"
        assert pulse["raw"] == "$custom.low"
        assert pulse["resolved"] == 10.0
        snapshot = tab.run().snapshot()
        assert snapshot is not None
        assert (
            snapshot.actual["fields"]["modules.readout.pulse_cfg.freq"].get("source")
            == "caller_low"
        )
        assert snapshot.actual["fields"]["rounds"].get("source") == "gui_default"


@pytest.mark.parametrize(
    "bad_field,bad_source",
    [("missing", "explicit"), ("modules.readout", "explicit"), ("rounds", " ")],
)
def test_batch_preflight_rejects_all_before_first_edit(
    tmp_path: Path, bad_field: str, bad_source: str
) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        with pytest.raises(ValueError, match="cfg field|source"):
            tab.set_fields(
                [
                    recipe.RecipeFieldEdit("rounds", 7, "valid"),
                    recipe.RecipeFieldEdit(bad_field, 9, bad_source),
                ]
            )
        assert len(client.transport.sent) == before
        assert gui.publication["cfg_ref"]["revision"] == "2"


def test_empty_batch_and_default_only_batch_do_not_edit(tmp_path: Path) -> None:
    with recipe_client(tmp_path, FieldGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        tab.set_fields([])
        tab.set_fields([recipe.RecipeFieldEdit("rounds", None, "explicit")])
        assert len(client.transport.sent) == before


def test_expression_set_and_sweep_are_forwarded_without_client_evaluation(
    tmp_path: Path,
) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set("rounds", recipe.RecipeExpression(__expr="$custom.low"), source="input")
        tab.set_sweep(
            "range",
            start=recipe.RecipeExpression(__expr="$custom.low"),
            stop=recipe.RecipeExpression(__expr="$custom.high"),
        )
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
        ]
        assert [params["expected"]["revision"] for params in edits] == ["2", "3"]
        assert edits[1]["edits"] == [
            {
                "path": ["range"],
                "value": {
                    "start": {"__expr": "$custom.low"},
                    "stop": {"__expr": "$custom.high"},
                },
            }
        ]
        state = gui.publication["tree"]["children"]["range"]["inputs"]
        assert state["start"]["resolved"] == 10.0
        assert state["stop"]["resolved"] == 20.0
        assert state["expts"]["resolved"] == 11


def test_batch_stale_failure_does_not_refresh_or_retry(tmp_path: Path) -> None:
    with recipe_client(tmp_path, FieldGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        client.transport.replies["tab.edit_cfg"] = {
            "ok": False,
            "error": {
                "code": "failed_precondition",
                "message": "stale cfg",
                "reason": "stale_cfg",
            },
        }
        before = len(client.transport.sent)
        with pytest.raises(GuiRpcError) as failure:
            tab.set_fields([recipe.RecipeFieldEdit("rounds", 7, "explicit")])
        assert failure.value.reason == "stale_cfg"
        assert [method for method, _ in client.transport.sent[before:]] == [
            "tab.edit_cfg"
        ]


def test_batch_invalid_reply_retains_edit_without_rollback(tmp_path: Path) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        gui.publication["tree"]["children"]["rounds"]["valid"] = False
        gui.publication["tree"]["children"]["rounds"]["input"]["validation_error"] = (
            "outside range"
        )
        client.transport.replies["tab.edit_cfg"] = {
            "ok": True,
            "result": gui.publication,
        }
        before = len(client.transport.sent)
        with pytest.raises(GuiRpcError) as failure:
            tab.set_fields([recipe.RecipeFieldEdit("rounds", 7, "explicit")])
        assert failure.value.reason == "invalid_cfg"
        assert [method for method, _ in client.transport.sent[before:]] == [
            "tab.edit_cfg"
        ]
