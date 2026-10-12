"""Generic batch and expression author edits retain the GUI cfg protocol."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.mcp.measure import recipe
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_execution_support import definition
from ._recipe_support import LookbackGui, recipe_client, scalar
from ._support import make_client


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
                for key, value in edits[0]["value"].items():
                    state = scalar(value)["input"]
                    if isinstance(value, dict):
                        state.update(
                            mode="expression",
                            raw=value["__expr"],
                            resolved=self.md[value["__expr"]],
                        )
                    self.publication["tree"]["children"]["range"]["inputs"][key] = state
                return self.publication
        return super().__call__(method, params)


def test_retain_source_records_scalar_and_sweep_without_cfg_edits(
    tmp_path: Path,
) -> None:
    gui = FieldGui()

    def run(session: RecipeSession) -> RecipeGenerator:
        tab = session.open_tab("lookback")
        tab.retain_source("rounds", "caller_rounds")
        tab.retain_source("range", "caller_range")
        yield tab.run()

    sample = replace(
        definition(run),
        summary_parameters=(
            SummaryParameter("rounds", "rounds"),
            SummaryParameter("range", "range", "native"),
        ),
    )
    client = make_client(tmp_path, gui, recipes=(sample,))
    with closing(client.context.session):
        reply = client.call("sample", {})
        assert reply.data["status"] == "finished", reply.data
        assert reply.data["actual"] == {
            "rounds": {"value": 3, "source": "caller_rounds"},
            "range": {
                "value": {"start": 1.0, "stop": 2.0, "expts": 11},
                "source": "caller_range",
                "unit": "native",
            },
        }

        assert not any(method == "tab.edit_cfg" for method, _ in client.transport.sent)


@pytest.mark.parametrize(
    "field,source", [("missing", "caller"), ("rounds", ""), ("range", " ")]
)
def test_retain_source_rejects_invalid_fields_and_labels_without_rpc(
    tmp_path: Path, field: str, source: str
) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = list(client.transport.sent)
        with pytest.raises(ValueError, match="Unknown cfg field|source must be"):
            tab.retain_source(field, source)
        assert client.transport.sent == before


def test_set_fields_writes_one_batch_and_keeps_none_default(
    tmp_path: Path,
) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(
                SummaryParameter("frequency", "modules.readout.pulse_cfg.freq"),
                SummaryParameter("rounds", "rounds"),
            ),
        ).open_tab("lookback")
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
        assert len(edits) == 2
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


def test_actual_mixed_sweep_keeps_the_complete_parameter(tmp_path: Path) -> None:
    gui = FieldGui()

    def run(session: RecipeSession) -> RecipeGenerator:
        tab = session.open_tab("lookback")
        tab.set_sweep("range", start=10.0, stop=20.0)
        yield tab.run()

    sample = replace(
        definition(run),
        summary_parameters=(SummaryParameter("range", "range", "native"),),
    )
    client = make_client(tmp_path, gui, recipes=(sample,))
    with closing(client.context.session):
        initial = client.call("sample", {})
        assert initial.data["status"] == "finished"
        expected = {
            "value": {"start": 10.0, "stop": 20.0, "expts": 11},
            "source": {"start": "explicit", "stop": "explicit", "expts": "gui_default"},
            "unit": "native",
        }
        assert initial.data["actual"]["range"] == expected
        before = list(client.transport.sent)
        gui.publication["tree"]["children"]["range"]["inputs"]["start"] = scalar(99.0)[
            "input"
        ]
        key = initial.data["execution"]
        for detail in ("summary", "full"):
            assert (
                client.call("status", {"execution": key, "detail": detail})["actual"][
                    "range"
                ]
                == expected
            )
        assert client.transport.sent == before
