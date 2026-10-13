"""Generic batch and expression author edits retain the GUI cfg protocol."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.mcp.measure import recipe
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_actual_support import cfg_value
from ._recipe_execution_support import definition
from ._recipe_support import LookbackGui, recipe_client
from ._support import make_client


class FieldGui(LookbackGui):
    """Recording scalar/sweep edit owner; formulas use supplied logical values."""

    def __init__(self) -> None:
        super().__init__()
        self.md.update({"$custom.low": 10.0, "$custom.high": 20.0})


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
                SummaryParameter("frequency", "modules.readout.module.pulse_cfg.freq"),
                SummaryParameter("rounds", "rounds"),
            ),
        ).open_tab("lookback")
        before = len(client.transport.sent)
        tab.set_fields(
            [
                recipe.RecipeFieldEdit("rounds", None, "ignored"),
                recipe.RecipeFieldEdit(
                    "modules.readout.module.pulse_cfg.freq",
                    recipe.RecipeExpression(expr="$custom.low"),
                    "caller_low",
                ),
                recipe.RecipeFieldEdit(
                    "modules.readout.module.ro_cfg.ro_freq", 22.0, "caller_high"
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
                            "path": [
                                "modules",
                                "readout",
                                "module",
                                "pulse_cfg",
                                "freq",
                            ],
                            "value": {"expr": "$custom.low"},
                        },
                        {
                            "path": [
                                "modules",
                                "readout",
                                "module",
                                "ro_cfg",
                                "ro_freq",
                            ],
                            "value": 22.0,
                        },
                    ],
                },
            )
        ]

        assert cfg_value(gui.publication, "rounds") == 3
        assert gui.publication["input"]["modules"]["readout"]["module"]["pulse_cfg"][
            "freq"
        ] == {"expr": "$custom.low"}
        assert (
            cfg_value(gui.publication, "modules.readout.module.pulse_cfg.freq") == 10.0
        )
        snapshot = tab.run().snapshot()
        assert snapshot is not None
        assert (
            snapshot.actual["fields"]["modules.readout.module.pulse_cfg.freq"].get(
                "source"
            )
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
        tab.set("rounds", recipe.RecipeExpression(expr="$custom.low"), source="input")
        tab.set_sweep(
            "range",
            start=recipe.RecipeExpression(expr="$custom.low"),
            stop=recipe.RecipeExpression(expr="$custom.high"),
        )
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
        ]
        assert len(edits) == 2
        assert edits[1]["edits"] == [
            {"path": ["range", "start"], "value": {"expr": "$custom.low"}},
            {"path": ["range", "stop"], "value": {"expr": "$custom.high"}},
        ]
        assert cfg_value(gui.publication, "range.start") == 10.0
        assert cfg_value(gui.publication, "range.stop") == 20.0
        assert cfg_value(gui.publication, "range.n") == 11


def test_batch_invalid_reply_retains_edit_without_rollback(tmp_path: Path) -> None:
    gui = FieldGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        gui.publication["errors"] = [{"path": ["rounds"], "message": "outside range"}]
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
        gui.owner.write({("range", "start"): 99.0})
        key = initial.data["execution"]
        for detail in ("summary", "full"):
            assert (
                client.call("status", {"execution": key, "detail": detail})["actual"][
                    "range"
                ]
                == expected
            )
        assert client.transport.sent == before
