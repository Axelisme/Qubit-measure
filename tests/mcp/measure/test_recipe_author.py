"""Author-facing cfg helpers through the fixed GUI transport seam."""

from copy import deepcopy

import pytest
from zcu_tools.cfg_model import ConfigBase, LogPoints
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.form_adapter import Form
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeSession,
    SweepSources,
    WritebackError,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_field_support import SweepGui as _SweepGui
from ._recipe_support import LookbackGui, recipe_client


@pytest.mark.parametrize(
    "parameter,reason",
    [
        pytest.param("", "Provide a readout reference", id="empty-parameter"),
        pytest.param("readout_ref", "", id="empty-reason"),
    ],
)
def test_missing_parameter_rejects_empty_name_or_reason(parameter, reason):
    with pytest.raises(
        ValueError,
        match="^missing parameters must contain non-empty names and reasons$",
    ):
        MissingParameter(parameter, reason)


class IntegerControls(ConfigBase):
    depths: list[int]


class IntegerGui(LookbackGui):
    cfg_cls = IntegerControls

    def make_form(self):
        return Form(
            IntegerControls,
            {"depths": LogPoints(max=16, points=5, include_zero=False)},
        )


def test_author_integer_points_replaces_the_complete_log_branch_with_a_list(tmp_path):
    gui = IntegerGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_integer_points(("depths",), [7])
        assert tab.read_field(("depths",)).input == {"kind": "list", "values": [7]}
        cfg = gui.owner.result()
        assert not isinstance(cfg, FormErrors), cfg
        assert cfg.model_dump()["depths"] == [7]


def test_author_omitted_integer_points_preserves_unfinished_gui_draft(tmp_path):
    gui = IntegerGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        gui.owner.write({("depths", "points"): 0})
        before = deepcopy(tab.read_field(("depths",)).input)
        tab.set_integer_points(("depths",), None)
        assert tab.read_field(("depths",)).input == before


def test_author_integer_points_reports_owner_errors_and_retains_invalid_list(tmp_path):
    gui = IntegerGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(GuiRpcError, match="Invalid") as raised:
            tab.set_integer_points(("depths",), [0])
        assert raised.value.reason == "invalid_cfg"
        assert tab.read_field(("depths",)).input == {"kind": "list", "values": [0]}


def test_author_edits_preserve_defaults(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set(("rounds",), None)
        assert tab.use_library(("modules", "readout"), None) is None
        tab.set(("rounds",), 7)
        assert tab.use_library(("modules", "readout"), "R1.readout.main") is None
        tab.disable_library(("modules", "reset"))

    assert gui.publication["input"]["rounds"] == 7
    assert gui.publication["input"]["modules"]["readout"] == {
        "kind": "ref",
        "name": "R1.readout.main",
    }


@pytest.mark.parametrize(
    "busy_flag,adapter,reason",
    [
        ("is_running", "lookback", "tab_busy"),
        ("is_analyzing", "lookback", "tab_busy"),
        (None, "other", "wrong_experiment"),
    ],
)
def test_author_reuse_rejects_busy_or_wrong_adapter_without_reset(
    tmp_path, busy_flag, adapter, reason
):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = adapter
            if busy_flag is not None:
                reply["tabs"][0]["interaction"][busy_flag] = True
        return reply

    with recipe_client(tmp_path, respond) as client:
        with pytest.raises(GuiRpcError) as error:
            RecipeSession(client.context).open_tab("lookback", reuse="t")
        assert error.value.reason == reason
        assert not any(
            method in {"tab.new", "tab.reset_cfg"}
            for method, _ in client.transport.sent
        )


def test_author_reuse_resets_the_observed_cfg_before_editing(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback", reuse="t")
        tab.set(("rounds",), 9)
        cfg_calls = [
            (method, params)
            for method, params in client.transport.sent
            if method
            in {
                "tab.get_cfg",
                "tab.reset_cfg",
                "tab.edit_recipe_fields",
                "tab.new",
            }
        ]
    assert [method for method, _ in cfg_calls] == [
        "tab.reset_cfg",
        "tab.edit_recipe_fields",
    ]
    assert cfg_calls[0][1] == {"tab_id": "t"}
    assert cfg_calls[1][1] == {
        "tab_id": "t",
        "edits": [{"path": ["rounds"], "value": 9}],
    }
    assert gui.publication["input"]["rounds"] == 9


@pytest.mark.parametrize(
    "field", [(), ("missing",), ("modules", "", "readout"), ("rounds", "child")]
)
def test_author_rejects_unknown_fields_without_editing(tmp_path, field):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = gui.owner.input.data
        sent_before = len(client.transport.sent)
        if field in (("missing",), ("rounds", "child")):
            with pytest.raises(GuiRpcError) as error:
                tab.set(field, 5)
            assert error.value.code == "invalid_params"
            assert error.value.reason == "form_path"
            assert error.value.data == {
                "form_path": list(field),
                "path": list(field),
                "edit_index": 0,
            }
        else:
            with pytest.raises(ValueError, match="field"):
                tab.set(field, 5)
        assert gui.owner.input.data == before
        assert client.transport.sent[sent_before:] == (
            [
                (
                    "tab.edit_recipe_fields",
                    {
                        "tab_id": "t",
                        "edits": [{"path": list(field), "value": 5}],
                    },
                )
            ]
            if field in (("missing",), ("rounds", "child"))
            else []
        )


@pytest.mark.parametrize(
    "name,current_reference,expected",
    [
        (
            None,
            None,
            MissingParameter(
                "readout_ref", "Provide a valid ('modules', 'readout') library key"
            ),
        ),
        (None, "R1.readout.main", None),
        ("R1.readout.main", None, None),
    ],
)
def test_author_required_library_returns_parameter_gap_or_none(
    tmp_path, name, current_reference, expected
):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        if current_reference is not None:
            tab.use_library(("modules", "readout"), current_reference)
        before = len(client.transport.sent)
        assert (
            tab.use_library(("modules", "readout"), name, required="readout_ref")
            == expected
        )
        assert client.transport.sent[before:] == (
            [
                (
                    "tab.read_recipe_field",
                    {"tab_id": "t", "path": ["modules", "readout"]},
                )
            ]
            if name is None
            else [
                (
                    "tab.select_recipe_module",
                    {
                        "tab_id": "t",
                        "path": ["modules", "readout"],
                        "action": "use_library",
                        "key": name,
                    },
                )
            ]
        )


def test_author_explicit_invalid_library_does_not_fall_back(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        initial = tab.read_field(("modules", "readout"))
        before = list(client.transport.sent)
        with pytest.raises(GuiRpcError) as error:
            tab.use_library(
                ("modules", "readout"), "unknown.readout.main", required="readout_ref"
            )
        assert error.value.code == "invalid_params"
        assert error.value.reason == "library_key"
        assert error.value.data == {
            "path": ["modules", "readout"],
            "selector_path": ["modules", "readout"],
            "available_keys": ["R1.readout.main", "R2.readout.main"],
        }
        assert tab.read_field(("modules", "readout")) == initial
        assert client.transport.sent[len(before) :] == [
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "t",
                    "path": ["modules", "readout"],
                    "action": "use_library",
                    "key": "unknown.readout.main",
                },
            ),
            (
                "tab.read_recipe_field",
                {"tab_id": "t", "path": ["modules", "readout"]},
            ),
        ]


@pytest.mark.parametrize("expts", [0, 1, 1.5, True])
def test_author_sweep_rejects_invalid_counts_before_edit(tmp_path, expts):
    gui = _SweepGui()
    with (
        recipe_client(tmp_path, gui) as client,
        pytest.raises(ValueError, match="expts"),
    ):
        RecipeSession(client.context).open_tab("lookback").set_sweep(
            ("sweep", "freq"), expts=expts
        )


def test_author_sweep_only_edits_supplied_fields(tmp_path):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = deepcopy(tab.read_field(("sweep", "freq")).input)
        tab.set_sweep(("sweep", "freq"))
        tab.set_sweep(("sweep", "freq"), start=6000.0, expts=2)
        inputs = tab.read_field(("sweep", "freq")).input
        if not isinstance(inputs, dict) or not isinstance(before, dict):
            raise TypeError("Expected Form range input")
        assert inputs["start"] == 6000.0
        assert inputs["stop"] == before["stop"]
        assert inputs["n"] == 2


def test_author_sweep_captures_individual_labels_and_omitted_gui_sources(tmp_path):
    gui = _SweepGui()
    sources: SweepSources = {
        "start": "start_mhz",
        "stop": "stop_mhz",
        "expts": "points",
    }
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab = RecipeSession(
            client.context,
            summary_parameters=tuple(
                SummaryParameter(name, path)
                for name, path in (
                    ("sweep.freq", ("sweep", "freq")),
                    ("sweep.freq.start", ("sweep", "freq", "start")),
                    ("sweep.freq.stop", ("sweep", "freq", "stop")),
                    ("sweep.freq.n", ("sweep", "freq", "n")),
                )
            ),
        ).open_tab("lookback")
        tab.set_sweep(("sweep", "freq"), start=6000.0, expts=2, sources=sources)
        sources["start"] = "changed after preparation"
        capture = tab.run().snapshot()
        assert capture is not None
        fields = capture.actual["fields"]
        assert fields["sweep.freq"].get("source") == {
            "start": "start_mhz",
            "stop": "gui_default",
            "expts": "points",
        }
        assert fields["sweep.freq.start"].get("source") == "start_mhz"
        assert fields["sweep.freq.stop"].get("source") == "gui_default"
        assert fields.get("sweep.freq.n") == {"value": 2, "source": "points"}


@pytest.mark.parametrize("key", ["start", "stop", "expts"])
@pytest.mark.parametrize("label", ["", " "])
def test_author_sweep_rejects_empty_sources_before_any_edit(tmp_path, key, label):
    gui = _SweepGui()
    sources: SweepSources = {
        "start": "start_mhz",
        "stop": "stop_mhz",
        "expts": "points",
    }
    sources[key] = label
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        with pytest.raises(ValueError, match="sources"):
            tab.set_sweep(("sweep", "freq"), start=6000.0, sources=sources)
        assert client.transport.sent[before:] == []


def test_author_disconnected_binding_does_not_reconnect_or_replay_an_edit(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        client.transport.close()
        with pytest.raises(GuiRpcError) as error:
            tab.set(("rounds",), 5)
        assert error.value.reason == "connection_lost"
        assert len(client.transport.sent) == before


@pytest.mark.parametrize("items", [None, []])
def test_author_accept_returns_only_the_current_confirmed_writes(tmp_path, items):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        client.transport.replies["tab.get_analyze_result"] = {
            "ok": True,
            "result": {"summary": {}},
        }
        client.transport.replies["tab.get_post_analyze_result"] = {
            "ok": True,
            "result": {"summary": None},
        }
        client.transport.replies["tab.writeback_preview"] = {
            "ok": True,
            "result": {
                "has_draft": True,
                "items": [
                    {"id": "current", "target_name": "offset", "selected": False}
                ],
                "destination_context": {},
            },
        }
        client.transport.replies["tab.writeback_write"] = {
            "ok": True,
            "result": {
                "written": [
                    {
                        "id": "current",
                        "kind": "parameter",
                        "target": "offset",
                        "before": None,
                        "after": 1.0,
                    }
                ]
            },
        }
        receipt = tab.accept(items)
        assert receipt["status"] == "finished"
        if items is None:
            assert [
                (stage["stage"], [item["id"] for item in stage["written"]])
                for stage in receipt["completed"]
            ] == [("primary", ["current"])]
            assert receipt["skipped"] == ["post"]
        else:
            assert receipt["completed"] == []
            assert set(receipt["skipped"]) == {"primary", "post"}
        assert receipt["not_started"] == []


def test_author_accept_raises_with_confirmed_prefix_from_the_writeback_owner(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
            client.transport.replies[method] = {"ok": True, "result": {"summary": {}}}
        client.transport.replies["tab.writeback_preview"] = lambda params: {
            "ok": True,
            "result": {
                "has_draft": True,
                "items": [
                    {
                        "id": "current",
                        "target_name": params["subtab_id"],
                        "selected": False,
                    }
                ],
                "destination_context": {},
            },
        }
        client.transport.replies["tab.writeback_write"] = lambda params: (
            {
                "ok": True,
                "result": {
                    "written": [
                        {
                            "id": "current",
                            "kind": "parameter",
                            "target": "sample",
                            "before": None,
                            "after": 1.0,
                        }
                    ]
                },
            }
            if params["subtab_id"] == "analysis"
            else {
                "ok": False,
                "error": {"code": "internal_error", "message": "write interrupted"},
            }
        )
        with pytest.raises(WritebackError) as error:
            tab.accept(["analysis", "post_analysis"])
        assert error.value.receipt.get("failed_stage") == "post"
        assert error.value.receipt.get("failed_stage_may_have_partial_writes") is True
        assert [
            (stage["stage"], [item["id"] for item in stage["written"]])
            for stage in error.value.receipt["completed"]
        ] == [("primary", ["current"])]
