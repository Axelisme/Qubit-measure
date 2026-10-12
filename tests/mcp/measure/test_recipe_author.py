"""Author-facing cfg helpers through the fixed GUI transport seam."""

from copy import deepcopy

import pytest
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


@pytest.mark.parametrize("mode", ["inline", "ref"])
def test_author_reads_owner_reference_mode_and_child_readonly(tmp_path, mode):
    gui = LookbackGui()
    reference = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    reference.update(
        mode=mode,
        readonly=False,
        ref="Q1.readout.seed" if mode == "ref" else "<Custom:Readout>",
    )
    frequency = reference["children"]["ro_cfg"]["children"]["ro_freq"]
    frequency["readonly"] = mode == "ref"
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        state = tab.read_field("modules.readout")
        assert state.mode == mode
        assert not state.readonly
        assert tab.read_field("modules.readout.ro_cfg.ro_freq").readonly == (
            mode == "ref"
        )


def test_author_edits_preserve_defaults_and_use_each_returned_cfg_ref(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set("rounds", None)
        assert tab.use_library("modules.readout", None) is None
        assert gui.publication["cfg_ref"]["revision"] == "2"
        tab.set("rounds", 7)
        # This short reference is a wire-edit literal, not native admission.
        assert tab.use_library("modules.readout", "readout") is None
        tab.disable_library("modules.reset")

    assert gui.publication["tree"]["children"]["rounds"]["input"]["resolved"] == 7
    assert (
        gui.publication["tree"]["children"]["modules"]["children"]["readout"]["ref"]
        == "readout"
    )
    assert gui.publication["cfg_ref"]["revision"] == "5"


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
        tab.set("rounds", 9)
        cfg_calls = [
            (method, params)
            for method, params in client.transport.sent
            if method in {"tab.get_cfg", "tab.reset_cfg", "tab.edit_cfg", "tab.new"}
        ]
    assert [method for method, _ in cfg_calls] == [
        "tab.get_cfg",
        "tab.reset_cfg",
        "tab.edit_cfg",
    ]
    assert cfg_calls[1][1]["expected"] == {"cfg_id": "cfg", "revision": "2"}
    assert cfg_calls[2][1]["expected"] == {"cfg_id": "cfg", "revision": "3"}
    assert gui.publication["cfg_ref"]["revision"] == "4"
    assert gui.publication["tree"]["children"]["rounds"]["input"]["resolved"] == 9


@pytest.mark.parametrize("field", ["", "missing", "modules..readout", "rounds.child"])
def test_author_rejects_unknown_fields_without_editing(tmp_path, field):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(ValueError, match="field"):
            tab.set(field, 5)
    assert gui.publication["cfg_ref"]["revision"] == "2"


@pytest.mark.parametrize(
    "name,current_reference,expected",
    [
        pytest.param(
            None,
            None,
            MissingParameter(
                "readout_ref", "Provide a valid modules.readout library reference"
            ),
            id="absent-reference",
        ),
        pytest.param(
            None,
            "unknown",
            MissingParameter(
                "readout_ref", "Provide a valid modules.readout library reference"
            ),
            id="unavailable-reference",
        ),
        pytest.param(None, "Q1.readout.seed", None, id="retained-library"),
        pytest.param("Q1.readout.seed", None, None, id="explicit-library"),
    ],
)
def test_author_required_library_returns_parameter_gap_or_none(
    tmp_path, name, current_reference, expected
):
    gui = LookbackGui()
    gui.library["Q1.readout.seed"] = {}
    gui.publication["tree"]["children"]["modules"]["children"]["readout"]["ref"] = (
        current_reference
    )
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        assert tab.use_library("modules.readout", name, required="readout_ref") == (
            expected
        )
    assert gui.publication["cfg_ref"]["revision"] == ("2" if name is None else "3")


def test_author_explicit_invalid_library_does_not_fall_back(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.edit_cfg":
            reply["tree"]["children"]["modules"]["children"]["readout"].update(
                valid=False, error="unknown library"
            )
        return reply

    with recipe_client(tmp_path, respond) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(GuiRpcError) as error:
            tab.use_library("modules.readout", "unknown", required="readout_ref")
        assert error.value.reason == "invalid_cfg"
        assert sum(method == "tab.edit_cfg" for method, _ in client.transport.sent) == 1


@pytest.mark.parametrize("expts", [0, 1, 1.5, True])
def test_author_sweep_rejects_invalid_counts_before_edit(tmp_path, expts):
    gui = _SweepGui()
    with (
        recipe_client(tmp_path, gui) as client,
        pytest.raises(ValueError, match="expts"),
    ):
        RecipeSession(client.context).open_tab("lookback").set_sweep(
            "sweep.freq", expts=expts
        )
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_sweep_only_edits_supplied_fields(tmp_path):
    gui = _SweepGui()
    before = deepcopy(
        gui.publication["tree"]["children"]["sweep"]["children"]["freq"]["inputs"]
    )
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_sweep("sweep.freq")
        assert gui.publication["cfg_ref"]["revision"] == "2"
        tab.set_sweep("sweep.freq", start=6000.0, expts=2)
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["freq"]["inputs"]
    assert inputs["start"]["resolved"] == 6000.0
    assert inputs["stop"] == before["stop"]
    assert inputs["expts"]["resolved"] == 2


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
                SummaryParameter(name, name)
                for name in (
                    "sweep.freq",
                    "sweep.freq.start",
                    "sweep.freq.stop",
                    "sweep.freq.expts",
                )
            ),
        ).open_tab("lookback")
        tab.set_sweep("sweep.freq", start=6000.0, expts=2, sources=sources)
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
        assert fields["sweep.freq.expts"].get("source") == "points"


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
        with pytest.raises(ValueError, match="sources"):
            tab.set_sweep("sweep.freq", start=6000.0, sources=sources)
        assert not any(method == "tab.edit_cfg" for method, _ in client.transport.sent)


def test_author_disconnected_binding_does_not_reconnect_or_replay_an_edit(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        client.transport.close()
        with pytest.raises(GuiRpcError) as error:
            tab.set("rounds", 5)
        assert error.value.reason == "connection_lost"
        assert len(client.transport.sent) == before
    assert gui.publication["cfg_ref"]["revision"] == "2"


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
