"""Author-facing cfg helpers through the fixed GUI transport seam."""

from copy import deepcopy

import pytest
from simpleeval import simple_eval
from zcu_tools.mcp.measure.recipe import (
    RecipeNeedsParameters,
    RecipeSession,
    WritebackError,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, recipe_client, scalar, section


def test_author_edits_preserve_defaults_and_use_each_returned_cfg_ref(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set("rounds", None)
        tab.use_library("modules.readout", None)
        assert gui.publication["cfg_ref"]["revision"] == "2"
        tab.set("rounds", 7)
        tab.use_library("modules.readout", "readout")
        tab.disable_library("modules.reset")

    assert gui.publication["tree"]["children"]["rounds"]["input"]["resolved"] == 7
    assert (
        gui.publication["tree"]["children"]["modules"]["children"]["readout"]["ref"]
        == "readout"
    )
    assert gui.publication["cfg_ref"]["revision"] == "5"


@pytest.mark.parametrize(
    "busy,adapter,reason",
    [(True, "lookback", "tab_busy"), (False, "other", "wrong_experiment")],
)
def test_author_reuse_rejects_busy_or_wrong_adapter_without_reset(
    tmp_path, busy, adapter, reason
):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = adapter
            reply["tabs"][0]["interaction"]["is_running"] = busy
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


def test_author_missing_required_library_is_a_named_parameter_gap(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            tab.use_library("modules.readout", None, required="readout_ref")
        assert [item.parameter for item in error.value.missing] == ["readout_ref"]
    assert gui.publication["cfg_ref"]["revision"] == "2"


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


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("frequency", [None, 6200.0])
def test_author_frequency_hides_both_readout_shapes(tmp_path, single, frequency):
    gui = LookbackGui()
    gui.md["r_f"] = 6100.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    if single:
        readout["children"] = {"ro_freq": scalar(5000.0)}
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_frequency(
            "modules.readout",
            frequency,
            calibration="resonator",
            required="frequency_mhz",
        )
    expected = frequency if frequency is not None else 6100.0
    if single:
        assert readout["children"]["ro_freq"]["input"]["resolved"] == expected
    else:
        assert (
            readout["children"]["pulse_cfg"]["children"]["freq"]["input"]["resolved"]
            == expected
        )
        assert (
            readout["children"]["ro_cfg"]["children"]["ro_freq"]["input"]["resolved"]
            == expected
        )


@pytest.mark.parametrize("prefer_library", [True, False])
def test_author_frequency_only_uses_a_library_when_requested(tmp_path, prefer_library):
    gui = LookbackGui()
    gui.md["r_f"] = 6100.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    readout["ref"] = "readout"

    def respond(method, params):
        reply = gui(method, params)
        if method == "context.snapshot":
            reply["ml"]["modules"]["readout"] = {}
        return reply

    with recipe_client(tmp_path, respond) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_frequency(
            "modules.readout",
            calibration="resonator",
            prefer_library=prefer_library,
            required="frequency_mhz",
        )
    assert readout["children"]["pulse_cfg"]["children"]["freq"]["input"][
        "resolved"
    ] == (5000.0 if prefer_library else 6100.0)


def test_author_frequency_missing_calibration_does_not_partially_edit(tmp_path):
    gui = LookbackGui()
    before = deepcopy(gui.publication)
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            tab.set_frequency(
                "modules.readout", calibration="resonator", required="frequency_mhz"
            )
        assert [item.parameter for item in error.value.missing] == ["frequency_mhz"]
    assert gui.publication == before


def test_author_qubit_frequency_uses_the_named_calibration(tmp_path):
    gui = LookbackGui()
    gui.md.update(r_f=6100.0, q_f=4500.0)
    gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"] = {
        "kind": "reference",
        "valid": True,
        "ref": None,
        "children": {"freq": scalar(4000.0)},
    }
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_frequency(
            "modules.qub_pulse.freq", calibration="qubit", required="frequency_mhz"
        )
    assert (
        gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"][
            "children"
        ]["freq"]["input"]["resolved"]
        == 4500.0
    )


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), True])
def test_author_rejects_nonfinite_or_boolean_frequency_before_editing(
    tmp_path, frequency
):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(ValueError, match="finite"):
            tab.set_frequency(
                "modules.readout",
                frequency,
                calibration="resonator",
                required="frequency_mhz",
            )
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_stale_edit_propagates_without_refresh_or_retry(tmp_path):
    gui = LookbackGui()

    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["tab.edit_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale",
                "message": "cfg changed",
            },
        }
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(GuiRpcError) as error:
            tab.set("rounds", 5)
        assert error.value.reason == "stale"
        assert sum(method == "tab.edit_cfg" for method, _ in client.transport.sent) == 1
        assert sum(method == "tab.get_cfg" for method, _ in client.transport.sent) == 1


class _SweepGui(LookbackGui):
    """Cfg collaborator that resolves declared expressions like the GUI owner."""

    def __init__(self):
        super().__init__()
        self.md.update(
            r_f=6100.0, rf_w=4.0, q_f=4500.0, qf_w=3.0, flx_half=0.001, flx_int=0.003
        )
        self.device = {
            "unit": "A",
            "type_name": "YOKOGS200",
            "info": {"type": "YOKOGS200"},
        }
        self.publication["tree"]["children"].update(
            dev=section(flux_dev=scalar("coil")),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        "start": self.input("r_f - 2 * rf_w"),
                        "stop": self.input("r_f + 3 * rf_w"),
                        "expts": self.input(41),
                    },
                },
                flux={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        "start": self.input("2 * flx_int - flx_half"),
                        "stop": self.input("2 * flx_half - flx_int"),
                        "expts": self.input(19),
                    },
                },
            ),
        )

    def input(self, value: object) -> dict[str, object]:
        """Resolve a scalar or expression against this collaborator's md."""
        result: dict[str, object] = scalar(
            simple_eval(value, names=self.md) if isinstance(value, str) else value
        )["input"]
        if isinstance(value, str):
            result.update(mode="expression", raw=value)
        return result

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"][0] == "sweep":
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    edit["path"][1]
                ]["inputs"]
                for key, value in edit["value"].items():
                    inputs[key] = self.input(
                        value["__expr"] if isinstance(value, dict) else value
                    )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})

    def __call__(self, method, params):
        if method == "value.list":
            return {"values": [{"key": "device.flux.name"}]}
        if method == "value.read":
            assert params == {"key": "device.flux.name"}
            return {"value": "coil"}
        if method == "device.snapshot":
            return {"snapshot": {"name": params["name"], **self.device}}
        return super().__call__(method, params)


@pytest.mark.parametrize("center", [None, 6500.0])
@pytest.mark.parametrize("span", [None, 30.0])
@pytest.mark.parametrize("expts", [None, 7])
def test_author_frequency_sweep_retains_gui_width_and_omitted_count(
    tmp_path, center, span, expts
):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set_frequency_sweep(
            "sweep.freq",
            calibration="resonator",
            center_mhz=center,
            span_mhz=span,
            expts=expts,
        )
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["freq"]["inputs"]
    expected_center = 6100.0 if center is None else center
    expected_span = 20.0 if span is None else span
    assert inputs["start"]["resolved"] == expected_center - expected_span / 2
    assert inputs["stop"]["resolved"] == expected_center + expected_span / 2
    assert inputs["expts"]["resolved"] == (41 if expts is None else expts)
    if span is None:
        assert "rf_w" in inputs["start"]["raw"]
        assert "rf_w" in inputs["stop"]["raw"]


def test_author_qubit_frequency_sweep_retains_its_own_width(tmp_path):
    gui = _SweepGui()
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["freq"]["inputs"]
    inputs.update(start=gui.input("q_f - qf_w"), stop=gui.input("q_f + qf_w"))
    with recipe_client(tmp_path, gui) as client:
        RecipeSession(client.context).open_tab("lookback").set_frequency_sweep(
            "sweep.freq", calibration="qubit"
        )
    assert inputs["start"]["resolved"] == 4497.0
    assert inputs["stop"]["resolved"] == 4503.0


def test_author_frequency_sweep_reports_all_missing_inputs_before_edit(tmp_path):
    gui = _SweepGui()
    gui.md.clear()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            tab.set_frequency_sweep("sweep.freq", calibration="resonator")
        assert [item.parameter for item in error.value.missing] == [
            "center_mhz",
            "span_mhz",
        ]
    assert gui.publication["cfg_ref"]["revision"] == "2"


@pytest.mark.parametrize("expts", [None, 5])
def test_author_flux_sweep_preserves_gui_expressions_without_converting_coordinates(
    tmp_path, expts
):
    gui = _SweepGui()
    before = deepcopy(
        gui.publication["tree"]["children"]["sweep"]["children"]["flux"]["inputs"]
    )
    with recipe_client(tmp_path, gui) as client:
        RecipeSession(client.context).open_tab("lookback").set_flux_sweep(
            "sweep.flux", expts=expts
        )
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["flux"]["inputs"]
    assert inputs["start"] == before["start"]
    assert inputs["stop"] == before["stop"]
    assert inputs["expts"]["resolved"] == (19 if expts is None else expts)


@pytest.mark.parametrize(
    "start,stop", [(None, 1.0), (1.0, None), (1.0, 1.0), (float("inf"), 1.0)]
)
def test_author_flux_sweep_rejects_invalid_explicit_endpoints(tmp_path, start, stop):
    gui = _SweepGui()
    with (
        recipe_client(tmp_path, gui) as client,
        pytest.raises(ValueError, match="endpoints"),
    ):
        RecipeSession(client.context).open_tab("lookback").set_flux_sweep(
            "sweep.flux", start=start, stop=stop
        )
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_flux_sweep_requires_calibrations_only_for_omitted_endpoints(tmp_path):
    gui = _SweepGui()
    gui.md.clear()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            tab.set_flux_sweep("sweep.flux")
        assert [item.parameter for item in error.value.missing] == ["flux_range"]
        tab.set_flux_sweep("sweep.flux", start=-2.0, stop=3.0)
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["flux"]["inputs"]
    assert inputs["start"]["resolved"] == -2.0
    assert inputs["stop"]["resolved"] == 3.0


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


@pytest.mark.parametrize("name", [None, "coil"])
def test_author_flux_device_checks_units_without_changing_coordinates(tmp_path, name):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_flux_device(name, unit="A")
    assert (
        gui.publication["tree"]["children"]["dev"]["children"]["flux_dev"]["input"][
            "resolved"
        ]
        == "coil"
    )


@pytest.mark.parametrize(
    "device,unit",
    [
        (
            {"unit": "A", "type_name": "YOKOGS200", "info": {"type": "YOKOGS200"}},
            "native",
        ),
        ({"unit": "A", "type_name": "YOKOGS200", "info": {"type": "YOKOGS200"}}, "V"),
        (
            {"unit": "none", "type_name": "FakeDevice", "info": {"type": "FakeDevice"}},
            None,
        ),
        (
            {"unit": "none", "type_name": "FakeDevice", "info": {"type": "Other"}},
            "native",
        ),
    ],
)
def test_author_flux_device_rejects_mismatched_or_unconfirmed_units_before_edit(
    tmp_path, device, unit
):
    gui = _SweepGui()
    gui.device = device
    with recipe_client(tmp_path, gui) as client:
        with pytest.raises(GuiRpcError) as error:
            RecipeSession(client.context).open_tab("lookback").use_flux_device(
                "coil", unit=unit
            )
        assert error.value.reason == "invalid_device"
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_fake_flux_requires_explicit_native_opt_in(tmp_path):
    gui = _SweepGui()
    gui.device = {
        "unit": "none",
        "type_name": "FakeDevice",
        "info": {"type": "FakeDevice"},
    }
    with recipe_client(tmp_path, gui) as client:
        RecipeSession(client.context).open_tab("lookback").use_flux_device(
            "coil", unit="native"
        )
    assert gui.publication["cfg_ref"]["revision"] == "3"


def test_author_flux_device_without_a_default_reports_a_named_gap(tmp_path):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["value.list"] = {"ok": True, "result": {"values": []}}
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            tab.use_flux_device()
        assert [item.parameter for item in error.value.missing] == ["flux_device"]
    assert gui.publication["cfg_ref"]["revision"] == "2"


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
                        "kind": "md",
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
                            "kind": "md",
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
