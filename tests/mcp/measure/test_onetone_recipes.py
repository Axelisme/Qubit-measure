"""Onetone promises through the shipped recipe tools and recording GUI."""

from copy import deepcopy

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from ._onetone_recipe_support import (
    FluxGui,
    OnetoneGui,
    PowerGui,
    recipe_client,
    skip_writeback,
)
from ._recipe_actual_support import cfg_sweep, cfg_value
from ._support import full_execution_reply


@pytest.mark.parametrize(
    "arguments",
    [
        {"center_mhz": True},
        {"center_mhz": float("nan")},
        {"span_mhz": 0},
        {"span_mhz": -1},
        {"span_mhz": float("inf")},
        {"gain": False},
        {"gain": "bad"},
        {"points": 2.5},
        {"points": 1.0},
        {"reps": 1.0},
        {"rounds": 1.0},
        {"points": True},
        {"reps": 1.5},
        {"rounds": False},
        {"reuse_tab_id": ""},
        {"readout_ref": 23},
    ],
)
def test_spectrum_rejects_explicit_invalid_input_before_gui_work(tmp_path, arguments):
    gui = OnetoneGui({"$resonator.freq": 6000.0, "$resonator.kappa": 4.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)
        assert reply.data["error"]["phase"] == "preparing"
        assert not gui.ran


@pytest.mark.parametrize(
    "recipe,arguments,expected",
    [
        ("onetone_spectrum", {}, (6094.0, 6106.0)),
        ("onetone_spectrum_over_power", {}, (6094.0, 6106.0)),
        (
            "onetone_spectrum_over_flux",
            {
                "flux_device": "coil",
                "flux_unit": "A",
                "flux_range": [0.005, -0.001],
                "flux_points": 19,
            },
            (6096.0, 6104.0),
        ),
    ],
)
def test_resonator_recipes_write_numeric_calibrated_endpoints(
    tmp_path, recipe, arguments, expected
):
    if recipe == "onetone_spectrum_over_flux":
        gui = FluxGui()
    elif recipe == "onetone_spectrum_over_power":
        gui = PowerGui()
    else:
        gui = OnetoneGui({"$resonator.freq": 6100.0, "$resonator.kappa": 4.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(recipe, arguments)
        assert reply.data["status"] in ("finished", "awaiting_answer"), reply.data
        sweeps = [
            edit["value"]
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
            if edit["path"] == ["sweep", "freq"]
        ]
        assert sweeps == [{"start": expected[0], "stop": expected[1]}]
        assert (
            sum(method == "tab.run_start" for method, _ in client.transport.sent) == 1
        )


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("nan"), True])
def test_resonator_invalid_linewidth_stops_before_sweep_edit_or_run(tmp_path, width):
    gui = OnetoneGui({"$resonator.freq": 6100.0, "$resonator.kappa": width})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", {})
        assert reply.data["status"] in ("failed", "needs_parameters"), reply.data
        assert not any(
            edit["path"] == ["sweep", "freq"]
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
        )
        assert not gui.ran


def test_actual_mixed_sweep_keeps_the_complete_parameter(tmp_path):
    gui = OnetoneGui({"$resonator.freq": 6100.0, "$resonator.kappa": 4.0})
    with recipe_client(tmp_path, gui) as client:
        initial = client.call(
            "onetone_spectrum", {"center_mhz": 6100.0, "span_mhz": 20.0}
        )
        assert initial.data["status"] == "awaiting_answer", initial.data
        expected = {
            "value": {"start": 6090.0, "stop": 6110.0, "expts": 41},
            "source": {"start": "explicit", "stop": "explicit", "expts": "gui_default"},
            "unit": "MHz",
        }
        before = list(client.transport.sent)
        key = initial.data["execution"]
        assert initial.data["actual"]["frequency_sweep"] == expected
        assert (
            client.call("status", {"execution": key})["actual"]["frequency_sweep"]
            == expected
        )
        assert (
            client.call("status", {"execution": key, "detail": "full"})["actual"][
                "frequency_sweep"
            ]
            == expected
        )
        assert client.transport.sent == before


@pytest.mark.parametrize("stage", ["tab.edit_cfg", "tab.run_start"])
def test_spectrum_source_change_fails_without_retry_or_blind_scan(tmp_path, stage):
    gui = OnetoneGui({"$resonator.freq": 6100.0, "$resonator.kappa": 4.0})

    with recipe_client(tmp_path, gui) as client:
        client.transport.replies[stage] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale",
                "message": "Source changed",
            },
        }
        reply = client.call("onetone_spectrum", {})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "stale"
        assert not gui.ran
        assert sum(method == stage for method, _ in client.transport.sent) == 1


def test_spectrum_rejects_invalid_library_reference_without_fallback(tmp_path):
    gui = OnetoneGui({"$resonator.freq": 6100.0, "$resonator.kappa": 4.0})

    def respond(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg":
            reference = result["tree"]["children"]["modules"]["children"]["readout"]
            if reference["ref"] == "unknown":
                reference["error"] = "Unknown library key"
                result["status"] = "Invalid"
        return result

    with recipe_client(tmp_path, respond) as client:
        reply = client.call("onetone_spectrum", {"readout_ref": "unknown"})
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "invalid_cfg"
        assert not gui.ran
        assert sum(m == "tab.edit_cfg" for m, _ in client.transport.sent) == 1


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_onetone_reports_missing_frequency_without_running(tmp_path, reuse_tab_id):
    gui = OnetoneGui()
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", {"reuse_tab_id": reuse_tab_id})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
        }
        assert not reply.is_error
        assert not gui.ran
        methods = [method for method, _ in client.transport.sent]
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
        assert ("tab.new" in methods) is (reuse_tab_id is None)


@pytest.mark.parametrize(
    "md, arguments, start, stop, center_source, span_source",
    [
        (
            {"$resonator.freq": 6100.0, "$resonator.kappa": 4.0},
            {},
            6094.0,
            6106.0,
            "$resonator.freq",
            "gui_linewidth",
        ),
        (
            {"$resonator.kappa": 4.0},
            {"center_mhz": 6200.0},
            6194.0,
            6206.0,
            "explicit",
            "gui_linewidth",
        ),
        (
            {"$resonator.freq": None, "$resonator.kappa": 4.0},
            {"center_mhz": 6200.0},
            6194.0,
            6206.0,
            "explicit",
            "gui_linewidth",
        ),
        (
            {"$resonator.freq": 6100.0},
            {"span_mhz": 8.0},
            6096.0,
            6104.0,
            "$resonator.freq",
            "explicit",
        ),
        (
            {},
            {"center_mhz": 6200.0, "span_mhz": 16.0},
            6192.0,
            6208.0,
            "explicit",
            "explicit",
        ),
        (
            {"$resonator.freq": 6100.0, "$resonator.kappa": 4.0},
            {"center_mhz": 6200.0},
            6194.0,
            6206.0,
            "explicit",
            "gui_linewidth",
        ),
        (
            {"$resonator.freq": 6100.0, "$resonator.kappa": 4.0},
            {"span_mhz": 8.0},
            6096.0,
            6104.0,
            "$resonator.freq",
            "explicit",
        ),
    ],
)
def test_spectrum_saves_one_run_with_gui_derived_frequency_and_averages(
    tmp_path, md, arguments, start, stop, center_source, span_source
):
    gui = OnetoneGui(md)
    deepcopy(gui.publication)
    with recipe_client(tmp_path, gui) as client:
        question = client.call("onetone_spectrum", arguments)
        reply = full_execution_reply(client, skip_writeback(client, question))
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert not reply.is_error
        fields = data["actual"]
        assert cfg_sweep(gui.publication, "sweep.freq") == {
            "start": start,
            "stop": stop,
            "expts": 41,
        }
        source = (
            "gui_calibration"
            if center_source != "explicit" and span_source != "explicit"
            else center_source
            if center_source != "explicit"
            else span_source
        )
        assert fields["frequency_sweep"]["source"] == {
            "start": source,
            "stop": source,
            "expts": "gui_default",
        }
        assert fields["frequency_sweep"]["value"] == {
            "start": start,
            "stop": stop,
            "expts": 41,
        }
        assert cfg_value(gui.publication, "reps") == 17
        assert cfg_value(gui.publication, "rounds") == 3
        assert data["raw_save"] == {
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
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"][0]["proposed"] == 0.24
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
