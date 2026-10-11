"""Power recipe gain admission, raw saving and run-only preview delivery."""

import base64
from threading import Event

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure import tools_recipes

from ._onetone_recipe_support import (
    PowerGui,
    recipe_client,
)
from ._recipe_actual_support import cfg_sweep, cfg_value
from ._recipe_support import PNG
from ._support import full_execution_reply


@pytest.mark.parametrize(
    "arguments",
    [
        {"gain_range": [False, 1]},
        {"gain_range": [0, float("nan")]},
        {"gain_range": [0, 1, 2]},
        {"gain_range": {"start": 0, "stop": 1}},
        {"gain_points": 1.2},
        {"gain_points": 1.0},
        {"freq_points": 1.0},
        {"gain_points": True},
    ],
)
def test_power_rejects_invalid_gain_inputs_before_preparation(tmp_path, arguments):
    with recipe_client(tmp_path, PowerGui()) as client:
        reply = full_execution_reply(
            client, client.call("onetone_spectrum_over_power", arguments)
        )
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)


@pytest.mark.parametrize("failure", ["source", "base64", "png"])
def test_power_preview_failure_retains_the_saved_raw_and_true_run(tmp_path, failure):
    gui = PowerGui()
    with recipe_client(tmp_path, gui) as client:
        if failure == "source":
            client.transport.replies["tab.get_figure"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "result_superseded",
                    "message": "The source was replaced",
                },
            }
        elif failure in ("base64", "png"):
            client.transport.replies["tab.get_figure"] = {
                "ok": True,
                "result": {
                    "png_b64": "invalid!"
                    if failure == "base64"
                    else base64.b64encode(b"not-png").decode()
                },
            }
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_power", {"center_mhz": 6100.0, "span_mhz": 20.0}
            ),
        )
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        error = reply.data["error"]
        assert error["reason"] == (
            "result_superseded" if failure == "source" else "recipe_failed"
        )
        if failure == "source":
            assert error["code"] == "precondition_failed"
            assert "The source was replaced" in error["message"]
        else:
            assert ("base64" if failure == "base64" else "PNG") in error["message"]
        assert reply.data["raw_save"] == {
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
        assert reply.data["run_outcome"]["status"] == "finished"
        assert reply.data["analysis"] is None
        assert reply.data["writeback"] is None
        assert not reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == methods.count("tab.save_data") == 1
        assert methods.count("tab.get_figure") == 1
        assert methods.index("tab.save_data") < methods.index("tab.get_figure")


@pytest.mark.parametrize(
    "phase, fail", [("raw_save", False), ("preview", False), ("preview", True)]
)
def test_power_cancel_does_not_suppress_fast_save_or_preview_after_completed_run(
    tmp_path, monkeypatch, phase, fail
):
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    entered, release = Event(), Event()
    gui = PowerGui()

    def respond(method, params):
        should_block = (phase == "raw_save" and method == "tab.save_data") or (
            phase == "preview" and method == "tab.get_figure"
        )
        if should_block:
            entered.set()
            assert release.wait(3)
        if phase == "preview" and method == "tab.get_figure" and fail:
            return {"png_b64": "bad-png"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        try:
            initial = full_execution_reply(
                client,
                client.call(
                    "onetone_spectrum_over_power",
                    {"center_mhz": 6100.0, "span_mhz": 20.0},
                ),
            )
            assert entered.wait(2)
            execution = initial.data["execution"]
            client.call("cancel", {"execution": execution})
            release.set()
            terminal = full_execution_reply(
                client, client.call("wait", {"execution": execution, "timeout": 2})
            )
            data = terminal.data
            assert data["status"] == ("failed" if fail else "finished")
            assert data["cancel_requested"]
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
            assert [img.data for img in terminal.images] == ([] if fail else [PNG])
            if fail:
                assert data["error"]["reason"] == "recipe_failed"
                assert "base64" in data["error"]["message"]
            methods = [method for method, _ in client.transport.sent]
            assert "tab.get_figure" in methods
            assert "tab.analyze" not in methods
        finally:
            release.set()


@pytest.mark.parametrize("number", [1, 1.0])
def test_onetone_power_number_ranges_publish_float_endpoints_and_integer_counts(
    tmp_path, number
):
    gui = PowerGui()
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_power",
                {
                    "center_mhz": number,
                    "span_mhz": number,
                    "gain_range": [number - 1, number],
                    "freq_points": 3,
                    "gain_points": 2,
                    "reps": 2,
                    "rounds": 1,
                },
            ),
        ).data
        assert data["status"] == "finished", data
        assert cfg_sweep(gui.publication, "sweep.freq") == {
            "start": 0.5,
            "stop": 1.5,
            "expts": 3,
        }
        assert cfg_sweep(gui.publication, "sweep.gain") == {
            "start": 0.0,
            "stop": 1.0,
            "expts": 2,
        }
        for path in ("sweep.freq", "sweep.gain"):
            sweep = cfg_sweep(gui.publication, path)
            assert type(sweep["start"]) is float
            assert type(sweep["stop"]) is float
            assert type(sweep["expts"]) is int
        for name, count in (("reps", 2), ("rounds", 1)):
            assert cfg_value(gui.publication, name) == count
            assert type(cfg_value(gui.publication, name)) is int
        gain_edits = [
            edit["value"]
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
            if edit["path"] == ["sweep", "gain"]
        ]
        assert len(gain_edits) == 1
        assert type(gain_edits[0]["start"]) is float
        assert type(gain_edits[0]["stop"]) is float
        assert type(gain_edits[0]["expts"]) is int


@pytest.mark.parametrize(
    "arguments, expected_gain",
    [
        ({}, {"start": 0.03, "stop": 0.27, "expts": 13}),
        (
            {
                "reuse_tab_id": "t",
                "gain_range": [0.1, 0.7],
                "gain_points": 7,
                "freq_points": 23,
            },
            {"start": 0.1, "stop": 0.7, "expts": 7},
        ),
        (
            {"center_mhz": 6100.0, "span_mhz": 20.0},
            {"start": 0.03, "stop": 0.27, "expts": 13},
        ),
        (
            {
                "reuse_tab_id": "t",
                "center_mhz": 6100.0,
                "span_mhz": 20.0,
                "gain_range": [0.1, 0.7],
                "gain_points": 7,
                "freq_points": 23,
            },
            {"start": 0.1, "stop": 0.7, "expts": 7},
        ),
    ],
)
def test_power_saves_raw_and_delivers_only_a_run_preview(
    tmp_path, arguments, expected_gain
):
    gui = PowerGui()
    with recipe_client(tmp_path, gui) as client:
        initial = client.call("onetone_spectrum_over_power", arguments)
        reply = full_execution_reply(client, initial)
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert data["analysis_mode"] == "none"
        assert data["analysis"] is None
        assert data["writeback"] is None
        assert cfg_sweep(gui.publication, "sweep.gain") == expected_gain
        scalar_gain = data["actual"]["gain"]
        assert scalar_gain["value"] == 0.21
        assert scalar_gain["source"] == "gui_default"
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
        assert reply.images[0].data == PNG
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        waited = client.call("wait", {"execution": data["execution"], "timeout": 0})
        assert summary["artifacts"] == data["raw_save"]["artifacts"]
        assert (
            initial.data["artifacts"]
            == waited.data["artifacts"]
            == summary["artifacts"]
        )
        assert summary["actual"]["gain"] == {
            "value": 0.21,
            "source": "gui_default",
        }
        assert waited.data["actual"]["gain"] == summary["actual"]["gain"]
        assert summary["run_id"] == waited.data["run_id"] == data["raw_save"]["run_id"]
        assert len(client.transport.sent) == before
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert "tab.analyze" not in methods
        assert "tab.writeback_preview" not in methods
        assert methods.index("tab.save_data") < methods.index("tab.get_figure")
