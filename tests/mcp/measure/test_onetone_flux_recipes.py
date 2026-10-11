"""Flux recipe coordinates, device/unit admission and saved survey evidence."""

from typing import Any

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from ._onetone_recipe_support import FluxGui, OnetoneGui, recipe_client, skip_writeback
from ._recipe_actual_support import cfg_sweep, cfg_value
from ._recipe_support import PNG
from ._support import full_execution_reply


@pytest.mark.parametrize("parameter", ["flux_range", "flux_points"])
@pytest.mark.parametrize("null", [False, True], ids=["missing", "null"])
def test_flux_requires_explicit_range_and_count_before_gui_work(
    tmp_path, parameter, null
):
    gui = FluxGui()
    arguments: dict[str, object] = {
        "flux_device": "coil",
        "flux_unit": "A",
        "flux_range": [0.005, -0.001],
        "flux_points": 19,
    }
    if null:
        arguments[parameter] = None
    else:
        del arguments[parameter]
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum_over_flux", arguments)
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["phase"] == "preparing"
        assert not any(method.startswith("tab.") for method, _ in client.transport.sent)


def test_flux_reports_all_missing_sources_in_one_handoff(tmp_path):
    gui = FluxGui()
    gui.md.clear()

    def respond(method, params):
        if method == "value.list":
            return {"values": []}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        "flux_device": "coil",
                        "flux_unit": "A",
                    },
                },
            ),
        )
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
        }
        assert not gui.ran
        assert not reply.is_error


@pytest.mark.parametrize(
    "arguments",
    [
        {"flux_device": ""},
        {"flux_device": False},
        {"flux_device": " "},
        {"flux_unit": ""},
        {"flux_unit": " "},
        {"flux_unit": False},
        {"flux_unit": 1},
        {"freq_points": True},
        {"freq_points": 1.5},
        {"freq_points": 1.0},
        {"flux_points": 1.0},
        {"flux_points": False},
        {"flux_range": [0, True]},
        {"flux_range": [0, float("inf")]},
        {"flux_range": [0]},
        {"flux_range": [0, 1, 2]},
        {"flux_range": {"start": 0, "stop": 1}},
        {"flux_range": "0,1"},
    ],
)
def test_flux_rejects_invalid_explicit_inputs_instead_of_missing_handoff(
    tmp_path, arguments
):
    gui = OnetoneGui(experiment="onetone/flux_dep")
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {"flux_range": [0.005, -0.001], "flux_points": 19, **arguments},
            ),
        )
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)


def test_flux_rejects_unknown_explicit_device_without_selecting_default(tmp_path):
    gui = FluxGui()
    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["device.snapshot"] = {
            "ok": False,
            "error": {
                "code": "invalid_params",
                "reason": "device_not_found",
                "message": "No device named missing-coil",
            },
        }
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        "flux_device": "missing-coil",
                        "flux_unit": "A",
                    },
                },
            ),
        )
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "device_not_found"
        assert not gui.ran
        assert ("device.snapshot", {"name": "missing-coil"}) in client.transport.sent


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"flux_device": "coil"},
        {"flux_unit": "A"},
        {"flux_device": None, "flux_unit": "A"},
        {"flux_device": "coil", "flux_unit": None},
    ],
)
def test_flux_requires_device_and_unit_before_gui_preparation(tmp_path, arguments):
    gui = FluxGui()
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client, client.call("onetone_spectrum_over_flux", arguments)
        )
        assert reply.is_error, reply.data
        assert not gui.ran
        assert not client.transport.sent


@pytest.mark.parametrize(
    "device_type,snapshot_unit,requested_unit",
    [
        ("YOKOGS200", "A", "A"),
        ("YOKOGS200", "V", "V"),
        ("FakeDevice", "native", "native"),
    ],
)
def test_flux_explicit_assertion_preserves_coordinates_and_unit(
    tmp_path, device_type, snapshot_unit, requested_unit
):
    gui = FluxGui(
        snapshot={
            "name": "coil",
            "type_name": device_type,
            "unit": snapshot_unit,
            "info": {"type": device_type},
        }
    )
    with recipe_client(tmp_path, gui) as client:
        question = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        "flux_device": "coil",
                        "flux_unit": requested_unit,
                        "center_mhz": 6100.0,
                        "span_mhz": 8.0,
                        "flux_range": [0.005, -0.001],
                    },
                },
            ),
        )
        reply = full_execution_reply(client, skip_writeback(client, question))
        assert reply.data["status"] == "finished", reply.data
        assert cfg_value(gui.publication, "dev.flux_dev") == "coil"
        assert cfg_sweep(gui.publication, "sweep.flux") == {
            "start": 0.005,
            "stop": -0.001,
            "expts": 19,
        }
        assert cfg_value(gui.publication, "flux_unit") == requested_unit
        assert gui.ran


@pytest.mark.parametrize(
    "device_type,snapshot_unit,requested_unit",
    [
        ("YOKOGS200", "A", "V"),
        ("YOKOGS200", "V", "A"),
        ("YOKOGS200", "A", "native"),
        ("FakeDevice", "none", "A"),
        ("FakeDevice", "none", "V"),
    ],
)
def test_flux_explicit_unit_mismatch_fails_before_run(
    tmp_path, device_type, snapshot_unit, requested_unit
):
    gui = FluxGui(
        snapshot={
            "name": "coil",
            "type_name": device_type,
            "unit": snapshot_unit,
            "info": {"type": device_type},
        }
    )
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        "flux_device": "coil",
                        "flux_unit": requested_unit,
                        "center_mhz": 6100.0,
                        "span_mhz": 8.0,
                        "flux_range": [0.005, -0.001],
                    },
                },
            ),
        )
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["reason"] == "invalid_device"
        assert not gui.ran


@pytest.mark.parametrize(
    "explicit, unit, expected",
    [
        (True, "none", "failed"),
    ],
)
def test_flux_does_not_treat_unknown_units_as_physical_flux(
    tmp_path, explicit, unit, expected
):
    gui = FluxGui(snapshot={"name": "coil", "unit": unit})
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        **({"flux_device": "coil"} if explicit else {}),
                        "flux_unit": "native",
                    },
                },
            ),
        )
        assert reply.data["status"] == expected
        assert reply.data["error"]["reason"] == "invalid_device"
        assert not gui.ran


@pytest.mark.parametrize(
    "explicit_range,expected_endpoints",
    [
        ([-0.003, 0.007], (-0.003, 0.007)),
        ([0, 1], (0.0, 1.0)),
        ([0.0, 1.0], (0.0, 1.0)),
    ],
)
def test_flux_saves_one_survey_with_physical_device_and_actual_conditions(
    tmp_path, explicit_range, expected_endpoints
):
    explicit = explicit_range is not None
    gui = FluxGui()
    arguments: dict[str, Any] = {
        "readout_ref": "R1.readout.calibrated",
        "flux_device": "alternate" if explicit else "coil",
        "flux_unit": "V" if explicit else "A",
        "freq_points": 31,
        "reps": 23,
        "rounds": 7,
        "gain": 0.13,
    }
    if explicit:
        arguments.update(
            reuse_tab_id="t",
            flux_device="alternate",
            flux_range=explicit_range,
            flux_points=11,
        )
    with recipe_client(tmp_path, gui) as client:
        question = client.call(
            "onetone_spectrum_over_flux",
            {"flux_range": [0.005, -0.001], "flux_points": 19, **arguments},
        )
        reply = full_execution_reply(client, skip_writeback(client, question))
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert (
            cfg_value(gui.publication, "dev.flux_dev") == "alternate"
            if explicit
            else "coil"
        )
        assert cfg_sweep(gui.publication, "sweep.flux") == {
            "start": expected_endpoints[0],
            "stop": expected_endpoints[1],
            "expts": 11 if explicit else 19,
        }
        assert type(cfg_sweep(gui.publication, "sweep.flux")["start"]) is float
        assert type(cfg_sweep(gui.publication, "sweep.flux")["stop"]) is float
        assert type(cfg_sweep(gui.publication, "sweep.flux")["expts"]) is int
        assert cfg_sweep(gui.publication, "sweep.freq")["expts"] == 31
        assert cfg_value(gui.publication, "modules.readout") == "R1.readout.calibrated"
        assert cfg_value(gui.publication, "modules.readout.pulse_cfg.gain") == 0.13
        assert cfg_value(gui.publication, "reps") == 23
        assert cfg_value(gui.publication, "rounds") == 7
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
        assert data["writeback"]["items"]
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert ("tab.reset_cfg" in methods) is explicit
        if explicit:
            flux_edits = [
                edit["value"]
                for method, params in client.transport.sent
                if method == "tab.edit_cfg"
                for edit in params["edits"]
                if edit["path"] == ["sweep", "flux"]
            ]
            assert len(flux_edits) == 1
            assert type(flux_edits[0]["start"]) is float
            assert type(flux_edits[0]["stop"]) is float
            assert type(flux_edits[0]["expts"]) is int


def test_explicit_flux_saves_report_images_and_actual_conditions(tmp_path):
    gui = FluxGui()
    with recipe_client(tmp_path, gui) as client:
        question = client.call(
            "onetone_spectrum_over_flux",
            {
                "flux_range": [0.005, -0.001],
                "flux_points": 19,
                **{
                    "flux_device": "alternate",
                    "flux_unit": "V",
                    "flux_range": [0.01, 0.04],
                    "flux_points": 7,
                    "center_mhz": 6100.0,
                    "span_mhz": 20.0,
                },
            },
        )
        reply = full_execution_reply(client, skip_writeback(client, question))
        data = reply.data
        assert data["status"] == "finished", data
        assert cfg_value(gui.publication, "dev.flux_dev") == "alternate"
        assert cfg_sweep(gui.publication, "sweep.flux") == {
            "start": 0.01,
            "stop": 0.04,
            "expts": 7,
        }
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
        analysis_report = data["analysis"]["save_report"]
        assert analysis_report["run_id"] == data["raw_save"]["run_id"]
        assert analysis_report["artifacts"]["analysis"]["results"]["members"] == {
            "analysis:primary": [
                {
                    "format": "json",
                    "path": "/runs/run-a/analysis/results.json",
                    "status": "saved",
                    "error": None,
                }
            ],
            "figure:trace": [
                {
                    "format": "png",
                    "path": "/runs/run-a/analysis/trace.png",
                    "status": "saved",
                    "error": None,
                }
            ],
        }
        assert [img.data for img in reply.images] == [PNG]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        assert summary["artifacts"] == {
            **data["raw_save"]["artifacts"],
            **analysis_report["artifacts"],
        }
        assert summary["run_id"] == analysis_report["run_id"]
        assert len(client.transport.sent) == before
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == methods.count("tab.save_data") == 1
        assert methods.count("tab.save_artifacts") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")


def test_fake_flux_native_opt_in_preserves_coordinates_and_saved_result(tmp_path):
    gui = FluxGui(
        snapshot={
            "name": "coil",
            "type_name": "FakeDevice",
            "unit": "native",
            "info": {"type": "FakeDevice", "value": 0.0},
        }
    )
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{
                        "flux_device": "coil",
                        "flux_unit": "native",
                        "flux_range": [-0.25, 1.5],
                        "flux_points": 7,
                    },
                },
            ),
        )
        reply = full_execution_reply(client, skip_writeback(client, reply))
        assert reply.data["status"] == "finished", reply.data
        assert cfg_value(gui.publication, "dev.flux_dev") == "coil"
        assert cfg_sweep(gui.publication, "sweep.flux") == {
            "start": -0.25,
            "stop": 1.5,
            "expts": 7,
        }
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
        assert reply.data["analysis"]["status"] == "finished"
        assert gui.ran


@pytest.mark.parametrize(
    "device,requested_unit,expected",
    [
        ("coil", "A", "finished"),
        ("alternate", "V", "finished"),
        ("coil", "V", "failed"),
        ("alternate", "A", "failed"),
        ("coil", "native", "failed"),
        ("alternate", "native", "failed"),
    ],
)
def test_physical_flux_unit_assertion_is_checked_before_run(
    tmp_path, device, requested_unit, expected
):
    gui = FluxGui()
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{"flux_device": device, "flux_unit": requested_unit},
                },
            ),
        )
        if expected == "finished":
            reply = full_execution_reply(client, skip_writeback(client, reply))
        assert reply.data["status"] == expected, reply.data
        assert gui.ran is (expected == "finished")
        if expected == "finished":
            assert cfg_value(gui.publication, "flux_unit") == requested_unit
        else:
            assert reply.data["error"]["reason"] == "invalid_device"


@pytest.mark.parametrize(
    "type_name,info_type,unit,requested_unit",
    [
        ("FakeDevice", "FakeDevice", "none", None),
        ("FakeDevice", "FakeDevice", "none", "A"),
        ("FakeDevice", "FakeDevice", "none", "V"),
        ("UnknownDevice", "UnknownDevice", "none", "native"),
        ("FakeDevice", "UnknownDevice", "none", "native"),
        ("UnknownDevice", "FakeDevice", "none", "native"),
        ("FakeDevice", None, "none", "native"),
        ("FakeDevice", "FakeDevice", "A", "A"),
        ("FakeDevice", "FakeDevice", "V", "V"),
        ("UnknownDevice", "UnknownDevice", "native", None),
    ],
)
def test_native_flux_rejects_unconfirmed_or_physical_coordinates_before_run(
    tmp_path, type_name, info_type, unit, requested_unit
):
    # A/V select mismatched requests against the real Fake Info native unit.
    snapshot_unit = (
        "native"
        if type_name == info_type == "FakeDevice" and unit in ("A", "V")
        else unit
    )
    gui = FluxGui(
        snapshot={
            "name": "coil",
            "type_name": type_name,
            "unit": snapshot_unit,
            "info": {"type": info_type} if info_type is not None else None,
        }
    )
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "onetone_spectrum_over_flux",
                {
                    "flux_range": [0.005, -0.001],
                    "flux_points": 19,
                    **{"flux_device": "coil", "flux_unit": requested_unit},
                },
            ),
        )
        assert reply.data["status"] == "failed", reply.data
        if requested_unit is None:
            assert reply.is_error
            assert reply.data["error"] == {
                "phase": "preparing",
                "reason": "recipe_failed",
                "message": "Invalid recipe argument flux_unit: None is not of type 'string'",
                "code": None,
            }
            assert not client.transport.sent
        else:
            assert reply.data["error"]["reason"] == "invalid_device"
        assert not gui.ran
