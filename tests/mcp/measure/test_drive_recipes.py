"""Drive recipe behavior through shipped tools and the GUI wire boundary."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from threading import Event
from typing import Any

import pytest
from simpleeval import simple_eval
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import PNG, LookbackGui, scalar, section
from ._support import MeasureClient, full_execution_reply, make_client


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


def skip_writeback(client: MeasureClient, question: ToolReply) -> ToolReply:
    """Finish the captured Primary question without writing the GUI draft."""
    assert question.data["status"] == "awaiting_answer", question.data
    before = list(client.transport.sent)
    reply = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "skipped"}
    )
    assert reply.data["status"] == "finished", reply.data
    assert client.transport.sent == before
    return reply


class TimeRabiGui(LookbackGui):
    """Rabi wire collaborator using the configured publication as reset defaults."""

    def __init__(self, md=None, *, interactive=False):
        super().__init__()
        self.md = md or {}
        self.interactive = interactive
        self.done = Event()
        self.writes: list[dict[str, object]] = []
        self.adapter_name = "twotone/rabi/len_rabi"
        self.target_name = "pi_len"
        self.fit_summary = {"pi_len": 0.21, "pi_len_err": 0.01, "pi2_len": 0.11}
        tree = self.publication["tree"]["children"]
        tree["reps"] = scalar(19)
        tree["sweep"] = section(
            length={
                "kind": "sweep",
                "valid": True,
                "inputs": {
                    key: scalar(value)["input"]
                    for key, value in {"start": 0.04, "stop": 0.8, "expts": 61}.items()
                },
            }
        )
        tree["modules"]["children"]["readout"]["ref"] = "<Custom:Pulse Readout>"
        tree["modules"]["children"]["qub_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {
                "freq": scalar(0.0),
                "gain": scalar(0.14),
                "waveform": section(length=scalar(1.0)),
            },
        }

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"] == ["sweep", "length"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "length"
                ]["inputs"]
                inputs.update(
                    {
                        key: scalar(value)["input"]
                        for key, value in edit["value"].items()
                    }
                )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})
        modules = self.publication["tree"]["children"]["modules"]["children"]
        for edit in ordinary:
            if edit["path"][0] == "modules" and len(edit["path"]) == 2:
                name = edit["value"]["__ref"]
                if name is not None and name not in ("drive", "calibrated", "reset"):
                    modules[edit["path"][1]].update(
                        valid=False, error="unknown library"
                    )

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.interact":
            if params.get("payload", {}).get("command") == "done":
                self.done.set()
            return {
                "operation_id": 93,
                "plugin": "rabi-picker",
                "state": {},
                "info": {},
                "commands": [{"name": "done"}],
                "preview_active": True,
                "figure": {"png_b64": base64.b64encode(PNG).decode()}
                if params.get("include_figure", True)
                else None,
            }
        if (
            method == "operation.await"
            and params["operation_id"] == 93
            and self.interactive
            and not self.done.is_set()
        ):
            return {"reason": "timeout", "status": "interactive"}
        if method == "tab.writeback_write":
            self.writes.append(deepcopy(params))
            return {
                "written": [
                    {
                        "id": item["id"],
                        "kind": "md",
                        "target": self.target_name,
                        "before": {"value": 0.14},
                        "after": {"value": 0.21},
                    }
                    for item in params["write"]
                ]
            }
        return self._native_reply(method, params)

    def _native_reply(self, method, params):
        if method == "tab.new":
            assert params == {"adapter_name": self.adapter_name}
            return {"tab_id": "t"}
        if method == "tab.writeback_preview":
            assert params["subtab_id"] == "analysis"
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "md-1",
                        "kind": "metadict",
                        "target_name": self.target_name,
                        "proposed": 0.21,
                        "current": 0.14,
                        "selected": False,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        if method == "tab.get_post_analyze_result":
            assert "operation_id" not in params
            return {"summary": None}
        if method == "tab.get_analyze_result" and "operation_id" not in params:
            return {"summary": {}}
        result = super().__call__(method, params)
        if method == "tab.analyze":
            result["interactive"] = self.interactive
        if method == "tab.get_analyze_result":
            result["summary"] = dict(self.fit_summary)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = self.adapter_name
        if method == "context.snapshot":
            result["ml"]["modules"] = {
                "drive": {"type": "pulse"},
                "calibrated": {"type": "readout/pulse"},
                "reset": {"type": "reset"},
            }
        return result


@pytest.mark.parametrize("number", [1, 1.0])
def test_time_rabi_number_inputs_publish_float_frequency_gain_and_length(
    tmp_path, number
):
    with recipe_client(tmp_path, TimeRabiGui({"r_f": 7200.0})) as client:
        data = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call(
                    "time_rabi",
                    {
                        "frequency_mhz": number,
                        "gain": number,
                        "max_length_us": number,
                        "points": 3,
                    },
                ),
            ),
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        for path in ("modules.qub_pulse.freq", "modules.qub_pulse.gain"):
            assert fields[path]["value"] == 1.0
            assert type(fields[path]["value"]) is float
        sweep = fields["sweep.length"]["value"]
        assert sweep["stop"] == 1.0
        assert type(sweep["stop"]) is float
        assert sweep["expts"] == 3
        assert type(sweep["expts"]) is int
        by_path = {
            tuple(edit["path"]): edit["value"]
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
        }
        for path in (
            ("modules", "qub_pulse", "freq"),
            ("modules", "qub_pulse", "gain"),
        ):
            assert by_path[path] == 1.0
            assert type(by_path[path]) is float
        assert type(by_path["sweep", "length"]["stop"]) is float
        assert fields["modules.qub_pulse.gain"]["source"] == "gain"
        assert fields["sweep.length"]["source"] == {
            "start": "gui_default",
            "stop": "max_length_us",
            "expts": "points",
        }


def test_time_rabi_without_pi_uses_explicit_frequency_and_preserves_gui_start(tmp_path):
    gui = TimeRabiGui({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call(
                    "time_rabi",
                    {
                        "frequency_mhz": 6150.0,
                        "gain": 0.21,
                        "max_length_us": 1.5,
                        "points": 71,
                        "reps": 13,
                        "rounds": 9,
                    },
                ),
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.qub_pulse.freq"]["value"] == 6150.0
        assert fields["modules.qub_pulse.freq"]["source"] == "frequency_mhz"
        assert fields["modules.qub_pulse.gain"]["value"] == 0.21
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 1.5,
            "expts": 71,
        }
        assert fields["reps"]["value"] == 13
        assert fields["rounds"]["value"] == 9
        assert fields["modules.reset"]["source"] == "disabled"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")


class AmplitudeRabiGui(TimeRabiGui):
    def __init__(self, md=None, *, interactive=False):
        super().__init__(md, interactive=interactive)
        self.adapter_name = "twotone/rabi/amp_rabi"
        self.target_name = "pi_gain"
        self.fit_summary = {"pi_gain": 0.21, "pi_gain_err": 0.01, "pi2_gain": 0.11}
        tree = self.publication["tree"]["children"]
        tree["sweep"] = section(
            gain={
                "kind": "sweep",
                "valid": True,
                "inputs": {
                    key: scalar(value)["input"]
                    for key, value in {"start": -0.2, "stop": 0.7, "expts": 43}.items()
                },
            }
        )

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"] == ["sweep", "gain"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "gain"
                ]["inputs"]
                inputs.update(
                    {
                        key: scalar(value)["input"]
                        for key, value in edit["value"].items()
                    }
                )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})


@pytest.mark.parametrize("number", [1, 1.0])
def test_amplitude_rabi_number_array_publishes_floats_and_integer_counts(
    tmp_path, number
):
    with recipe_client(tmp_path, AmplitudeRabiGui({"r_f": 7200.0})) as client:
        data = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call(
                    "amplitude_rabi",
                    {
                        "frequency_mhz": number,
                        "pulse_length_us": number,
                        "gain_range": [number - 1, number],
                        "points": 3,
                        "reps": 2,
                        "rounds": 1,
                    },
                ),
            ),
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        for path in ("modules.qub_pulse.freq", "modules.qub_pulse.waveform.length"):
            assert fields[path]["value"] == 1.0
            assert type(fields[path]["value"]) is float
        sweep = fields["sweep.gain"]["value"]
        assert sweep == {"start": 0.0, "stop": 1.0, "expts": 3}
        assert type(sweep["start"]) is float
        assert type(sweep["stop"]) is float
        assert type(sweep["expts"]) is int
        for name, count in (("reps", 2), ("rounds", 1)):
            assert fields[name]["value"] == count
            assert type(fields[name]["value"]) is int
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
            if edit["path"] == ["sweep", "gain"]
        ]
        assert len(edits) == 1
        assert type(edits[0]["value"]["start"]) is float
        assert type(edits[0]["value"]["stop"]) is float
        assert fields["sweep.gain"]["source"] == {
            "start": "gain_range",
            "stop": "gain_range",
            "expts": "points",
        }
        assert (
            fields["modules.qub_pulse.waveform.length"]["source"] == "pulse_length_us"
        )


def test_amplitude_rabi_without_pi_uses_gain_range_and_fixed_pulse(tmp_path):
    gui = AmplitudeRabiGui({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call(
                    "amplitude_rabi",
                    {
                        "frequency_mhz": 6150.0,
                        "pulse_length_us": 0.23,
                        "gain_range": [-0.1, 0.5],
                        "points": 29,
                        "reps": 13,
                        "rounds": 9,
                    },
                ),
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.qub_pulse.freq"]["value"] == 6150.0
        assert fields["modules.qub_pulse.waveform.length"]["value"] == 0.23
        assert fields["sweep.gain"]["value"] == {
            "start": -0.1,
            "stop": 0.5,
            "expts": 29,
        }
        assert fields["reps"]["value"] == 13
        assert fields["rounds"]["value"] == 9
        assert fields["modules.reset"]["source"] == "disabled"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")


@pytest.mark.parametrize(
    "recipe,gui_type,sweep,expected",
    [
        ("time_rabi", TimeRabiGui, "length", {"start": 0.04, "stop": 0.8, "expts": 61}),
        (
            "amplitude_rabi",
            AmplitudeRabiGui,
            "gain",
            {"start": -0.2, "stop": 0.7, "expts": 43},
        ),
    ],
)
@pytest.mark.parametrize("source", ["explicit", "library", "q_f", "invalid_library"])
def test_rabi_frequency_precedence_preserves_gui_defaults(
    tmp_path, recipe, gui_type, sweep, expected, source
):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    drive = gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"]
    arguments: dict[str, Any] = {"reuse_tab_id": "t"}
    if source in ("explicit", "library", "invalid_library"):
        drive["ref"] = "drive"
        drive["children"]["freq"] = scalar(
            6280.0 if source != "invalid_library" else None
        )
    if source == "explicit":
        arguments["frequency_mhz"] = 6150.0
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client, skip_writeback(client, client.call(recipe, arguments))
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        frequency = fields["modules.qub_pulse.freq"]
        assert frequency["value"] == {"explicit": 6150.0, "library": 6280.0}.get(
            source, 6300.0
        )
        assert frequency["source"] == {
            "explicit": "frequency_mhz",
            "library": "library:drive",
        }.get(source, "q_f")
        assert fields[f"sweep.{sweep}"]["value"] == expected
        assert fields[f"sweep.{sweep}"]["source"] == {
            "start": "gui_default",
            "stop": "gui_default",
            "expts": "gui_default",
        }
        assert fields["reps"]["value"] == 19
        assert fields["rounds"]["value"] == 3
        scalar_path = (
            "modules.qub_pulse.gain"
            if recipe == "time_rabi"
            else "modules.qub_pulse.waveform.length"
        )
        assert fields[scalar_path]["value"] == (0.14 if recipe == "time_rabi" else 1.0)
        assert fields[scalar_path]["source"] == "gui_default"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.reset_cfg") == 1
        assert "tab.new" not in methods


@pytest.mark.parametrize(
    "recipe,gui_type",
    [
        ("time_rabi", TimeRabiGui),
        ("amplitude_rabi", AmplitudeRabiGui),
    ],
)
@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_rabi_inline_frequency_does_not_substitute_for_missing_sources(
    tmp_path, recipe, gui_type, reuse_tab_id
):
    gui = gui_type()
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, {"reuse_tab_id": reuse_tab_id}).data
        assert data["status"] == "needs_parameters", data
        assert {item["parameter"] for item in data["missing"]} == {
            "frequency_mhz",
            "readout_ref",
        }
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize(
    "recipe,gui_type,arguments",
    [
        ("time_rabi", TimeRabiGui, {"frequency_mhz": True}),
        ("time_rabi", TimeRabiGui, {"gain": float("nan")}),
        ("time_rabi", TimeRabiGui, {"max_length_us": float("inf")}),
        ("time_rabi", TimeRabiGui, {"points": 2.5}),
        ("time_rabi", TimeRabiGui, {"points": 1.0}),
        ("time_rabi", TimeRabiGui, {"reps": 1.0}),
        ("time_rabi", TimeRabiGui, {"rounds": 1.0}),
        ("amplitude_rabi", AmplitudeRabiGui, {"pulse_length_us": True}),
        ("amplitude_rabi", AmplitudeRabiGui, {"frequency_mhz": float("nan")}),
        ("amplitude_rabi", AmplitudeRabiGui, {"gain_range": [0.1]}),
        ("amplitude_rabi", AmplitudeRabiGui, {"gain_range": [0.1, 0.2, 0.3]}),
        ("amplitude_rabi", AmplitudeRabiGui, {"gain_range": [True, 0.2]}),
        ("amplitude_rabi", AmplitudeRabiGui, {"gain_range": [0.1, float("inf")]}),
        (
            "amplitude_rabi",
            AmplitudeRabiGui,
            {"gain_range": {"start": 0.1, "stop": 0.2}},
        ),
        ("amplitude_rabi", AmplitudeRabiGui, {"rounds": False}),
    ],
)
def test_rabi_invalid_explicit_values_fail_before_preparing(
    tmp_path, recipe, gui_type, arguments
):
    gui = gui_type()
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, arguments).data
        assert data["status"] == "failed", data
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )
        assert not gui.ran


@pytest.mark.parametrize(
    "recipe,gui_type",
    [
        ("time_rabi", TimeRabiGui),
        ("amplitude_rabi", AmplitudeRabiGui),
    ],
)
def test_rabi_valid_library_sources_work_without_metadata_and_enable_explicit_reset(
    tmp_path, recipe, gui_type
):
    gui = gui_type()
    drive = gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"]
    drive["children"]["freq"] = scalar(6280.0)
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call(
                    recipe,
                    {
                        "drive_ref": "drive",
                        "readout_ref": "calibrated",
                        "use_reset": "reset",
                    },
                ),
            ),
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.qub_pulse.freq"]["value"] == 6280.0
        assert fields["modules.qub_pulse.freq"]["source"] == "library:drive"
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5000.0
        assert fields["modules.reset"]["value"] == "reset"


@pytest.mark.parametrize(
    "recipe,gui_type",
    [
        ("time_rabi", TimeRabiGui),
        ("amplitude_rabi", AmplitudeRabiGui),
    ],
)
def test_rabi_stale_edit_stops_without_retry_or_run(tmp_path, recipe, gui_type):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["tab.edit_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale_cfg",
                "message": "changed",
            },
        }
        data = client.call(recipe, {"reuse_tab_id": "t"}).data
        assert data["status"] == "failed", data
        assert not gui.ran
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.edit_cfg") == 1
        assert "tab.new" not in methods


@pytest.mark.parametrize(
    "recipe,gui_type,target",
    [
        ("amplitude_rabi", AmplitudeRabiGui, "pi_gain"),
        ("time_rabi", TimeRabiGui, "pi_len"),
    ],
)
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("decision", ["accepted", "skipped"])
def test_rabi_primary_handoff_waits_for_question_before_actual_write(
    tmp_path, recipe, gui_type, target, interactive, decision
):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0}, interactive=interactive)
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(recipe, {})
        execution = reply.data["execution"]
        if interactive:
            assert reply.data["status"] == "interactive", reply.data
            assert gui.raw_saved
            assert not gui.writes
            before = list(client.transport.sent)
            status = client.call("status", {"execution": execution})
            assert status["question_items"] is None
            assert client.transport.sent == before
            reply = client.call(
                "tab_interact", {"tab": "t", "payload": {"command": "done"}}
            )
            assert reply.data["execution"] == execution
        assert reply.data["status"] == "awaiting_answer", reply.data
        assert tuple(reply.data["question_items"]) == (target,)
        assert reply.data["previews"]["primary"]
        assert not gui.writes
        before = list(client.transport.sent)
        status = client.call("status", {"execution": execution})
        assert status["question_preview"]["primary"]["items"][0]["selected"] is False
        assert client.transport.sent == before
        reply = client.call("answer", {"recipe": execution, "decision": decision})
        assert reply.data["status"] == "finished", reply.data
        receipts = reply.data["writeback"]["receipts"]
        if decision == "accepted":
            assert gui.writes == [
                {"tab_id": "t", "subtab_id": "analysis", "write": [{"id": "md-1"}]}
            ]
            assert receipts[0]["completed"][0]["written"][0]["target"] == target
        else:
            assert client.transport.sent == before
            assert not gui.writes
            assert not receipts
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")


@pytest.mark.parametrize(
    "recipe,gui_type",
    [("amplitude_rabi", AmplitudeRabiGui), ("time_rabi", TimeRabiGui)],
)
def test_rabi_cancelled_run_stops_before_raw_save(tmp_path, recipe, gui_type):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["operation.await"] = {
            "ok": True,
            "result": {"reason": "completed", "status": "cancelled"},
        }
        reply = full_execution_reply(client, client.call(recipe, {}))
        assert reply.data["status"] == "cancelled", reply.data
        assert reply.data["run_outcome"]["status"] == "cancelled"
        assert reply.data["tab"] == "t"
        assert not gui.raw_saved
        assert not gui.writes
        methods = [method for method, _ in client.transport.sent]
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods


@pytest.mark.parametrize(
    "recipe,gui_type",
    [("amplitude_rabi", AmplitudeRabiGui), ("time_rabi", TimeRabiGui)],
)
def test_rabi_cancelled_question_retains_preview_without_writing(
    tmp_path, recipe, gui_type
):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        question = client.call(recipe, {})
        assert question.data["status"] == "awaiting_answer", question.data
        execution = question.data["execution"]
        before = list(client.transport.sent)
        client.call("cancel", {"execution": execution})
        reply = client.call("wait", {"execution": execution, "timeout": 5})
        assert reply.data["status"] == "cancelled", reply.data
        assert reply.data["previews"]["primary"]
        assert not reply.data["writeback"]["receipts"]
        assert client.transport.sent == before
        assert not gui.writes


@pytest.mark.parametrize(
    "recipe,gui_type",
    [("amplitude_rabi", AmplitudeRabiGui), ("time_rabi", TimeRabiGui)],
)
@pytest.mark.parametrize("parameter", ["readout_ref", "drive_ref", "use_reset"])
def test_rabi_invalid_reference_stops_without_fallback(
    tmp_path, recipe, gui_type, parameter
):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(recipe, {parameter: "absent"})
        assert reply.data["status"] == "failed", reply.data
        assert not gui.ran
        assert not gui.raw_saved
        assert "tab.run_start" not in [method for method, _ in client.transport.sent]


@pytest.mark.parametrize(
    "arguments",
    [
        {"reuse_tab_id": " "},
        {"readout_ref": " "},
        {"drive_ref": " "},
        {"use_reset": " "},
        {"points": 1},
        {"reps": 1.0},
        {"rounds": 1.0},
    ],
)
@pytest.mark.parametrize(
    "recipe,gui_type",
    [("amplitude_rabi", AmplitudeRabiGui), ("time_rabi", TimeRabiGui)],
)
def test_rabi_invalid_arguments_fail_before_gui_binding(
    tmp_path, recipe, gui_type, arguments
):
    gui = gui_type({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(recipe, arguments)
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["phase"] == "preparing"
        assert "context.snapshot" not in [method for method, _ in client.transport.sent]
        before = list(client.transport.sent)
        status = client.call("status", {"execution": reply.data["execution"]})
        assert status["status"] == "failed"
        assert client.transport.sent == before


class DriveGui(LookbackGui):
    def __init__(self, md=None):
        super().__init__()
        self.md = md or {}
        self.publication["tree"]["children"]["modules"]["children"]["readout"][
            "ref"
        ] = "<Custom:Pulse Readout>"
        self.publication["tree"]["children"].update(
            reps=scalar(17),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        key: scalar(value)["input"]
                        for key, value in {
                            "start": 4500.0,
                            "stop": 5500.0,
                            "expts": 41,
                        }.items()
                    },
                }
            ),
        )
        self.publication["tree"]["children"]["modules"]["children"]["qub_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": None,
            "error": None,
            "children": {
                "freq": scalar(0.0),
                "gain": scalar(0.12),
                "waveform": section(length=scalar(2.0)),
            },
        }

        if "qf_w" in self.md:
            inputs = self.publication["tree"]["children"]["sweep"]["children"]["freq"][
                "inputs"
            ]
            for key, sign in (("start", "-"), ("stop", "+")):
                expression = f"q_f {sign} 2.5 * qf_w"
                inputs[key] = {
                    **scalar(
                        simple_eval(expression, names=self.md)
                        if "q_f" in self.md
                        else None
                    )["input"],
                    "mode": "expression",
                    "raw": expression,
                }

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"] == ["sweep", "freq"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "freq"
                ]["inputs"]
                for key, value in edit["value"].items():
                    if isinstance(value, dict):
                        expression = value["__expr"]
                        inputs[key] = {
                            **scalar(simple_eval(expression, names=self.md))["input"],
                            "mode": "expression",
                            "raw": expression,
                        }
                    else:
                        inputs[key] = scalar(value)["input"]
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})
        for edit in ordinary:
            if edit["path"] == ["modules", "readout"] and edit["value"] == {
                "__ref": "direct"
            }:
                readout = self.publication["tree"]["children"]["modules"]["children"][
                    "readout"
                ]
                readout["children"] = {"ro_freq": scalar(7100.0)}

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": "twotone/freq"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "context.snapshot":
            result["ml"]["modules"] = {
                "calibrated": {"type": "readout/pulse"},
                "direct": {"type": "readout/direct"},
                "drive": {"type": "pulse"},
                "reset": {"type": "reset/pulse"},
            }
        if method == "tab.snapshot":
            result = deepcopy(result)
            result["tabs"][0]["adapter_name"] = "twotone/freq"
        return result


@pytest.mark.parametrize("readout_ref", [None, "calibrated", "direct"])
def test_twotone_runs_once_with_gui_sweep_and_preserved_readout(tmp_path, readout_ref):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    client = make_client(tmp_path, gui)
    try:
        reply = full_execution_reply(
            client,
            client.call(
                "twotone_spectrum",
                {
                    "readout_ref": readout_ref,
                    "gain": 0.18,
                    "pulse_length_us": 3.0,
                    "points": 31,
                    "reps": 23,
                    "rounds": 7,
                },
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["sweep.freq"]["value"] == {
            "start": 6090.0,
            "stop": 6110.0,
            "expts": 31,
        }
        assert fields["modules.qub_pulse.gain"]["value"] == 0.18
        assert fields["modules.qub_pulse.waveform.length"]["value"] == 3.0
        assert fields["reps"]["value"] == 23
        assert fields["rounds"]["value"] == 7
        assert fields["modules.reset"]["source"] == "disabled"
        if readout_ref == "direct":
            assert fields["modules.readout.ro_freq"]["value"] == 7100.0
        else:
            expected = 5000.0 if readout_ref else 7200.0
            assert fields["modules.readout.pulse_cfg.freq"]["value"] == expected
            assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == expected
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"center_mhz": True},
        {"center_mhz": float("nan")},
        {"span_mhz": float("inf")},
        {"span_mhz": 0.0},
        {"span_mhz": -1.0},
        {"gain": False},
        {"pulse_length_us": "bad"},
        {"points": 2.5},
        {"points": True},
        {"reps": 1.5},
        {"rounds": False},
        {"readout_ref": ""},
        {"drive_ref": 1},
        {"use_reset": True},
        {"reuse_tab_id": " "},
    ],
)
def test_twotone_rejects_invalid_explicit_values_before_preparation(
    tmp_path, arguments
):
    gui = DriveGui()
    client = make_client(tmp_path, gui)
    try:
        reply = full_execution_reply(client, client.call("twotone_spectrum", arguments))
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["phase"] == "preparing"
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )
        assert not gui.ran
    finally:
        client.context.session.close()


def test_twotone_requires_calibrated_readout_instead_of_inline_template(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0})
    client = make_client(tmp_path, gui)
    try:
        reply = full_execution_reply(client, client.call("twotone_spectrum", {}))
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"readout_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_twotone_reports_all_missing_frequency_sources_without_running(
    tmp_path, reuse_tab_id
):
    gui = DriveGui()
    client = make_client(tmp_path, gui)
    try:
        reply = full_execution_reply(
            client, client.call("twotone_spectrum", {"reuse_tab_id": reuse_tab_id})
        )
        methods = [method for method, _ in client.transport.sent]
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
            "readout_ref",
        }
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize("span_mhz", [None, 80.0])
def test_twotone_explicit_center_removes_missing_calibration_dependency(
    tmp_path, span_mhz
):
    gui = DriveGui({"qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "twotone_spectrum", {"center_mhz": 6300.0, "span_mhz": span_mhz}
            ),
        )
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["center_mhz"]["value"] == 6300.0
        assert fields["span_mhz"]["value"] == (20.0 if span_mhz is None else 80.0)


def test_twotone_preserves_valid_library_leaf_and_fills_missing_leaf(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    readout["ref"] = "calibrated"
    readout["children"]["ro_cfg"]["children"]["ro_freq"] = scalar(None)
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(client, client.call("twotone_spectrum", {}))
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5000.0
        assert (
            fields["modules.readout.pulse_cfg.freq"]["source"] == "library:calibrated"
        )
        assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == 7200.0
        assert fields["modules.readout.ro_cfg.ro_freq"]["source"] == "r_f"


@pytest.mark.parametrize("parameter", ["readout_ref", "drive_ref", "use_reset"])
def test_twotone_invalid_reference_stops_without_fallback(tmp_path, parameter):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})

    def respond(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg":
            for node in result["tree"]["children"]["modules"]["children"].values():
                if node.get("ref") == "unknown":
                    node["error"] = "Unknown library key"
                    result["status"] = "Invalid"
        return result

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("twotone_spectrum", {parameter: "unknown"})
        )
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["reason"] == "invalid_cfg"
        assert not gui.ran
        assert sum(method == "tab.edit_cfg" for method, _ in client.transport.sent) == 1


def test_twotone_reuse_applies_explicit_drive_and_reset_once(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "twotone_spectrum",
                {"reuse_tab_id": "t", "drive_ref": "drive", "use_reset": "reset"},
            ),
        )
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["modules.qub_pulse"] == {"value": "drive", "source": "explicit"}
        assert fields["modules.reset"] == {"value": "reset", "source": "explicit"}
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert methods.count("tab.reset_cfg") == 1
        assert methods.count("tab.run_start") == 1


@pytest.mark.parametrize("failure", ["stale", "missing", "busy", "wrong_adapter"])
def test_twotone_reuse_failure_stops_without_retry_or_replacement(tmp_path, failure):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        if failure == "stale":
            client.transport.replies["tab.reset_cfg"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_cfg",
                    "message": "changed",
                },
            }
        else:
            snapshot = gui("tab.snapshot", {"tab_id": "t"})
            if failure == "missing":
                snapshot["tabs"] = []
            elif failure == "busy":
                snapshot["tabs"][0]["interaction"]["is_analyzing"] = True
            else:
                snapshot["tabs"][0]["adapter_name"] = "other"
            client.transport.replies["tab.snapshot"] = {"ok": True, "result": snapshot}
        reply = full_execution_reply(
            client, client.call("twotone_spectrum", {"reuse_tab_id": "t"})
        )
        assert reply.is_error
        assert reply.data["tab"] == "t"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert "tab.run_start" not in methods
        assert methods.count("tab.reset_cfg") == int(failure == "stale")
