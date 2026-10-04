"""Coherence behavior through shipped tools and the GUI wire boundary."""

import json
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import pytest

from ._recipe_support import LookbackGui, scalar, section
from ._support import full_execution_reply, make_client


@contextmanager
def recipe_client(tmp_path, gui):
    client = make_client(tmp_path, gui)
    try:
        yield client
    finally:
        client.context.session.close()


class CoherenceGui(LookbackGui):
    def __init__(self, pi_ref="<Custom:Pulse>", adapter="t1", pi2_ref="<Custom:Pulse>"):
        super().__init__()
        self.adapter = adapter
        self.md = {"r_f": 5100.0}
        self.library = {
            "pi": {"type": "pulse"},
            "pi2": {"type": "pulse"},
            "readout": {"type": "readout/pulse"},
            "reset": {"type": "reset"},
        }
        tree = self.publication["tree"]["children"]
        tree["reps"] = scalar(19)
        tree["detune_ratio"] = scalar(0.2)
        tree["modules"]["children"]["pi2_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": pi2_ref,
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }
        tree["sweep"] = section(
            length={
                "kind": "sweep",
                "valid": True,
                "inputs": {
                    key: scalar(value)["input"]
                    for key, value in {"start": 0.04, "stop": 60.0, "expts": 61}.items()
                },
            }
        )
        self.publication["tree"]["children"]["modules"]["children"]["pi_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": pi_ref,
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

        self.defaults = deepcopy(self.publication)

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

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": f"twotone/{self.adapter}"}
            return {"tab_id": "t"}
        if method == "tab.reset_cfg":
            revision = self.publication["cfg_ref"]
            self.publication = deepcopy(self.defaults)
            self.publication["cfg_ref"] = revision
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = f"twotone/{self.adapter}"
        if method == "context.snapshot":
            result["ml"]["modules"] = self.library
        return result


def test_recipe_estimates_share_native_fit_quality_and_precise_issue_paths(tmp_path):
    gui = CoherenceGui(pi_ref="pi")
    quality = {
        "fit": {
            "r2": -0.25,
            "normalized_residual_rms": 0.31,
            "relative_parameter_errors": {"decay_time": None},
            "invalid": [
                {
                    "path": "summary.fit_quality.fit.relative_parameter_errors.decay_time",
                    "reason": "covariance_unavailable",
                }
            ],
        }
    }

    def respond(method, params):
        result = gui(method, params)
        if method == "tab.get_analyze_result":
            result["summary"] = {
                "t1": 25.0,
                "t1_err": None,
                "t1b": 50.0,
                "t1b_err": None,
                "fit_quality": quality,
            }
        return result

    with recipe_client(tmp_path, respond) as client:
        initial = client.call("t1", {})
        assert initial.data["status"] == "finished", initial.data
        execution = initial.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        waited = client.call("wait", {"execution": execution, "timeout": 0}).data
        assert {k: v for k, v in initial.data.items() if k != "elapsed_s"} == summary
        assert {k: v for k, v in waited.items() if k != "elapsed_s"} == summary
        assert len(client.transport.sent) == before
        assert full["analysis"]["result"]["summary"]["fit_quality"] == quality
        expected_invalid = []
        for name, value in (("t1", 25.0), ("t1b", 50.0)):
            estimate = summary["analysis"]["primary"]["estimates"][name]
            assert estimate["value"] == value
            assert estimate["stderr"] is None
            assert estimate["quality"]["fit"]["r2"] == -0.25
            assert estimate["quality"]["fit"]["normalized_residual_rms"] == 0.31
            issue = {
                "path": f"analysis.primary.estimates.{name}.quality.fit.relative_parameter_errors.decay_time",
                "reason": "covariance_unavailable",
            }
            assert estimate["quality"]["fit"]["invalid"] == [issue]
            expected_invalid.append(issue)
        assert summary["invalid"] == expected_invalid
        assert summary["analysis"]["primary"]["details"] == {}
        json.dumps(summary, allow_nan=False)
        json.dumps(full, allow_nan=False)


@pytest.mark.parametrize("recipe", ["t1", "t2ramsey", "t2echo"])
def test_coherence_reuse_discards_old_overrides_and_keeps_gui_expressions(
    tmp_path, recipe
):
    gui = CoherenceGui(pi_ref="pi", pi2_ref="pi2", adapter=recipe)
    inputs = gui.defaults["tree"]["children"]["sweep"]["children"]["length"]["inputs"]
    inputs["stop"].update(mode="expression", raw="calibrated_decay", resolved=37.0)
    gui.publication["tree"]["children"]["sweep"]["children"]["length"]["inputs"][
        "stop"
    ] = scalar(999)["input"]
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client, client.call(recipe, {"reuse_tab_id": "t"})
        ).data
        assert data["status"] == "finished", data
        sweep = data["actual"]["fields"]["sweep.length"]
        assert sweep["value"]["stop"] == 37.0
        assert sweep["input"]["stop"]["raw"] == "calibrated_decay"
        assert sweep["input"]["stop"]["mode"] == "expression"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.reset_cfg") == 1
        assert methods.count("tab.run_start") == 1
        assert "tab.new" not in methods


@pytest.mark.parametrize("recipe", ["t1", "t2ramsey", "t2echo"])
def test_coherence_explicit_readout_and_reset_use_library_without_rf(tmp_path, recipe):
    gui = CoherenceGui(pi_ref="pi", pi2_ref="pi2", adapter=recipe)
    gui.md = {}
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client,
            client.call(recipe, {"readout_ref": "readout", "use_reset": "reset"}),
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.readout"]["value"] == "readout"
        assert fields["modules.reset"] == {"value": "reset", "source": "explicit"}
        assert fields["modules.readout.pulse_cfg.freq"]["source"] == "library:readout"
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5000.0


@pytest.mark.parametrize(
    "recipe,missing",
    [
        ("t2ramsey", {"pi2_ref", "readout_ref"}),
        ("t2echo", {"pi_ref", "pi2_ref", "readout_ref"}),
    ],
)
def test_t2_reports_every_missing_calibration(tmp_path, recipe, missing):
    gui = CoherenceGui(adapter=recipe)
    gui.md = {}
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, {}).data
        assert data["status"] == "needs_parameters", data
        assert {item["parameter"] for item in data["missing"]} == missing
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize("recipe", ["t1", "t2ramsey", "t2echo"])
@pytest.mark.parametrize(
    "failure", ["incompatible_reference", "invalid_cfg", "stale_reuse"]
)
def test_coherence_gui_rejection_never_runs_or_retries(tmp_path, recipe, failure):
    gui = CoherenceGui(pi_ref="pi", pi2_ref="pi2", adapter=recipe)

    def respond(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg" and failure != "stale_reuse":
            result["status"] = "Invalid"
            if failure == "incompatible_reference":
                result["tree"]["children"]["modules"]["children"]["readout"][
                    "error"
                ] = "Incompatible module"
        return result

    with recipe_client(tmp_path, respond) as client:
        arguments = {"readout_ref": "readout"}
        if failure == "stale_reuse":
            arguments["reuse_tab_id"] = "t"
            client.transport.replies["tab.reset_cfg"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_cfg",
                    "message": "Changed",
                },
            }
        data = client.call(recipe, arguments).data
        assert data["status"] == "failed", data
        assert data["tab"] == "t"
        assert not gui.ran
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.new") == (0 if failure == "stale_reuse" else 1)
        if failure == "stale_reuse":
            assert methods.count("tab.reset_cfg") == 1
        assert "tab.run_start" not in methods


@pytest.mark.parametrize("recipe", ["t1", "t2ramsey", "t2echo"])
@pytest.mark.parametrize("stage", ["tab.run_start", "tab.analyze", "tab.get_figure"])
def test_coherence_failure_retains_tab_and_already_saved_paths(tmp_path, recipe, stage):
    gui = CoherenceGui(pi_ref="pi", pi2_ref="pi2", adapter=recipe)
    with recipe_client(tmp_path, gui) as client:
        client.transport.replies[stage] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded",
                "message": "Changed",
            },
        }
        reply = full_execution_reply(client, client.call(recipe, {}))
        assert reply.is_error
        data = reply.data
        assert data["status"] == "failed"
        assert data["tab"] == "t"
        if stage != "tab.run_start":
            assert data["raw_save"]["path"] == "/actual/raw.h5"
            assert data["run_outcome"]["status"] == "finished"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert not any("accept" in method for method in methods)


@pytest.mark.parametrize("recipe", ["t2ramsey", "t2echo"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_t2_rejects_nonfinite_or_boolean_detune_before_preparing(
    tmp_path, recipe, value
):
    gui = CoherenceGui(pi_ref="pi", pi2_ref="pi2", adapter=recipe)
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, {"detune_ratio": value}).data
        assert data["status"] == "failed"
        assert not gui.ran
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )


@pytest.mark.parametrize(
    "recipe,parameter",
    [
        ("t1", "pi_ref"),
        ("t2ramsey", "pi2_ref"),
        ("t2echo", "pi_ref"),
        ("t2echo", "pi2_ref"),
        ("t1", "readout_ref"),
        ("t1", "use_reset"),
    ],
)
def test_coherence_explicit_reference_must_be_a_library_entry(
    tmp_path, recipe, parameter
):
    gui = CoherenceGui(pi_ref="pi", adapter=recipe)
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, {parameter: "<Custom:Pulse>"}).data
        assert data["status"] == "failed", data
        assert data["error"]["reason"] == "invalid_cfg"
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize("recipe", ["t2ramsey", "t2echo"])
def test_t2_runs_with_total_delay_and_unchanged_detune_units(tmp_path, recipe):
    gui = CoherenceGui(adapter=recipe)
    arguments = {
        "pi2_ref": "pi2",
        "max_delay_us": 84.0,
        "points": 43,
        "detune_ratio": 0.37,
        "reps": 7,
        "rounds": 5,
    }
    if recipe == "t2echo":
        arguments["pi_ref"] = "pi"
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(client, client.call(recipe, arguments))
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 84.0,
            "expts": 43,
        }
        assert fields["detune_ratio"]["value"] == 0.37
        assert fields["modules.pi2_pulse"]["value"] == "pi2"
        if recipe == "t2echo":
            assert fields["modules.pi_pulse"]["value"] == "pi"
        assert fields["modules.reset"]["value"] is None
        assert fields["reps"]["value"] == 7
        assert fields["rounds"]["value"] == 5
        assert data["tab"] == "t"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert reply.images
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1


@pytest.mark.parametrize("number", [1, 1.0])
def test_t2_number_delay_and_detune_publish_floats(tmp_path, number):
    gui = CoherenceGui(pi_ref="pi", adapter="t2ramsey", pi2_ref="pi2")
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client,
            client.call(
                "t2ramsey",
                {"max_delay_us": number, "detune_ratio": number, "points": 3},
            ),
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["detune_ratio"]["value"] == 1.0
        assert type(fields["detune_ratio"]["value"]) is float
        sweep = fields["sweep.length"]["value"]
        assert sweep["stop"] == 1.0
        assert type(sweep["stop"]) is float
        assert sweep["expts"] == 3
        assert type(sweep["expts"]) is int
        detune_edits = [
            edit["value"]
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
            if edit["path"] == ["detune_ratio"]
        ]
        assert detune_edits == [1.0]
        assert type(detune_edits[0]) is float


def test_t1_requires_calibrated_pi_instead_of_custom_template(tmp_path):
    gui = CoherenceGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("t1", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"pi_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_t1_selected_library_pulse_preserves_gui_delay_defaults(tmp_path, reuse_tab_id):
    gui = CoherenceGui(pi_ref="pi")
    with recipe_client(tmp_path, gui) as client:
        data = full_execution_reply(
            client, client.call("t1", {"reuse_tab_id": reuse_tab_id})
        ).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.pi_pulse"] == {"value": "pi", "source": "gui_default"}
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 60.0,
            "expts": 61,
        }
        assert fields["sweep.length"]["source"]["stop"] == "gui_default"
        assert fields["modules.readout.pulse_cfg.freq"]["input"]["raw"] == "r_f"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.reset_cfg") == (1 if reuse_tab_id else 0)
        assert methods.count("tab.new") == (0 if reuse_tab_id else 1)
        assert methods.count("tab.run_start") == 1


@pytest.mark.parametrize("arguments", [{}, {"pi_ref": "pi"}])
def test_t1_reports_all_missing_calibration_sources(tmp_path, arguments):
    gui = CoherenceGui()
    gui.md = {}
    with recipe_client(tmp_path, gui) as client:
        data = client.call("t1", arguments).data
        assert data["status"] == "needs_parameters", data
        missing = {item["parameter"] for item in data["missing"]}
        assert missing == (
            {"readout_ref", "pi_ref"} if not arguments else {"readout_ref"}
        )
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_delay_us": True},
        {"max_delay_us": float("nan")},
        {"max_delay_us": float("inf")},
        {"max_delay_us": 0},
        {"max_delay_us": -1},
        {"points": 1},
        {"points": 2.5},
        {"points": 1.0},
        {"reps": 1.0},
        {"rounds": 1.0},
        {"points": True},
        {"reps": False},
        {"rounds": 1.5},
        {"pi_ref": " "},
        {"readout_ref": 7},
        {"use_reset": False},
        {"reuse_tab_id": ""},
    ],
)
def test_t1_rejects_invalid_inputs_before_preparing(tmp_path, arguments):
    gui = CoherenceGui(pi_ref="pi")
    client = make_client(tmp_path, gui)
    try:
        data = client.call("t1", arguments).data
        assert data["status"] == "failed", data
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)
        assert not gui.ran
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )
    finally:
        client.context.session.close()


@pytest.mark.parametrize("delay", [1, 1.0])
def test_t1_number_inputs_publish_floats_and_keep_integer_counts(tmp_path, delay):
    with recipe_client(tmp_path, CoherenceGui(pi_ref="pi")) as client:
        data = full_execution_reply(
            client,
            client.call(
                "t1", {"max_delay_us": delay, "points": 3, "reps": 2, "rounds": 1}
            ),
        ).data
        assert data["status"] == "finished", data
        sweep = data["actual"]["fields"]["sweep.length"]["value"]
        assert sweep["stop"] == 1.0
        assert type(sweep["stop"]) is float
        assert sweep["expts"] == 3
        assert type(sweep["expts"]) is int
        for name, count in (("reps", 2), ("rounds", 1)):
            value = data["actual"]["fields"][name]["value"]
            assert value == count
            assert type(value) is int
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
            if edit["path"] == ["sweep", "length"]
        ]
        assert len(edits) == 1
        assert edits[0]["value"]["stop"] == 1.0
        assert type(edits[0]["value"]["stop"]) is float


def test_t1_runs_once_with_calibrated_pi_and_explicit_delay(tmp_path):
    gui = CoherenceGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.get_analyze_result":
            reply["summary"] = {
                "t1": 42.0,
                "t1_err": None,
                "warnings": ["singular error"],
            }
            reply["invalid"] = [{"path": "summary.t1_err", "reason": "non_finite"}]
        return reply

    client = make_client(tmp_path, respond)
    try:
        reply = full_execution_reply(
            client,
            client.call(
                "t1",
                {
                    "pi_ref": "pi",
                    "max_delay_us": 80.0,
                    "points": 81,
                    "reps": 13,
                    "rounds": 9,
                },
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.pi_pulse"]["value"] == "pi"
        assert fields["modules.reset"]["value"] is None
        assert fields["modules.reset"]["source"] == "disabled"
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 80.0,
            "expts": 81,
        }
        assert fields["reps"]["value"] == 13
        assert fields["rounds"]["value"] == 9
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5100.0
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        assert len(client.transport.sent) == before
        assert summary["analysis"]["primary"] == {
            "params": {"threshold": 0.5},
            "estimates": {
                "t1": {"value": 42.0, "stderr": None, "unit": "us", "quality": None}
            },
            "details": {},
            "warnings": ["singular error"],
        }
        assert summary["invalid"] == [
            {"path": "analysis.primary.estimates.t1.stderr", "reason": "non_finite"}
        ]
        assert data["analysis"]["result"]["summary"]["t1"] == 42.0
        assert summary["actual"]["parameters"]["delay"]["value"] == {
            "start": 0.04,
            "stop": 80.0,
            "expts": 81,
        }
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
    finally:
        client.context.session.close()
