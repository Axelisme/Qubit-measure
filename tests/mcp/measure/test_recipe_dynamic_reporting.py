"""Variable fixed-list facts through public recipe capture and execution projection."""

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.app.measure.remote.form_observation import observe_elements
from zcu_tools.mcp.measure import execution_reply as reporting
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession

from ._recipe_element_support import (
    INPUTS,
    RECIPE,
    ElementGui,
    Group,
    Override,
    element_client,
)
from ._recipe_support import ExecutionGui


class DynamicGui(ElementGui):
    """Native analysis rows follow participants, not complete Form list order."""

    def __call__(self, method, params):
        if method == "tab.read_elements":
            return dict(observe_elements(
                self.owner, tuple(params["path"]),
                fields=tuple(tuple(path) for path in params.get("fields", [])),
            ))
        if method == "tab.run_start":
            self.accepted = Group.model_validate(self.owner.accept().cfg)
            self.native_summary = {
                "tones": [
                    {"fit": 4000.0 + index, "stderr": 0.2,
                     "element": ElementIdentity(index=index, name=self.names[index])}
                    for index in self.accepted.active_indices
                ] + [{"fit": None}, {"element": ElementIdentity(index=0, name="Q1")}]
            }
        return ExecutionGui.__call__(self, method, params)


def dynamic_author(
    session: RecipeSession, *, participants: list[str], overrides: list[Override]
) -> RecipeGenerator:
    tab = session.open_tab("lookback")
    tab.set(("participants",), participants)
    for edit in overrides:
        tab.set(tuple(edit["path"]), edit["value"], source="derived")
    # An explicit override must win over the declaration's default source.
    tab.set(("tones", 0, "drive", "gain"), 0.4)
    result = yield from tab.run_step()
    if result.run is not None:
        result.run.save_raw()
        yield from result.run.analyze_step("primary")


@pytest.mark.parametrize("count,participants", [(3, ["Q3", "Q1"]), (1, ["Q1"]), (4, ["Q4", "Q1"])])
def test_variable_elements_capture_all_leaves_and_project_native_row_identity(
    tmp_path: Path, count: int, participants: list[str]
) -> None:
    assert hasattr(reporting, "SummaryElements"), "Missing variable-list actual declaration"
    assert hasattr(reporting, "SummaryEstimateRows"), "Missing native-row estimate declaration"
    recipe = replace(
        RECIPE, run=dynamic_author, input_schema=INPUTS,
        summary_parameters=(reporting.SummaryElements(
            "tones", ("tones",),
            (reporting.SummaryParameter("gain", ("drive", "gain")),
             reporting.SummaryParameter("start", ("sweep", "start"), "MHz")),
            source="default",
        ),),
        summary_estimates=(reporting.SummaryEstimateRows(
            "tones", ("tones",),
            (reporting.SummaryEstimate("freq", ("fit",), ("stderr",), "MHz", element_key=("element",)),),
        ),),
    )
    gui = DynamicGui()
    gui.names = [f"Q{index + 1}" for index in range(count)]
    with element_client(tmp_path, gui, recipe) as client:
        reply = client.call("elements", {
            "participants": participants,
            "overrides": [] if count == 1 else [{"path": ["tones", count - 1, "drive", "gain"], "value": 0.6}],
        }).data
        assert reply["status"] == "finished", reply
        expected_actual = {}
        for index in range(count):
            expected_actual[f"tones/{index}/gain"] = {
                "value": 0.4 if index == 0 else 0.6 if index == count - 1 else 0.1,
                "source": "explicit" if index == 0 else "derived" if index == count - 1 else "default",
                "element": {"index": index, "name": f"Q{index + 1}"},
            }
            expected_actual[f"tones/{index}/start"] = {
                "value": 1.0, "source": "default", "unit": "MHz",
                "element": {"index": index, "name": f"Q{index + 1}"},
            }
        assert reply["actual"] == expected_actual
        expected_estimates = {
            f"tones/{row}/freq": {
                "value": 4000.0 + int(name[1:]) - 1, "stderr": 0.2, "unit": "MHz",
                "element": {"index": int(name[1:]) - 1, "name": name},
            } for row, name in enumerate(participants)
        }
        expected_estimates[f"tones/{len(participants)}/freq"] = {"value": None, "stderr": None, "unit": "MHz"}
        assert reply["analysis"]["primary"]["estimates"] == expected_estimates
        aggregates = [params for method, params in client.transport.sent if method == "tab.read_elements"]
        assert aggregates == [{"tab_id": "t", "path": ["tones"], "fields": [["drive", "gain"], ["sweep", "start"]]}]
        # Capture is detached: all subsequent views must avoid live GUI reads.
        gui.names = ["later"]
        gui.native_summary.clear()
        gui.md["$qubit.freq"] = 9000.0
        gui.library["later"] = {"gain": 0.9}
        gui.reset_form({"qubit": "Q2"})
        before = list(client.transport.sent)
        for detail in ("summary", "full"):
            retained = client.call("status", {"execution": reply["execution"], "detail": detail})
            assert retained["actual"] == expected_actual
            assert retained["analysis"]["primary"]["estimates"] == expected_estimates
        waited = client.call("wait", {"execution": reply["execution"], "timeout": 0}).data
        assert waited["actual"] == expected_actual
        assert waited["analysis"]["primary"]["estimates"] == expected_estimates
        assert client.transport.sent == before
