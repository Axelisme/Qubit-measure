"""Typed author carriers at the recording GUI wire, not indexed owner traversal.

Real indexed logical installation is owned by the later owner integration. The
original owner scenario remains in commit 393b89c71 for that acceptance seam.
"""

from copy import deepcopy
from pathlib import Path

from typing import Any

import pytest
from pydantic import JsonValue
from zcu_tools.gui.app.measure.remote.form_description import recipe_field_wire
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeFieldEdit, RecipeSession

from ._recipe_support import LookbackGui, recipe_client


class CarrierGui(LookbackGui):
    """Return prescribed wire facts; never claim to install indexed Form input."""

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.read_module":
            return {"value": {"gain": 0.5}, "mode": "inline", "source": "0"}
        if method == "tab.edit_recipe_fields":
            return {"changed": True, "errors": []}
        if method == "tab.read_recipe_field":
            reply = deepcopy(recipe_field_wire(self.owner.read_recipe_field(("rounds",))))
            path = params["path"]
            reply["description"]["path"] = path
            target = reply["description"]["target"]
            assert target is not None
            target["path"] = path
            value: JsonValue = [3, 4] if path[-1] == "samples" else 0.75
            reply.update(input=value, value=value)
            return dict(reply)
        return super().__call__(method, params)


def test_read_module_forwards_str_segments_and_rejects_unsupported_indexes(tmp_path: Path) -> None:
    with recipe_client(tmp_path, CarrierGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        observed = tab.read_module(("0",))
        assert observed.value == {"gain": 0.5}
        assert client.transport.sent[before:] == [("tab.read_module", {"tab_id": "t", "field": ["0"]})]
        before = len(client.transport.sent)
        with pytest.raises(ValueError, match="str|index"):
            tab.read_module(("tones", 1))
        assert client.transport.sent[before:] == []


def test_indexed_batch_read_and_run_capture_keep_original_position(tmp_path: Path) -> None:
    with recipe_client(tmp_path, CarrierGui()) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(SummaryParameter("q2_gain", ("tones", 1, "drive", "gain")),),
        ).open_tab("lookback")
        before = len(client.transport.sent)
        try:
            tab.set_fields(
                (
                    RecipeFieldEdit(("tones", 0, "drive", "gain"), None),
                    RecipeFieldEdit(("tones", 1, "drive", "gain"), {"expr": "0.75"}, "derived"),
                )
            )
            previous = tab.read_field(("tones", 1, "drive", "samples"))
            tab.set(("tones", 1, "drive", "samples"), [7, 8])
        except (AttributeError, TypeError, ValueError) as error:
            pytest.fail(f"Typed str/int author paths must reach the wire unchanged: {error}")
        assert client.transport.sent[before:] == [
            ("tab.edit_recipe_fields", {"tab_id": "t", "edits": [{"path": ["tones", 1, "drive", "gain"], "value": {"expr": "0.75"}}]}),
            ("tab.read_recipe_field", {"tab_id": "t", "path": ["tones", 1, "drive", "samples"]}),
            ("tab.edit_recipe_fields", {"tab_id": "t", "edits": [{"path": ["tones", 1, "drive", "samples"], "value": [7, 8]}]}),
        ]
        assert isinstance(previous.value, list)
        previous.value.append(99)
        assert tab.read_field(("tones", 1, "drive", "samples")).value == [3, 4]
        captured = tab.run().snapshot()
        assert captured is not None
        assert captured.actual["fields"] == {"q2_gain": {"value": 0.75, "source": "derived"}}
        tab.set(("tones", 1, "drive", "gain"), 0.9)
        assert captured.actual["fields"]["q2_gain"]["value"] == 0.75
