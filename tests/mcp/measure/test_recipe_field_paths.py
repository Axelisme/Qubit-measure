"""Typed author paths retain original list positions through the owner wire."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form_adapter import Form
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeFieldEdit, RecipeSession

from ._recipe_support import LookbackGui, recipe_client


class Drive(ConfigBase):
    gain: float
    samples: list[int]


class Tone(ConfigBase):
    name: str
    drive: Drive


class IndexedControls(ConfigBase):
    tones: list[Tone] = [
        Tone(name="Q1", drive=Drive(gain=0.1, samples=[1, 2])),
        Tone(name="Q2", drive=Drive(gain=0.2, samples=[3, 4])),
    ]


class IndexedGui(LookbackGui):
    def make_form(self):
        return Form(IndexedControls, {})


def test_indexed_batch_read_and_run_capture_keep_original_position(tmp_path: Path) -> None:
    gui = IndexedGui()
    with recipe_client(tmp_path, gui) as client:
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
            pytest.fail(f"Typed str/int author paths must reach the owner unchanged: {error}")
        assert client.transport.sent[before] == (
            "tab.edit_recipe_fields",
            {"tab_id": "t", "edits": [{"path": ["tones", 1, "drive", "gain"], "value": {"expr": "0.75"}}]},
        )
        assert previous.value == [3, 4]
        assert tab.read_field(("tones", 1, "drive", "samples")).value == [7, 8]
        assert tab.read_field(("tones", 0, "drive", "gain")).value == 0.1
        captured = tab.run().snapshot()
        assert captured is not None
        assert captured.actual["fields"] == {"q2_gain": {"value": 0.75, "source": "derived"}}
        tab.set(("tones", 1, "drive", "gain"), 0.9)
        assert captured.actual["fields"]["q2_gain"]["value"] == 0.75
