"""Synthetic fixed-list author using public inputs, Form owner and Run capture."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range
from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeExpression,
    RecipeFieldEdit,
    RecipeGenerator,
    RecipeInputSchema,
    RecipeSession,
    RecipeTab,
    ask,
)
from zcu_tools.program.v2 import SweepCfg

from ._recipe_support import ExecutionGui
from ._support import make_client


class Drive(ConfigBase):
    gain: float = Field(default=0.1, gt=0)


class Tone(ConfigBase):
    name: Annotated[str, FieldMeta(readonly=True)]
    drive: Drive = Drive()
    sweep: SweepCfg


class Group(ConfigBase):
    tones: Annotated[list[Tone], FieldMeta(element_label="name")]
    participants: list[str]

    @property
    def active_indices(self) -> list[int]:
        """Synthetic runtime preserves original positions in participant order."""
        names = [tone.name for tone in self.tones]
        return [names.index(name) for name in self.participants]


class Override(TypedDict):
    path: list[str | int]
    value: float | RecipeExpression


INPUTS: RecipeInputSchema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["participants", "overrides"],
    "properties": {
        "participants": {"type": "array", "items": {"type": "string"}},
        "overrides": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "value"],
                "properties": {
                    "path": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": ["string", "integer"]},
                    },
                    "value": {
                        "type": ["number", "object"],
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    },
}


def configure(
    session: RecipeSession, participants: list[str], overrides: list[Override]
) -> RecipeTab:
    """Install inputs through owner operations, never evaluate expressions here."""
    tab = session.open_tab("lookback")
    tab.set(("participants",), [name for name in participants])
    for index in range(3):
        tab.retain_source(("tones", index, "drive", "gain"), "default")
        tab.retain_source(("tones", index, "sweep", "start"), "default")
    tab.set(("tones", 2, "drive", "gain"), 0.6, source="derived")
    tab.set_fields(
        [RecipeFieldEdit(tuple(edit["path"]), edit["value"]) for edit in overrides]
    )
    return tab


def author(
    session: RecipeSession, *, participants: list[str], overrides: list[Override]
) -> RecipeGenerator:
    yield from configure(session, participants, overrides).run_step()


def analysis_author(
    session: RecipeSession, *, participants: list[str], overrides: list[Override]
) -> RecipeGenerator:
    result = yield from configure(session, participants, overrides).run_step()
    run = result.run
    if run is None:
        return
    run.save_raw()
    yield from run.analyze_step("primary")
    yield from ask(run.propose_writeback())


RECIPE = RecipeDefinition(
    "elements",
    "Run synthetic fixed elements without hardware",
    INPUTS,
    author,
    adapter_name="lookback",
    summary_parameters=tuple(
        parameter
        for index in range(3)
        for parameter in (
            SummaryParameter(f"gain{index}", ("tones", index, "drive", "gain")),
            SummaryParameter(f"start{index}", ("tones", index, "sweep", "start")),
        )
    ),
    summary_estimates=(),
)


ANALYSIS_RECIPE = replace(
    RECIPE,
    run=analysis_author,
    summary_estimates=(
        SummaryEstimate(
            "missing", ("tones", 0, "fit"), element_key=("tones", 0, "element")
        ),
        SummaryEstimate(
            "without_identity", ("tones", 1, "fit"), element_key=("tones", 1, "element")
        ),
        SummaryEstimate(
            "fit",
            ("tones", 2, "fit"),
            ("tones", 2, "stderr"),
            "us",
            element_key=("tones", 2, "element"),
        ),
    ),
)


class ElementGui(ExecutionGui):
    """Wire collaborator delegates edits/admission to the real Form owner."""

    def __init__(self) -> None:
        super().__init__()
        self.md["$qubit.freq"] = 4500.0
        self.names = ["Q1", "Q2", "Q3"]
        self.accepted: Group | None = None
        self.native_summary: dict[str, object] = {}
        self.candidate_element: ElementIdentity | None = None

    def make_form(self) -> Form[Group]:
        return Form(
            Group,
            {
                "tones": [
                    {
                        "name": name,
                        "drive": {"gain": 0.1},
                        "sweep": Range.between(1, 2, expts=3),
                    }
                    for name in self.names
                ],
                "participants": ["Q1"],
            },
            fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
        )

    def __call__(self, method, params):
        if method == "tab.run_start":
            self.accepted = Group.model_validate(self.owner.accept().cfg)
            element = ElementIdentity(index=2, name=self.accepted.tones[2].name)
            self.candidate_element = element.copy()
            self.native_summary = {
                "tones": [
                    {},
                    {"fit": 2.5},
                    {"fit": 7.0, "stderr": 0.2, "element": element},
                ]
            }
        return super().__call__(method, params)

    def _analysis(self, method, params):
        reply = super()._analysis(method, params)
        if method == "tab.get_analyze_result":
            reply["summary"] = deepcopy(self.native_summary)
        elif method == "tab.writeback_preview":
            reply["items"][0]["element"] = deepcopy(self.candidate_element)
        return reply


@contextmanager
def element_client(tmp_path: Path, gui: ElementGui, recipe: RecipeDefinition = RECIPE):
    client = make_client(tmp_path, gui, recipes=(recipe,))
    try:
        yield client
    finally:
        client.context.session.close()
