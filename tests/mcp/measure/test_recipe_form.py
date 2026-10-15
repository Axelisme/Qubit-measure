"""Recipe author fields consume the real generated Form wire contract."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pytest
from pydantic import Field
from pydantic_core import PydanticCustomError
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.remote.form_observation import form_writes, observe_form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv, FormModel
from zcu_tools.gui.form_adapter import Form
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeSession
from zcu_tools.mcp.measure.recipe_form import read_form_field
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from ._recipe_support import LookbackGui, recipe_client


class AuthorCfg(ConfigBase):
    rounds: int = 3
    pulse: PulseCfg | None = None
    locked_pulse: PulseCfg | None = None


class FormGui(LookbackGui):
    """Record GUI requests; the real Form owner supplies all cfg observations."""

    def __init__(self) -> None:
        super().__init__()
        pulse = PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
            )
        )
        form = Form(
            AuthorCfg,
            {
                "pulse": Slot("$qubit.module.pulse"),
                "locked_pulse": Slot("Q1.pulses.pi", lock={"freq": 100.0}),
            },
        )
        self.owner = TabFormResource(
            lambda _ctx: form,
            capture=lambda aliases: TabFormSources(FormEnv(library=library)),
            mutation_allowed=lambda: True,
        )

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.reset_cfg":
            return dict(observe_form(self.owner))
        if method == "tab.edit_cfg":
            self.owner.write(form_writes(params["edits"]))
            return dict(observe_form(self.owner))
        if method == "tab.run_start":
            assert params == {"tab_id": "t"}
            self.ran = True
            return {"operation_id": 71}
        if method == "tab.read_module":
            return asdict(
                self.owner.read_module(tuple(str(params["field"]).split(".")))
            )
        if method == "tab.use_inline":
            path = tuple(str(params["field"]).split("."))
            self.owner.require_mutation_allowed()
            if self.owner.read_module(path).mode == "key":
                self.owner.run_command(path, "to_inline")
            return dict(observe_form(self.owner))
        return super().__call__(method, params)


def test_read_field_reports_form_input_values_and_directed_errors(
    tmp_path: Path,
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            before = tab.read_field("rounds")
        except (KeyError, ValueError) as error:
            raise AssertionError("read_field must consume Form observations") from error
        assert before.input == 3
        assert before.value == 3
        assert before.mode is None
        assert before.errors == ()
        with pytest.raises(GuiRpcError, match="Invalid"):
            tab.set("rounds", {"expr": "1 / 0"})
        invalid = tab.read_field("rounds")
        assert invalid.input == {"expr": "1 / 0"}
        assert invalid.value is None
        assert invalid.errors[0].path == ("rounds",)
        assert invalid.errors[0].message
        assert before.input == 3


def test_use_library_sets_actual_key_and_exposes_key_mode(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            tab.use_library("pulse", "Q1.pulses.pi")
        except (KeyError, ValueError) as error:
            raise AssertionError("use_library must consume Form Slot schema") from error
        selected = tab.read_field("pulse")
        assert selected.mode == "key"
        assert selected.key == "Q1.pulses.pi"
        assert selected.input == {"kind": "ref", "name": "Q1.pulses.pi"}
        assert not selected.locked
        assert selected.fields == ()
        assert selected.errors == ()
        values = gui.owner.accept().values["pulse"]
        if not isinstance(values, dict):
            raise TypeError("Expected pulse cfg values")
        assert values["freq"] == 100.0


def test_use_library_writes_one_declared_option_value(tmp_path: Path) -> None:
    with recipe_client(tmp_path, FormGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        tab.use_library("pulse", "Q1.pulses.pi")
        assert client.transport.sent[before:] == [
            (
                "tab.edit_cfg",
                {
                    "tab_id": "t",
                    "edits": [
                        {
                            "path": ["pulse"],
                            "value": {"kind": "ref", "name": "Q1.pulses.pi"},
                        }
                    ],
                },
            )
        ]


@pytest.mark.parametrize("captured", [False, True])
def test_unoffered_library_key_reports_available_options_without_writing(
    tmp_path: Path, captured: bool
) -> None:
    gui = FormGui()
    if not captured:
        gui.owner = TabFormResource(
            lambda _ctx: Form(AuthorCfg, {}),
            capture=lambda aliases: TabFormSources(FormEnv()),
            mutation_allowed=lambda: True,
        )
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        assert not tab.read_field("pulse").locked
        before = gui.owner.input.data
        sent = list(client.transport.sent)
        with pytest.raises((GuiRpcError, PydanticCustomError)) as failure:
            tab.use_library("pulse", "Q1.pulses.absent")
        error = failure.value
        assert isinstance(error, GuiRpcError), error
        assert error.reason == "invalid_cfg"
        assert "pulse" in str(error)
        assert "Available keys" in str(error)
        if captured:
            assert "Q1.pulses.pi" in str(error)
        assert client.transport.sent == sent
        assert gui.owner.input.data == before


def test_use_library_switches_active_inline_slot_to_actual_key(tmp_path: Path) -> None:
    gui = FormGui()
    library = gui.owner.env.library
    if library is None:
        raise TypeError("Expected library")
    gui.owner.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    gui.owner.run_command(("pulse",), "to_inline")
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        assert tab.read_field("pulse").mode == "inline"
        try:
            tab.use_library("pulse", "Q1.pulses.pi")
        except ValueError as error:
            raise AssertionError(
                "use_library must switch an active inline Slot"
            ) from error
        assert tab.read_field("pulse").key == "Q1.pulses.pi"
        cfg = AuthorCfg.model_validate(gui.owner.accept().cfg)
        if cfg.pulse is None:
            raise TypeError("Expected accepted pulse")
        assert cfg.pulse.freq == 100.0


class NumericOption(FormModel):
    kind: Literal[1, 2] = 1
    amount: float = 7.0


class StringOption(FormModel):
    kind: Literal["1"] = "1"
    text: str = "sample"


class TypedOptionsForm(FormModel):
    choice: NumericOption | StringOption = Field(discriminator="kind")


@pytest.mark.parametrize(
    "tag, fields", [(1, ("amount",)), (2, ("amount",)), ("1", ("text",))]
)
def test_read_field_projects_selected_typed_option_fields(
    tag: int | str, fields: tuple[str, ...]
) -> None:
    publication: dict[str, object] = {
        "schema": TypedOptionsForm.input_schema(FormEnv()),
        "input": {"choice": {"kind": tag}},
        "values": [],
        "errors": [],
    }
    observed = read_form_field(publication, ("choice",))
    assert observed.fields == fields
    assert observed.input == {"kind": tag}


class RequiredCfg(ConfigBase):
    pulse: PulseCfg


def test_use_library_switches_required_inline_slot_to_actual_key(
    tmp_path: Path,
) -> None:
    gui = FormGui()
    env = gui.owner.env
    form = Form(RequiredCfg, {"pulse": Template("pulse")})
    form.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Pulse",
                {
                    "ch": 0,
                    "nqz": 1,
                    "freq": 100.0,
                    "gain": 0.2,
                    "waveform.style": "const",
                    "waveform.length": 0.1,
                },
            )
        },
        adapter="lookback",
    )
    gui.owner = TabFormResource(
        lambda _ctx: form,
        capture=lambda aliases: TabFormSources(env),
        mutation_allowed=lambda: True,
    )
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        assert tab.read_field("pulse").mode == "inline"
        try:
            tab.use_library("pulse", "Q1.pulses.pi")
        except ValueError as error:
            raise AssertionError(
                "use_library must switch a required inline Slot"
            ) from error
        assert tab.read_field("pulse").key == "Q1.pulses.pi"
        assert RequiredCfg.model_validate(gui.owner.accept().cfg).pulse.freq == 100.0


def test_locked_inline_slot_rejects_library_selection_before_rpc(
    tmp_path: Path,
) -> None:
    with recipe_client(tmp_path, FormGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            locked = tab.read_field("locked_pulse")
        except (KeyError, ValueError) as error:
            raise AssertionError(
                "read_field must expose inline-only Form lock"
            ) from error
        assert locked.mode == "inline"
        assert locked.key is None
        assert locked.locked
        assert locked.fields == ("module",)
        before = list(client.transport.sent)
        with pytest.raises(ValueError, match="locked"):
            tab.use_library("locked_pulse", "Q1.pulses.pi")
        assert client.transport.sent == before


@pytest.mark.parametrize("field", ["pulse.kind", "locked_pulse.kind"])
def test_read_field_does_not_expose_option_selected_tag_leaves(
    tmp_path: Path, field: str
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library("pulse", "Q1.pulses.pi")
        before = gui.owner.input.data
        with pytest.raises(ValueError, match="Unknown cfg field"):
            tab.read_field(field)
        assert gui.owner.input.data == before
        assert tab.read_field("locked_pulse.module.freq").value == 100.0


def test_read_module_returns_lowered_values_without_editing_key(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library("pulse", "Q1.pulses.pi")
        before = gui.owner.input.data
        assert hasattr(tab, "read_module"), "Missing public execution module read"
        module = tab.read_module("pulse")
        assert module.mode == "key"
        assert module.source == "library:Q1.pulses.pi"
        assert module.value["freq"] == 100.0
        assert gui.owner.input.data == before
        module.value["freq"] = -1.0
        assert tab.read_module("pulse").value["freq"] == 100.0


@pytest.mark.parametrize("source", ["explicit", "calibration"])
def test_use_inline_copies_library_then_accepts_public_frequency_write(
    tmp_path: Path, source: str
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library("pulse", "Q1.pulses.pi")
        assert hasattr(tab, "use_inline"), "Missing public inline conversion"
        tab.use_inline("pulse")
        assert tab.read_field("pulse").mode == "inline"
        assert tab.read_field("pulse.module.gain").value == 0.2
        tab.set("pulse.module.freq", 125.0, source=source)
        module = tab.read_module("pulse")
        assert module.mode == "inline"
        assert module.source == "pulse"
        assert module.value["freq"] == 125.0
        assert AuthorCfg.model_validate(gui.owner.accept().cfg).pulse == PulseCfg(
            ch=0, nqz=1, freq=125.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        )


def test_retained_module_source_keeps_key_and_numeric_summary(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(
                SummaryParameter("frequency", "pulse.module.freq", "MHz"),
            ),
        ).open_tab("lookback")
        tab.use_library("pulse", "Q1.pulses.pi")
        before = gui.owner.input.data
        assert hasattr(tab, "retain_module_source"), "Missing module summary projection"
        tab.retain_module_source(
            "pulse", summary_field="pulse.module.freq", value_path=("freq",)
        )
        assert gui.owner.input.data == before
        assert tab.read_field("pulse").mode == "key"
        operation = tab.run()
        captured = operation.snapshot()
        assert captured is not None
        assert captured.actual["fields"]["pulse.module.freq"] == {
            "value": 100.0,
            "source": "library:Q1.pulses.pi",
            "unit": "MHz",
        }
        tab.use_inline("pulse")
        tab.set("pulse.module.freq", 125.0, source="calibration")
        assert captured.actual["fields"]["pulse.module.freq"]["value"] == 100.0
        next_capture = tab.run().snapshot()
        assert next_capture is not None
        assert next_capture.actual["fields"]["pulse.module.freq"] == {
            "value": 125.0,
            "source": "calibration",
            "unit": "MHz",
        }


def test_invalid_module_projection_records_no_success_prefix(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(SummaryParameter("frequency", "pulse.module.freq"),),
        ).open_tab("lookback")
        tab.use_library("pulse", "Q1.pulses.pi")
        assert hasattr(tab, "retain_module_source"), (
            "Missing validated summary projection"
        )
        with pytest.raises(ValueError, match="module value path"):
            tab.retain_module_source(
                "pulse", summary_field="pulse.module.freq", value_path=("missing",)
            )
        captured = tab.run().snapshot()
        assert captured is not None
        assert "pulse.module.freq" not in captured.actual["fields"]
