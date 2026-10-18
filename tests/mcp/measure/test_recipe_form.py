"""Recipe author fields consume the real generated Form wire contract."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pytest
from pydantic import Field
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.remote.form_description import description_wire
from zcu_tools.gui.app.measure.remote.form_observation import observe_form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import RecipeSession
from zcu_tools.mcp.measure.recipe_form import field_observation, read_form_field
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from tests.gui._form_converters import RebuildingForm

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
        if method in ("tab.reset_cfg", "tab.get_cfg"):
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
            before = tab.read_field(('rounds',))
        except (KeyError, ValueError) as error:
            raise AssertionError("read_field must consume Form observations") from error
        assert before.input == 3
        assert before.value == 3
        assert before.mode is None
        assert before.errors == ()
        with pytest.raises(GuiRpcError, match="Invalid"):
            tab.set(('rounds',), {"expr": "1 / 0"})
        invalid = tab.read_field(('rounds',))
        assert invalid.input == {"expr": "1 / 0"}
        assert invalid.value is None
        assert invalid.errors[0].path == ("rounds",)
        assert invalid.errors[0].message
        assert before.input == 3


def test_read_field_queries_current_owner_once_and_detaches_reply(
    tmp_path: Path,
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        gui.owner.write({("rounds",): 19})
        before = len(client.transport.sent)
        field = tab.read_field(('rounds',))
        assert field.input == 19
        assert field.value == 19
        assert client.transport.sent[before:] == [
            ("tab.read_recipe_field", {"tab_id": "t", "path": ["rounds"]})
        ]
        gui.owner.write({("rounds",): 23})
        assert field.input == 19
        assert tab.read_field(('rounds',)).value == 23


def test_use_library_sets_actual_key_and_exposes_key_mode(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            tab.use_library(('pulse',), "Q1.pulses.pi")
        except (KeyError, ValueError) as error:
            raise AssertionError("use_library must consume Form Slot schema") from error
        selected = tab.read_field(('pulse',))
        assert selected.mode == "key"
        assert selected.key == "Q1.pulses.pi"
        assert selected.input == {"kind": "ref", "name": "Q1.pulses.pi"}
        assert not selected.locked
        assert "freq" in selected.fields
        assert selected.errors == ()
        values = gui.owner.accept().values["pulse"]
        if not isinstance(values, dict):
            raise TypeError("Expected pulse cfg values")
        assert values["freq"] == 100.0


def test_use_library_sends_one_selector_command(tmp_path: Path) -> None:
    with recipe_client(tmp_path, FormGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        tab.use_library(('pulse',), "Q1.pulses.pi")
        assert client.transport.sent[before:] == [
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "t",
                    "path": ["pulse"],
                    "action": "use_library",
                    "key": "Q1.pulses.pi",
                },
            ),
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
        assert not tab.read_field(('pulse',)).locked
        before = gui.owner.input.data
        sent = list(client.transport.sent)
        with pytest.raises(GuiRpcError) as failure:
            tab.use_library(('pulse',), "Q1.pulses.absent")
        error = failure.value
        assert error.reason == "library_key"
        assert error.code == "invalid_params"
        assert error.data is not None
        assert error.data["path"] == ["pulse"]
        assert error.data["available_keys"] == (["Q1.pulses.pi"] if captured else [])
        assert client.transport.sent[len(sent) :] == [
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "t",
                    "path": ["pulse"],
                    "action": "use_library",
                    "key": "Q1.pulses.absent",
                },
            ),
        ]
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
        assert tab.read_field(('pulse',)).mode == "inline"
        try:
            tab.use_library(('pulse',), "Q1.pulses.pi")
        except ValueError as error:
            raise AssertionError(
                "use_library must switch an active inline Slot"
            ) from error
        assert tab.read_field(('pulse',)).key == "Q1.pulses.pi"
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
    state = FormState(FormInput(TypedOptionsForm, {"choice": {"kind": tag}}))
    observation = field_observation(
        {
            "description": description_wire(state.describe(("choice",), FormEnv())),
            "input": {"kind": tag},
            "input_present": True,
            "value": None,
            "errors": [],
            "value_status": "missing",
            "selector": None,
            "source": None,
            "readonly": False,
        }
    )
    observed = read_form_field(observation, ("choice",))
    assert observed.fields == fields
    assert observed.input == {"kind": tag}


class OuterInline(RebuildingForm):
    branch_models = {("module",): (NumericOption, StringOption)}

    kind: Literal["inline"] = "inline"
    module: NumericOption | StringOption = Field(discriminator="kind")


class OuterLockedExtras(FormModel):
    choice: OuterInline = Field(
        default_factory=lambda: OuterInline(module=NumericOption())
    )

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            ("choice",): (
                Option(
                    "Numeric",
                    {"kind": "inline", "module": {"kind": 1}},
                    fields=OuterInline,
                ),
                Option(
                    "String",
                    {"kind": "inline", "module": {"kind": "1"}},
                    fields=OuterInline,
                ),
            )
        }


def keep_author_cfg(extras: OuterLockedExtras, cfg: AuthorCfg) -> AuthorCfg:
    return cfg


def test_locked_outer_branch_keeps_inner_kind_editable(tmp_path: Path) -> None:
    gui = FormGui()
    gui.owner = TabFormResource(
        lambda _ctx: Form(
            AuthorCfg, {}, extras=OuterLockedExtras, lower=keep_author_cfg
        ),
        capture=lambda aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
    )
    gui.owner.describe(("extras", "choice"))
    assert gui.owner.preview().errors is None
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        outer = tab.read_field(('extras', 'choice'))
        assert outer.locked
        assert outer.fields == ("module",)
        assert not tab.read_field(('extras', 'choice', 'module')).locked
        with pytest.raises(GuiRpcError) as rejected:
            tab.use_library(('extras', 'choice'), "unoffered")
        assert rejected.value.reason == "module_selector"
        gui.owner.write(
            {("extras", "choice", "module"): {"kind": "1", "text": "changed"}}
        )
        assert tab.read_field(('extras', 'choice')).locked
        assert tab.read_field(('extras', 'choice', 'module')).fields == ("text",)
        assert tab.read_field(('extras', 'choice', 'module', 'text')).value == "changed"


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
        assert tab.read_field(('pulse',)).mode == "inline"
        try:
            tab.use_library(('pulse',), "Q1.pulses.pi")
        except ValueError as error:
            raise AssertionError(
                "use_library must switch a required inline Slot"
            ) from error
        assert tab.read_field(('pulse',)).key == "Q1.pulses.pi"
        assert RequiredCfg.model_validate(gui.owner.accept().cfg).pulse.freq == 100.0


def test_locked_inline_slot_rejects_library_selection_at_owner(
    tmp_path: Path,
) -> None:
    with recipe_client(tmp_path, FormGui()) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            locked = tab.read_field(('locked_pulse',))
        except (KeyError, ValueError) as error:
            raise AssertionError(
                "read_field must expose inline-only Form lock"
            ) from error
        assert locked.mode == "inline"
        assert locked.key is None
        assert locked.locked
        assert "freq" in locked.fields
        assert "gain" in locked.fields
        before = list(client.transport.sent)
        with pytest.raises(GuiRpcError) as rejected:
            tab.use_library(('locked_pulse',), "Q1.pulses.pi")
        assert rejected.value.reason == "module_selector"
        assert client.transport.sent[len(before) :] == [
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "t",
                    "path": ["locked_pulse"],
                    "action": "use_library",
                    "key": "Q1.pulses.pi",
                },
            ),
        ]


@pytest.mark.parametrize("field", ["pulse.kind", "locked_pulse.kind"])
def test_read_field_does_not_expose_option_selected_tag_leaves(
    tmp_path: Path, field: str
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        before = gui.owner.input.data
        with pytest.raises(GuiRpcError) as caught:
            tab.read_field(field)
        assert caught.value.code == "invalid_params"
        assert caught.value.reason == "form_path"
        assert caught.value.data == {"form_path": field.split(".")}
        assert gui.owner.input.data == before
        assert tab.read_field(('locked_pulse', 'freq')).value == 100.0


def test_read_module_returns_lowered_values_without_editing_key(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        before = gui.owner.input.data
        assert hasattr(tab, "read_module"), "Missing public execution module read"
        module = tab.read_module(('pulse',))
        assert module.mode == "key"
        assert module.source == "library:Q1.pulses.pi"
        assert module.value["freq"] == 100.0
        assert gui.owner.input.data == before
        module.value["freq"] = -1.0
        assert tab.read_module(('pulse',)).value["freq"] == 100.0


@pytest.mark.parametrize("source", ["explicit", "calibration"])
def test_use_inline_copies_library_then_accepts_public_frequency_write(
    tmp_path: Path, source: str
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        assert hasattr(tab, "use_inline"), "Missing public inline conversion"
        tab.use_inline(('pulse',))
        assert tab.read_field(('pulse',)).mode == "inline"
        assert tab.read_field(('pulse', 'gain')).value == 0.2
        tab.set(('pulse', 'freq'), 125.0, source=source)
        module = tab.read_module(('pulse',))
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
            summary_parameters=(SummaryParameter("frequency", ('pulse', 'freq'), "MHz"),),
        ).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        before = gui.owner.input.data
        assert hasattr(tab, "retain_module_source"), "Missing module summary projection"
        tab.retain_module_source(
            ('pulse',), summary_field=('pulse', 'freq'), value_path=("freq",)
        )
        assert gui.owner.input.data == before
        assert tab.read_field(('pulse',)).mode == "key"
        operation = tab.run()
        captured = operation.snapshot()
        assert captured is not None
        assert captured.actual["fields"]["pulse.freq"] == {
            "value": 100.0,
            "source": "library:Q1.pulses.pi",
            "unit": "MHz",
        }
        tab.use_inline(('pulse',))
        tab.set(('pulse', 'freq'), 125.0, source="calibration")
        assert captured.actual["fields"]["pulse.freq"]["value"] == 100.0
        next_capture = tab.run().snapshot()
        assert next_capture is not None
        assert next_capture.actual["fields"]["pulse.freq"] == {
            "value": 125.0,
            "source": "calibration",
            "unit": "MHz",
        }


def test_invalid_module_projection_records_no_success_prefix(tmp_path: Path) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(SummaryParameter("frequency", ('pulse', 'freq')),),
        ).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        assert hasattr(tab, "retain_module_source"), (
            "Missing validated summary projection"
        )
        with pytest.raises(ValueError, match="module value path"):
            tab.retain_module_source(
                ('pulse',), summary_field=('pulse', 'freq'), value_path=("missing",)
            )
        captured = tab.run().snapshot()
        assert captured is not None
        assert "pulse.freq" not in captured.actual["fields"]


def test_logical_ref_scalar_and_copy_keep_summary_facts_detached(
    tmp_path: Path,
) -> None:
    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(SummaryParameter("frequency", ('pulse', 'freq'), "MHz"),),
        ).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        before = gui.owner.input.data
        sent = len(client.transport.sent)
        observed = tab.read_field(('pulse', 'freq'))
        assert observed.value == 100.0
        assert observed.value_status == "available"
        assert observed.mode is None and observed.key is None
        assert observed.input_present is False and observed.input is None
        assert observed.readonly and not observed.locked
        assert observed.source is not None
        assert observed.source["key"] == "Q1.pulses.pi"
        assert client.transport.sent[sent:] == [
            ("tab.read_recipe_field", {"tab_id": "t", "path": ["pulse", "freq"]}),
        ]
        assert gui.owner.input.data == before
        with pytest.raises((GuiRpcError, ValueError)):
            tab.set(('pulse', 'freq'), 125.0)
        assert gui.owner.input.data == before
        tab.retain_source(('pulse', 'freq'), "library:Q1.pulses.pi")
        captured = tab.run().snapshot()
        assert captured is not None
        assert captured.actual["fields"]["pulse.freq"]["value"] == 100.0
        tab.use_inline(('pulse',))
        tab.set(('pulse', 'freq'), 125.0)
        assert captured.actual["fields"]["pulse.freq"]["value"] == 100.0
        assert observed.value == 100.0


def test_logical_scalar_batch_sends_one_owner_command_without_preflight_reads(
    tmp_path: Path,
) -> None:
    from zcu_tools.mcp.measure.recipe import RecipeFieldEdit

    gui = FormGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.use_library(('pulse',), "Q1.pulses.pi")
        tab.use_inline(('pulse',))
        sent = len(client.transport.sent)
        tab.set_fields(
            [RecipeFieldEdit(('rounds',), 9), RecipeFieldEdit(('pulse', 'freq'), 125.0)]
        )
        assert client.transport.sent[sent:] == [
            (
                "tab.edit_recipe_fields",
                {
                    "tab_id": "t",
                    "edits": [
                        {"path": ["rounds"], "value": 9},
                        {"path": ["pulse", "freq"], "value": 125.0},
                    ],
                },
            ),
        ]
        assert tab.read_field(('pulse', 'freq')).value == 125.0
