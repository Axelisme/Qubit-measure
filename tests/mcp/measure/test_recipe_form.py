"""Recipe author fields consume the real generated Form wire contract."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.remote.form_observation import form_writes, observe_form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv, ModuleInline
from zcu_tools.mcp.measure.recipe import RecipeSession
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
                format_version="1.0",
                components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
            )
        )
        form = Form(
            AuthorCfg,
            {
                "pulse": Slot("$qubit.module.pulse"),
                "locked_pulse": Slot("$qubit.module.pulse", lock={"freq": 100.0}),
            },
        )
        self.owner = TabFormResource(
            form,
            capture=lambda aliases: TabFormSources(FormEnv(library=library)),
            mutation_allowed=lambda: True,
        )
        self.owner.write(
            {("locked_pulse",): ModuleInline(module=pulse).model_dump(mode="json")}
        )

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.reset_cfg":
            return dict(observe_form(self.owner))
        if method == "tab.edit_cfg":
            self.owner.write(form_writes(params["edits"]))
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
        assert selected.errors == ()
        values = gui.owner.accept().values["pulse"]
        if not isinstance(values, dict):
            raise TypeError("Expected pulse cfg values")
        assert values["freq"] == 100.0


def test_use_library_switches_active_inline_slot_to_actual_key(tmp_path: Path) -> None:
    gui = FormGui()
    library = gui.owner.env.library
    if library is None:
        raise TypeError("Expected library")
    module = library.module_spec("Q1.pulses.pi")
    gui.owner.write({("pulse",): ModuleInline(module=module).model_dump(mode="json")})
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
        form, capture=lambda aliases: TabFormSources(env), mutation_allowed=lambda: True
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
        assert "module" in locked.fields
        before = list(client.transport.sent)
        with pytest.raises(ValueError, match="locked"):
            tab.use_library("locked_pulse", "Q1.pulses.pi")
        assert client.transport.sent == before
