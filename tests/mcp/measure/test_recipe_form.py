"""Recipe author fields consume the real generated Form wire contract."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConfigBase, ConstWaveformCfg, PulseCfg, PulseSpec, Slot
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.remote.form_observation import observe_form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
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
                "pulse": Slot("qubit", "pulse"),
                "locked_pulse": Slot("qubit", "pulse", lock={"freq": 100.0}),
            },
        )
        self.owner = TabFormResource(
            form,
            capture=lambda aliases: TabFormSources(FormEnv(library=library)),
            mutation_allowed=lambda: True,
        )
        self.owner.write({("locked_pulse",): {}})

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.reset_cfg":
            return dict(observe_form(self.owner))
        if method == "tab.edit_cfg":
            self.owner.write(
                {tuple(edit["path"]): edit["value"] for edit in params["edits"]}
            )
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
        assert gui.owner.accept().values["pulse"]["freq"] == 100.0


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
