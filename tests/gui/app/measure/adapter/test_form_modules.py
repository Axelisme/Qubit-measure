"""Core module fields accept family-limited references and persistent inline Specs."""

from typing import Annotated

import pytest
from pydantic import JsonValue, PydanticSchemaGenerationError
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseSpec
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleSection(ConfigBase):
    """The core declares a specific pulse family, not GUI reference modes."""

    probe: Annotated[PulseCfg, FieldMeta(label="Probe", description="Drive the qubit.")]


class ModuleCore(ConfigBase):
    """A synthetic experiment cfg is independent of real experiment modules."""

    modules: ModuleSection


def module_env() -> FormEnv:
    """A fixed library contains a usable persistent pulse."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "probe": PulseSpec.model_validate(
                                {
                                    "ch": 1,
                                    "nqz": 1,
                                    "freq": 100,
                                    "gain": 0.3,
                                    "waveform": {"style": "const", "length": 0.1},
                                }
                            )
                        }
                    )
                },
            )
        )
    )


@pytest.mark.parametrize("mode", ["ref", "inline"])
def test_module_reference_or_inline_materializes_to_native_core(mode: str) -> None:
    try:
        form = Form(ModuleCore, {})
    except PydanticSchemaGenerationError as exc:
        pytest.fail(
            f"Form must derive editable module input instead of native QickParam fields: {exc}"
        )
    env = module_env()
    module: JsonValue = (
        {"kind": "ref", "name": "Q.pulses.probe"}
        if mode == "ref"
        else {
            "kind": "inline",
            "module": {
                "type": "pulse",
                "ch": 2,
                "nqz": 1,
                "freq": 200,
                "gain": 0.1,
                "waveform": {"style": "const", "length": 0.2},
            },
        }
    )
    state = FormState(FormInput(form.model, {"modules": {"probe": module}}))
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ModuleCore), cfg
    assert cfg.modules.probe.ch == (1 if mode == "ref" else 2)
    assert cfg.modules.probe.freq == (100 if mode == "ref" else 200)
    assert cfg.modules.probe.waveform.length == (0.1 if mode == "ref" else 0.2)
    assert form.model.input_schema(env)


def test_native_cfg_backfill_is_detached_inline_and_direct_range() -> None:
    from zcu_tools.cfg_model import Expr, Hidden
    from zcu_tools.program.v2.sweep import SweepCfg

    class LoadedCore(ModuleCore):
        sweep: SweepCfg
        amplitude: complex = 1j
        hidden: float = 7.0

    form = Form(LoadedCore, {"amplitude": Expr("2j"), "hidden": Hidden()})
    native = LoadedCore(
        modules=ModuleSection(
            probe=PulseCfg.model_validate(
                {
                    "ch": 2,
                    "nqz": 1,
                    "freq": 200,
                    "gain": 0.1,
                    "waveform": {"style": "const", "length": 0.2},
                }
            )
        ),
        sweep=SweepCfg(start=10, stop=14, expts=3, step=2),
        amplitude=2 + 3j,
    )
    assert hasattr(form, "cfg_input"), "Form must own native cfg backfill"
    backfill = form.cfg_input(native)
    data = backfill.data
    assert data["sweep"] == {"kind": "linrange", "start": 10.0, "stop": 14.0, "n": 3}
    assert data["amplitude"] == {"__complex__": [2.0, 3.0]}
    assert "hidden" not in data
    env = FormEnv()
    state = FormState(backfill)
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg == native
    native.modules.probe.gain = 0.7
    assert cfg.modules.probe.gain == 0.1
