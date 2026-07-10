from __future__ import annotations

import pytest
import zcu_tools.program.v2 as program_v2
from pydantic import ValidationError
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.v2.autofluxdep.qubit_freq import (
    QubitFreqCfg,
    QubitFreqCfgTemplate,
    QubitFreqModuleCfg,
)
from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiCfg as SingleshotLenRabiCfg,
)
from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiModuleCfg as SingleshotLenRabiModuleCfg,
)
from zcu_tools.experiment.v2.twotone.fluxdep import (
    FreqFluxCfg,
    FreqFluxModuleCfg,
)
from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqModuleCfg
from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg, PowerModuleCfg
from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiModuleCfg,
)
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import (
    LenRabiCfg,
    LenRabiModuleCfg,
)
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import (
    QubitFreqCfgTemplate as GuiQubitFreqCfgTemplate,
)
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import (
    QubitFreqModuleCfg as GuiQubitFreqModuleCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, ProgramV2Cfg, PulseCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg

MODULE_CFG_TYPES: tuple[type[ConfigBase], ...] = (
    FreqModuleCfg,
    FreqFluxModuleCfg,
    PowerModuleCfg,
    AmpRabiModuleCfg,
    LenRabiModuleCfg,
    SingleshotLenRabiModuleCfg,
    QubitFreqModuleCfg,
    GuiQubitFreqModuleCfg,
)

CFG_MODULE_TYPES: tuple[tuple[type[ConfigBase], type[ConfigBase]], ...] = (
    (FreqCfg, FreqModuleCfg),
    (FreqFluxCfg, FreqFluxModuleCfg),
    (PowerCfg, PowerModuleCfg),
    (AmpRabiCfg, AmpRabiModuleCfg),
    (LenRabiCfg, LenRabiModuleCfg),
    (SingleshotLenRabiCfg, SingleshotLenRabiModuleCfg),
    (QubitFreqCfgTemplate, QubitFreqModuleCfg),
    (QubitFreqCfg, QubitFreqModuleCfg),
    (GuiQubitFreqCfgTemplate, GuiQubitFreqModuleCfg),
)


def _pulse() -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=0.1),
        ch=0,
        nqz=1,
        freq=3000.0,
        gain=0.2,
    )


def _readout() -> DirectReadoutCfg:
    return DirectReadoutCfg(
        type="readout/direct",
        ro_ch=0,
        ro_length=1.0,
        ro_freq=6000.0,
        gen_ch=0,
    )


@pytest.mark.parametrize("module_cfg_type", MODULE_CFG_TYPES)
def test_concrete_module_cfg_owns_full_contract(
    module_cfg_type: type[ConfigBase],
) -> None:
    assert set(module_cfg_type.model_fields) == {
        "reset",
        "init_pulse",
        "qub_pulse",
        "readout",
    }
    assert module_cfg_type.model_fields["reset"].default is None
    assert module_cfg_type.model_fields["init_pulse"].default is None

    values = {
        "reset": None,
        "init_pulse": None,
        "qub_pulse": _pulse(),
        "readout": _readout(),
    }
    module_cfg_type.model_validate(values)

    with pytest.raises(ValidationError):
        module_cfg_type.model_validate({**values, "qub_pusle": _pulse()})


@pytest.mark.parametrize(("cfg_type", "module_cfg_type"), CFG_MODULE_TYPES)
def test_concrete_cfg_directly_declares_its_module_contract(
    cfg_type: type[ConfigBase],
    module_cfg_type: type[ConfigBase],
) -> None:
    assert issubclass(cfg_type, ProgramV2Cfg)
    assert cfg_type.model_fields["modules"].annotation is module_cfg_type
    assert cfg_type.model_fields["reps"].default == 1
    assert cfg_type.model_fields["rounds"].default == 1
    assert cfg_type.model_fields["initial_delay"].default == 1.0
    assert cfg_type.model_fields["relax_delay"].default == 1.0


@pytest.mark.parametrize("cfg_type", (FreqFluxCfg, QubitFreqCfg))
def test_flux_run_cfg_requires_devices(cfg_type: type[ConfigBase]) -> None:
    assert cfg_type.model_fields["dev"].is_required()


@pytest.mark.parametrize(
    "cfg_type",
    (FreqCfg, PowerCfg, AmpRabiCfg, LenRabiCfg, QubitFreqCfgTemplate),
)
def test_regular_cfg_keeps_optional_devices(cfg_type: type[ConfigBase]) -> None:
    assert not cfg_type.model_fields["dev"].is_required()
    assert cfg_type.model_fields["dev"].default is None


def test_program_v2_barrel_has_no_tone_specific_cfg_or_program() -> None:
    for name in ("OneToneCfg", "OneToneProgram", "TwoToneCfg", "TwoToneProgram"):
        assert not hasattr(program_v2, name)
