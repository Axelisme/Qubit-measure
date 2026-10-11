from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv

from typing import cast

from zcu_tools.cfg_model import module_spec_from_raw
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionValue,
    ReferenceValue,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.program.v2 import (
    ModuleCfgFactory,
    PulseReadoutCfg,
)
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument

from tests.gui.app.measure._cfg_fakes import parameter_capture
from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.v2._support.measure.defaults.role_table import ROLE_TABLE, role_ref
from zcu_lab.v2._support.measure.spec_helpers import schema_from_module


def test_schema_from_module_returns_none_for_none():
    assert schema_from_module(None) is None


def _pulse_readout(*, freq: float = 6000.0) -> PulseReadoutCfg:
    return cast(
        PulseReadoutCfg,
        ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": freq,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": freq,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
        ),
    )


def test_schema_from_module_converts_proposed_readout_faithfully():
    """edit_schema is the spec+value view of the proposed module (freq carried).

    The proposed module is built the way adapters now build it: a direct
    with_updates on the strongly-typed PulseReadoutCfg.
    """
    readout = _pulse_readout()
    proposed = readout.with_updates(
        pulse_cfg=readout.pulse_cfg.with_updates(freq=6150.0),
        ro_cfg=readout.ro_cfg.with_updates(ro_freq=6150.0),
    )

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    pulse_cfg = schema.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"].value == 6150.0  # type: ignore[union-attr]


def test_schema_from_module_converts_proposed_waveform_faithfully():
    """A proposed waveform cfg auto-routes through the waveform converter."""
    readout = _pulse_readout()
    proposed = readout.pulse_cfg.waveform.with_updates(length=4.5)

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    length = schema.value.fields["length"]
    assert length.value == 4.5  # type: ignore[union-attr]


def _make_ctx(library: LibrarySnapshot | None = None) -> SessionEnv:
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    readout = "R1.readout.readout_rf" if library is not None else None
    parameters = parameter_capture(
        {
            "Q1": QubitSchema(
                kind="qubit/transmon",
                module={} if readout is None else {"readout": readout},
            ),
            "R1": ResonatorSchema(kind="resonator"),
        },
        library,
    ).bind_aliases((AliasRow("qubit", "Q1"), AliasRow("resonator", "R1")), {})
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


def test_readout_role_ref_uses_library_selection_before_fallback():
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    ).replaced_module(
        "R1.readout.readout_rf",
        module_spec_from_raw(_pulse_readout(freq=6100.0).model_dump(mode="python")),
    )
    ctx = _make_ctx(library)
    module_ref = role_ref(ROLE_TABLE["readout"], ctx)

    assert isinstance(module_ref, ReferenceValue)
    assert module_ref.chosen_key == "$qubit.module.readout"


def test_readout_role_ref_falls_back_to_custom_when_lib_empty():
    ctx = _make_ctx()
    module_ref = role_ref(ROLE_TABLE["readout"], ctx)

    assert isinstance(module_ref, ReferenceValue)
    assert module_ref.chosen_key == "<Custom:Pulse Readout>"


def test_readout_role_ref_returns_disabled_when_optional_and_empty():
    # ADR-0010: optional ref with an empty library → None (disabled).
    ctx = _make_ctx()
    module_ref = role_ref(ROLE_TABLE["readout"], ctx, optional=True)

    assert module_ref is None
