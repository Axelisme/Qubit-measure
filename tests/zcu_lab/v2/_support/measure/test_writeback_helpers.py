"""Calibration gates read canonical values from the supplied Run capture."""

from dataclasses import dataclass

import pytest
from zcu_tools.cfg_model import PulseResetSpec
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.modules import PulseResetCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.components import QubitSchema
from zcu_lab.v2._support.measure.writeback_helpers import reset_module_writeback_items


@dataclass(frozen=True)
class ResetModules:
    tested_reset: PulseResetCfg


@dataclass(frozen=True)
class ResetSnapshot:
    modules: ResetModules


@pytest.fixture
def reset_calibration() -> tuple[ParameterContext, ResetSnapshot]:
    raw = {
        "type": "reset/pulse",
        "pulse_cfg": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.5},
            "ch": 0,
            "nqz": 1,
            "freq": 4120.0,
            "gain": 0.3,
        },
    }
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
        components={
            "Q1": QubitSchema(
                kind="qubit/fluxonium", module={"sideband": "Q1.reset.cal"}
            )
        },
    )
    parameters = ParameterContext(
        point=point,
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
                components={
                    "Q1": ComponentLibrary(
                        reset={"cal": PulseResetSpec.model_validate(raw)}
                    )
                },
            )
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})
    template = PulseResetCfg.model_validate(raw).with_updates(
        pulse_cfg=PulseResetCfg.model_validate(raw).pulse_cfg.with_updates(freq=3000.0)
    )
    return parameters, ResetSnapshot(ResetModules(template))


def test_reset_proposal_gate_reads_canonical_library_calibration(
    reset_calibration,
) -> None:
    parameters, snapshot = reset_calibration
    items = reset_module_writeback_items(
        SessionEnv(parameters=parameters, soc=None, soccfg=None),
        snapshot,
        target="reset_10",
        desc="calibrated reset",
        field_md_map=[("pulse_cfg.freq", "$qubit.module.sideband.pulse_cfg.freq")],
    )
    assert len(items) == 1
    schema = items[0].edit_schema
    assert schema is not None
    proposal = PulseResetCfg.model_validate(schema_to_raw_dict(schema, parameters))
    assert proposal.pulse_cfg.freq == 4120.0
    assert snapshot.modules.tested_reset.pulse_cfg.freq == 3000.0


def test_reset_proposal_without_parameter_capture_does_not_read_current_globals(
    reset_calibration,
) -> None:
    _, snapshot = reset_calibration
    assert (
        reset_module_writeback_items(
            SessionEnv(parameters=None, soc=None, soccfg=None),
            snapshot,
            target="reset_10",
            desc="calibrated reset",
            field_md_map=[("pulse_cfg.freq", "$qubit.module.sideband.pulse_cfg.freq")],
        )
        == []
    )


def test_reset_proposal_waits_for_all_calibration_fields(reset_calibration) -> None:
    parameters, snapshot = reset_calibration
    assert (
        reset_module_writeback_items(
            SessionEnv(parameters=parameters, soc=None, soccfg=None),
            snapshot,
            target="reset_10",
            desc="calibrated reset",
            field_md_map=[("pulse_cfg.freq", "$qubit.ext.not_calibrated")],
        )
        == []
    )
