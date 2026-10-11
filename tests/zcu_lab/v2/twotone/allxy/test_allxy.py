"""AllXY GUI adapter lowers its form to the core cfg and reuses its analysis."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.cfg_model import CfgValidationContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_resolved_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.definitions import ADAPTERS
from zcu_lab.v2.twotone.allxy.core import (
    ALLXY_SEQUENCE,
    AllXY_Result,
    AllXYCfg,
    predict_state_with_error,
)
from zcu_lab.v2.twotone.allxy.gui import (
    AllXYAdapter,
    AllXYAnalyzeParams,
)


def test_allxy_adapter_is_registered() -> None:
    assert ADAPTERS["twotone/allxy"] is AllXYAdapter


def test_allxy_default_form_lowers_to_core_cfg() -> None:
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
        components={
            "Q1": QubitSchema.model_validate(
                {
                    "kind": "qubit/transmon",
                    "freq": 4000.0,
                    "t1": 20.0,
                    "resonator": "R1",
                    "wiring": {"L01": {"ch": 3}},
                }
            ),
            "R1": ResonatorSchema.model_validate(
                {
                    "kind": "resonator",
                    "freq": 6200.0,
                    "wiring": {"ch": 2, "ro_ch": 2},
                    "ext": {"time_of_fly": 0.5},
                }
            ),
        },
    )
    parameters = ParameterContext(
        point=point,
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    ).bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )
    host = MagicMock()
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    schema = AllXYAdapter.cfg_definition().instantiate(
        SessionEnv(parameters=parameters, soc=None, soccfg=None)
    )
    draft = MeasureCfgBindings(host).new_draft(schema, parameters=parameters)
    try:
        raw = schema_to_resolved_dict(draft.snapshot())
    finally:
        draft.close()
    request = RunRequest(
        soc=MagicMock(),
        soccfg=MagicMock(),
        cfg_context=CfgValidationContext(modules=parameters.library),
    )

    cfg = AllXYAdapter().build_exp_cfg(raw, request)

    assert isinstance(cfg, AllXYCfg)
    assert cfg.modules.X90_pulse is not None
    assert cfg.modules.X180_pulse is not None
    assert cfg.modules.I_pulse is None
    assert cfg.relax_delay == pytest.approx(100.0)


@pytest.mark.parametrize("fit_ge", [False, True])
def test_allxy_analysis_reports_fitted_errors(fit_ge: bool) -> None:
    states = np.array(
        [predict_state_with_error(seq, 0.05, 0.0) for seq in ALLXY_SEQUENCE]
    )
    record: RunRecord[AllXYCfg, AllXY_Result] = RunRecord(
        cfg=None,
        result=AllXY_Result(
            gate_idxs=np.arange(len(ALLXY_SEQUENCE), dtype=np.int64),
            signals=(1.0 + 0.5 * states).astype(np.complex128),
        ),
    )
    plots = Plots(NonPresentingHost())
    try:
        answer = AllXYAdapter().analyze(
            AnalyzeRequest(
                record,
                AllXYAnalyzeParams(fit_ge=fit_ge),
                parameters=None,
                    ),
            plots=plots,
        )
        assert answer.amplitude_error == pytest.approx(0.05 / (np.pi / 2), abs=0.003)
        assert answer.detune_param == pytest.approx(0.0, abs=0.005)
        assert answer.power_err > 0.0
        assert answer.residual_rms < 0.01
        assert set(answer.to_summary_dict()) == {
            "amplitude_error",
            "detune_param",
            "power_err",
            "detune_err",
            "residual_rms",
        }
        assert tuple(plots) == ("fit",)
    finally:
        plots.finish(present=False)
        plots.release()
