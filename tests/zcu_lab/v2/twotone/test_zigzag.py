"""Zig-zag GUI adapters lower their forms to the core cfgs and reuse its analysis."""

from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.cfg_model import CfgValidationContext, PulseSpec
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_resolved_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.definitions import ADAPTERS
from zcu_lab.v2.twotone.zigzag.core import ZigZagCfg
from zcu_lab.v2.twotone.zigzag.gui import ZigZagAdapter
from zcu_lab.v2.twotone.zigzag_sweep.core import (
    ZigZagScanCfg,
    ZigZagScanResult,
)
from zcu_lab.v2.twotone.zigzag_sweep.gui import (
    ZigZagScanAnalyzeParams,
    ZigZagScanFreqAdapter,
    ZigZagScanGainAdapter,
)


def _lowered_cfg(
    adapter_type: type[ZigZagAdapter | ZigZagScanGainAdapter | ZigZagScanFreqAdapter],
) -> ZigZagCfg | ZigZagScanCfg:
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
                    "module": {"x180": "Q1.pulses.cal"},
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
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "cal": PulseSpec.model_validate(
                            {
                                "type": "pulse",
                                "waveform": {"style": "const", "length": 0.05},
                                "ch": 3,
                                "nqz": 2,
                                "freq": 4000.0,
                                "gain": 0.5,
                            }
                        )
                    }
                )
            },
        )
    )
    parameters = ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_id="entry", point="working"),
        roles=None,
    ).bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )
    host = MagicMock()
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    schema = adapter_type.cfg_definition().instantiate(
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
    return adapter_type().build_exp_cfg(raw, request)


def test_zigzag_adapters_are_registered() -> None:
    assert ADAPTERS["twotone/zigzag"] is ZigZagAdapter
    assert ADAPTERS["twotone/zigzag_scan/gain"] is ZigZagScanGainAdapter
    assert ADAPTERS["twotone/zigzag_scan/freq"] is ZigZagScanFreqAdapter


def test_zigzag_default_form_lowers_to_core_cfg() -> None:
    cfg = _lowered_cfg(ZigZagAdapter)

    assert isinstance(cfg, ZigZagCfg)
    assert cfg.n_times == 10
    assert cfg.repeat_on == "X180_pulse"
    assert cfg.modules.X180_pulse is not None
    assert cfg.relax_delay == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("adapter_type", "swept", "unset", "start", "stop"),
    [
        (ZigZagScanGainAdapter, "gain", "freq", 0.4, 0.6),
        (ZigZagScanFreqAdapter, "freq", "gain", 3998.0, 4002.0),
    ],
)
def test_zigzag_scan_default_form_sets_exactly_one_sweep(
    adapter_type: type[ZigZagScanGainAdapter | ZigZagScanFreqAdapter],
    swept: Literal["gain", "freq"],
    unset: Literal["gain", "freq"],
    start: float,
    stop: float,
) -> None:
    cfg = _lowered_cfg(adapter_type)

    assert isinstance(cfg, ZigZagScanCfg)
    sweep = getattr(cfg.sweep, swept)
    assert sweep is not None
    assert (sweep.start, sweep.stop, sweep.expts) == pytest.approx((start, stop, 101))
    assert getattr(cfg.sweep, unset) is None
    assert cfg.n_times == 6


def _scan_record(best: float) -> RunRecord[ZigZagScanCfg, ZigZagScanResult]:
    times = np.arange(7)
    values = np.linspace(0.4, 0.6, 41)
    # Error grows with repetitions in proportion to the distance from ``best``.
    signals = np.cos(np.outer(times, values - best) * 20.0).astype(np.complex128)
    return RunRecord(cfg=None, result=ZigZagScanResult(times, values, signals))


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (ZigZagScanAnalyzeParams(), 0.5),
        (ZigZagScanAnalyzeParams(find_min=0.55), 0.55),
    ],
)
def test_zigzag_scan_analysis_reports_flattest_sweep_value(
    params: ZigZagScanAnalyzeParams, expected: float
) -> None:
    plots = Plots(NonPresentingHost())
    try:
        answer = ZigZagScanGainAdapter().analyze(
            AnalyzeRequest(
                _scan_record(0.5),
                params,
                parameters=None,
                predictor=None,
            ),
            plots=plots,
        )
        assert answer.best_value == pytest.approx(expected, abs=0.006)
        assert tuple(plots) == ("fit",)
    finally:
        plots.finish(present=False)
        plots.release()
