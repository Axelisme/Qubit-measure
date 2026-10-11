"""Recipe adapters propose calibrations for the captured Run environment."""

from dataclasses import dataclass

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalysisMode,
    AnalyzeResult,
    PostWritebackRequest,
    WritebackRequest,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.types import SessionEnv

from tests.zcu_lab._writeback_support import WritebackSession
from tests.zcu_lab._writeback_support import writeback_registry as _registry
from zcu_lab.recipes import RECIPES
from zcu_lab.v2._support.measure.interactive_flux_pick import FluxPickResult
from zcu_lab.v2.lookback.core import LookbackResult
from zcu_lab.v2.lookback.gui import LookbackAnalyzeResult
from zcu_lab.v2.onetone.flux_dep.core import FluxDepResult
from zcu_lab.v2.onetone.freq.core import FreqCfg as OneToneCfg
from zcu_lab.v2.onetone.freq.core import FreqResult as OneToneResult
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAnalyzeResult
from zcu_lab.v2.singleshot.ge.core import GE_Result
from zcu_lab.v2.singleshot.ge.gui import GEAnalyzeResult, GEPostAnalyzeResult
from zcu_lab.v2.twotone.freq.core import FreqResult
from zcu_lab.v2.twotone.freq.gui import FreqAnalyzeResult
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiCfg, AmpRabiResult
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAnalyzeResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiResult
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAnalyzeResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Analysis, T1Result
from zcu_lab.v2.twotone.time_domain.t1.gui import T1AnalyzeResult
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAnalyzeResult
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg, T2RamseyResult
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAnalyzeResult


def _recipe_adapters() -> list[str]:
    registry = _registry()
    return [
        recipe.adapter_name
        for recipe in RECIPES
        if registry.create(recipe.adapter_name).capabilities.analysis
        != AnalysisMode.NONE
    ]


@dataclass(frozen=True)
class Calibration:
    source: object
    analysis: AnalyzeResult
    # Alias paths and independently specified fitted values for point writes.
    scalars: dict[str, object]
    # Library root -> one calibrated leaf and its value.
    modules: dict[str, tuple[str, float]]


def _calibration(name: str) -> Calibration:
    axis = np.array([0.0, 1.0])
    signals = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    pulse = {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 4200.0,
        "gain": 0.2,
        "waveform": {"style": "const", "length": 0.1},
    }
    readout = {
        "type": "readout/pulse",
        "pulse_cfg": pulse,
        "ro_cfg": {"ro_ch": 0, "ro_length": 1.0},
    }
    sweep = {"start": 0.0, "stop": 1.0, "step": 1.0, "expts": 2}
    match name:
        case "lookback":
            calibration = Calibration(
                RunRecord(None, LookbackResult(axis, signals)),
                LookbackAnalyzeResult(0.25),
                {"$resonator.ext.time_of_fly": 0.25},
                {},
            )
        case "onetone/freq":
            cfg = OneToneCfg.model_validate(
                {"modules": {"readout": readout}, "sweep": {"freq": sweep}}
            )
            calibration = Calibration(
                RunRecord(cfg, OneToneResult(axis, signals)),
                OneToneFreqAnalyzeResult(
                    6100.0, 5.0, {"theta0": 0.4}, edelay=0.012, edelay_persistable=True
                ),
                {
                    "$resonator.freq": 6100.0,
                    "$resonator.kappa": 5.0,
                    "$resonator.ext.theta0": 0.4,
                    "$resonator.ext.edelay": {"edelay": 0.012, "res_ch": 0, "ro_ch": 0},
                },
                {"$resonator.readout.readout_rf": ("pulse_cfg.freq", 6100.0)},
            )
        case "onetone/flux_dep":
            calibration = Calibration(
                RunRecord(
                    None,
                    FluxDepResult(axis, axis, np.zeros((2, 2), dtype=np.complex128)),
                ),
                FluxPickResult(0.5, 1.0, 1.0),
                {
                    "$qubit.flux_half": 0.5,
                    "$qubit.flux_int": 1.0,
                    "$qubit.flux_period": 1.0,
                },
                {},
            )
        case "singleshot/ge":
            analysis = GEAnalyzeResult(
                "ground", 0.95, 0.0, 0.5, 0.1, 1.0 + 2.0j, 3.0 + 4.0j, np.eye(2)
            )
            calibration = Calibration(
                RunRecord(
                    None,
                    GE_Result(
                        np.zeros((2, 2), dtype=np.complex128),
                        np.arange(2),
                        np.arange(2),
                    ),
                ),
                analysis,
                {
                    "$qubit.ext.readout_cal.fidelity": 0.95,
                    "$qubit.ext.readout_cal.sigma": 0.1,
                    "$qubit.ext.readout_cal.g_center": {"__complex__": [1.0, 2.0]},
                    "$qubit.ext.readout_cal.e_center": {"__complex__": [3.0, 4.0]},
                },
                {},
            )
        case "twotone/t1":
            calibration = Calibration(
                RunRecord(None, T1Result(axis, signals)),
                T1AnalyzeResult(T1Analysis(12.0, 0.2)),
                {"$qubit.t1": 12.0},
                {},
            )
        case "twotone/t2echo":
            calibration = Calibration(
                RunRecord(None, T2EchoResult(axis, signals)),
                T2EchoAnalyzeResult(15.0, 0.2),
                {"$qubit.t2e": 15.0},
                {},
            )
        case "twotone/t2ramsey":
            cfg = T2RamseyCfg.model_validate(
                {
                    "modules": {"pi2_pulse": pulse, "readout": readout},
                    "sweep": {"length": sweep},
                }
            )
            calibration = Calibration(
                RunRecord(cfg, T2RamseyResult(axis, signals, 0.5)),
                T2RamseyAnalyzeResult(10.0, 0.1, 0.25, True),
                {"$qubit.t2r": 10.0, "$qubit.freq": 4200.25},
                {},
            )
        case "twotone/rabi/amp_rabi":
            cfg = AmpRabiCfg.model_validate(
                {
                    "modules": {"qub_pulse": pulse, "readout": readout},
                    "sweep": {"gain": sweep},
                }
            )
            calibration = Calibration(
                RunRecord(cfg, AmpRabiResult(axis, signals)),
                AmpRabiAnalyzeResult(0.4, 0.01, 0.2, 0.01),
                {
                    "$qubit.module.x180": "Q1.pulses.pi_amp",
                    "$qubit.module.x90": "Q1.pulses.pi2_amp",
                    "$qubit.ext.pi_product": 0.04,
                },
                {
                    "$qubit.pulses.pi_amp": ("gain", 0.4),
                    "$qubit.pulses.pi2_amp": ("gain", 0.2),
                },
            )
        case "twotone/rabi/len_rabi":
            cfg = LenRabiCfg.model_validate(
                {
                    "modules": {"qub_pulse": pulse, "readout": readout},
                    "sweep": {"length": sweep},
                }
            )
            calibration = Calibration(
                RunRecord(cfg, LenRabiResult(axis, signals)),
                LenRabiAnalyzeResult(0.2, 0.01, 0.1, 0.01, 2.5, 0.1),
                {
                    "$qubit.ext.rabi_freq": 2.5,
                    "$qubit.module.x180": "Q1.pulses.pi_amp",
                    "$qubit.module.x90": "Q1.pulses.pi2_amp",
                    "$qubit.ext.pi_product": 0.04,
                },
                {
                    "$qubit.pulses.pi_amp": ("waveform.length", 0.2),
                    "$qubit.pulses.pi2_amp": ("waveform.length", 0.1),
                },
            )
        case "twotone/freq":
            calibration = Calibration(
                RunRecord(None, FreqResult(axis, signals)),
                FreqAnalyzeResult(4300.0, 0.1, 8.0, 0.2),
                {"$qubit.freq": 4300.0, "$qubit.kappa": 8.0},
                {},
            )
        case _:
            raise ValueError(f"No calibration example for {name}")
    return calibration


@pytest.mark.parametrize("name", _recipe_adapters())
@pytest.mark.parametrize(
    "change_point", [False, True], ids=["rebind", "rebind-and-point"]
)
@pytest.mark.io
def test_recipe_writeback_keeps_run_component_after_environment_change(
    session: WritebackSession, name: str, change_point: bool
) -> None:
    adapter = _registry().create(name)
    calibration = _calibration(name)
    captured = session.context.read_context()
    assert captured is not None
    run_env = SessionEnv(parameters=captured, soc=None, soccfg=None)
    session.control.set_alias_rows(
        (AliasRow("qubit", "Q2"), AliasRow("resonator", "$qubit.resonator"))
    )
    if change_point:
        session.control.use_point(PointUseRequest("later-point"))
    selected = session.control.read_context().target
    proposals = list(
        adapter.get_writeback_items(
            WritebackRequest(calibration.source, calibration.analysis, run_env)
        )
    )
    scalars = dict(calibration.scalars)
    if name == "singleshot/ge":
        confusion = [[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.0, 0.0, 1.0]]
        proposals.extend(
            adapter.get_post_writeback_items(
                PostWritebackRequest(
                    calibration.source,
                    calibration.analysis,
                    GEPostAnalyzeResult(0.3, confusion),
                    run_env,
                )
            )
        )
        scalars.update(
            {
                "$qubit.ext.readout_cal.radius": 0.3,
                "$qubit.ext.readout_cal.confusion_matrix": confusion,
            }
        )
    assert {
        item.target_name for item in proposals
    } == scalars.keys() | calibration.modules.keys()
    draft = session.service.create_draft(
        proposals, parameters=captured, experiment_name="test/writeback"
    )
    try:
        receipt = draft.apply()
        destination = session.control.read_target(captured.target)
        for path, value in scalars.items():
            concrete = path.replace("$qubit", "Q1").replace("$resonator", "R1")
            actual = destination.value(concrete)
            assert actual == (
                pytest.approx(value) if isinstance(value, float) else value
            )
        for path, (leaf, value) in calibration.modules.items():
            concrete = path.replace("$qubit", "Q1").replace("$resonator", "R1")
            assert destination.value(f"{concrete}.{leaf}") == pytest.approx(value)
        assert receipt["written"]["parameter"] == [
            item.target_name.replace("$qubit", "Q1").replace("$resonator", "R1")
            for item in proposals
            if item.target_name in scalars
        ]
        current = session.control.read_context()
        assert current.target == selected
        for path in scalars:
            concrete = path.replace("$qubit", "Q2").replace("$resonator", "R2")
            assert current.lookup(concrete).value is None
        assert not any(
            key.startswith(("Q2.", "R2.")) for key in current.library.module_keys()
        )
    finally:
        draft.teardown()
