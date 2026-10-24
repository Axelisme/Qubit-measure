"""Deterministic hardware-free adapters for application and remote contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, PulseSpec, Range
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

ALIAS_PRESET = (
    AliasRow("qubit", ""),
    AliasRow("resonator", "$qubit.resonator"),
    AliasRow("flux_dev", ""),
)


@dataclass(frozen=True)
class SampleResult:
    data: NDArray[np.float64]


class SampleCfg(ConfigBase):
    reps: int = 100
    rounds: int = 10
    gain: float = 0.1
    noise_scale: float = 0.1
    sweep: SweepCfg


SampleRun = RunRecord[SampleCfg, SampleResult]


@dataclass
class SampleParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)] = 0.5


@dataclass
class SampleAnalysis:
    peak: float


def SampleAdapter_make_form(ctx: FormContext) -> Form[SampleCfg]:
    del ctx
    return Form(SampleCfg, {"sweep": Range(5.0, 6.0, expts=11)})


def SampleAdapter_run[C: ConfigBase](
    cfg: C, *, context: RunContext
) -> RunRecord[C, SampleResult]:
    return RunRecord(cfg=cfg, result=SampleResult(np.array([0.1, 0.2, 0.3])))


def SampleAdapter_analyze[C: ConfigBase](
    req: AnalyzeRequest[RunRecord[C, SampleResult], SampleParams], *, plots: Plots
) -> SampleAnalysis:
    _, ax = plots.subplots("fit")
    ax.plot(req.run_result.result.data)
    return SampleAnalysis(peak=float(np.max(req.run_result.result.data)))


def SampleAdapter_get_writeback_items[C: ConfigBase](
    req: WritebackRequest[RunRecord[C, SampleResult], SampleAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            "general.ext.fake_peak", "Sample peak", req.analyze_result.peak
        )
    ]


SampleAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("data", "data", ""),
        result_type=SampleResult,
        cfg_type=SampleCfg,
        tag="test/sample",
    ),
    make_form=SampleAdapter_make_form,
    run=SampleAdapter_run,
    guide=None,
    analysis=Analysis(
        fn=SampleAdapter_analyze,
        options=lambda source, env: SampleParams(),
        writeback=SampleAdapter_get_writeback_items,
    ),
    post_analysis=None,
)


MODULE_PROPOSAL_TARGET = "Q1.pulses.tuned"
MODULE_PROPOSAL_WAVEFORM = "Q1.waveforms.flat"


def ModuleProposalAdapter_get_writeback_items[C: ConfigBase](
    req: WritebackRequest[RunRecord[C, SampleResult], SampleAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ModuleWriteback(
            MODULE_PROPOSAL_TARGET,
            "Tuned pulse",
            PulseSpec(
                ch=0,
                nqz=1,
                freq=5000.0,
                gain=req.analyze_result.peak,
                waveform=MODULE_PROPOSAL_WAVEFORM,
            ),
        )
    ]


ModuleProposalAdapter = Adapter(
    spec=SampleAdapter.spec,
    make_form=SampleAdapter.make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=Analysis(
        fn=SampleAdapter.analyze,
        options=lambda source, env: SampleParams(),
        writeback=ModuleProposalAdapter_get_writeback_items,
    ),
    post_analysis=SampleAdapter.post_analysis,
)


@dataclass
class ChoiceParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model")] = "hm"
    fit_bg_amp_slope: bool = True
    fit_bg_phase_curvature: bool = False
    edelay_mode: Literal["auto", "manual"] = "auto"
    manual_edelay: float | None = None
    max_edelay_search_radius: float = 100.0


@dataclass
class EmptyParams:
    pass


def PairAdapter_get_writeback_items[C: ConfigBase](
    req: WritebackRequest[RunRecord[C, SampleResult], SampleAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback("r_f", "First", 6100.0),
        ParameterWriteback("rf_w", "Second", 2.0),
    ]


PairAdapter = Adapter(
    spec=SampleAdapter.spec,
    make_form=SampleAdapter.make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=Analysis(
        fn=SampleAdapter.analyze,
        options=lambda source, env: SampleParams(),
        writeback=PairAdapter_get_writeback_items,
    ),
    post_analysis=SampleAdapter.post_analysis,
)


def register_all(registry) -> None:
    from tests.gui.app.measure._flux_adapter import (
        PhaseSpectrumAdapter,
        SpectrumAdapter,
    )

    registry.register("fake/freq", PairAdapter)
    registry.register("synthetic/magnitude", SpectrumAdapter)
    registry.register("synthetic/phase", PhaseSpectrumAdapter)
