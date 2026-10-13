"""Deterministic hardware-free adapters for application and remote contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import PulseSpec, Range
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_tools.gui.cfg import SweepSpec, SweepValue
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from tests.gui.app.measure._templates import ROLE_FACTORIES

ALIAS_PRESET = (
    AliasRow("qubit", ""),
    AliasRow("resonator", "$qubit.resonator"),
    AliasRow("flux_dev", ""),
)


@dataclass(frozen=True)
class SampleResult:
    data: NDArray[np.float64]


class SampleCfg(ExpCfgModel):
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
class SampleAnalysis(AnalyzeResultBase):
    peak: float


class SampleAdapter(BaseAdapter[SampleCfg, SampleRun, SampleAnalysis, SampleParams]):
    form = Form(SampleCfg, {"sweep": Range.between(5.0, 6.0, expts=11)})
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=False
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(role_factories={})
            .reps(100)
            .rounds(10)
            .field(
                "sweep",
                spec=SweepSpec(label="Sample"),
                default=SweepValue(start=5.0, stop=6.0, expts=11),
            )
            .float("gain", label="Gain", default=0.1)
            .float("noise_scale", label="Noise scale", default=0.1)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> SampleCfg:
        return SampleCfg.model_validate(raw_cfg)

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SampleRun:
        return RunRecord(
            self.build_exp_cfg(raw_cfg, req), SampleResult(np.array([0.1, 0.2, 0.3]))
        )

    def analyze(
        self, req: AnalyzeRequest[SampleRun, SampleParams], *, plots: Plots
    ) -> SampleAnalysis:
        _, ax = plots.subplots("fit")
        ax.plot(req.run_result.result.data)
        return SampleAnalysis(peak=float(np.max(req.run_result.result.data)))

    def get_writeback_items(
        self, req: WritebackRequest[SampleRun, SampleAnalysis]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                "general.ext.fake_peak", "Sample peak", req.analyze_result.peak
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_fake"


MODULE_PROPOSAL_TARGET = "Q1.pulses.tuned"
MODULE_PROPOSAL_WAVEFORM = "Q1.waveforms.flat"


class ModuleProposalAdapter(SampleAdapter):
    """Propose a pulse at MODULE_PROPOSAL_TARGET whose gain is the fitted peak.

    The pulse waveform is the library key MODULE_PROPOSAL_WAVEFORM, so callers
    install that waveform and a Q1 component before applying the proposal.
    """

    def get_writeback_items(
        self, req: WritebackRequest[SampleRun, SampleAnalysis]
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


class PairAdapter(SampleAdapter):
    """Config reference and two scalar proposals; no scientific fit involved."""

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
            .readout(
                init=ModuleInit.INLINE,
                locked={"pulse_cfg.freq": 0.0, "ro_cfg.ro_freq": 0.0},
            )
            .sweep(
                "freq",
                label="Frequency",
                default=SweepValue(start=6000.0, stop=6002.0, expts=3, step=0.5),
            )
            .reps(100)
            .rounds(1)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> SampleCfg:
        return SampleCfg(sweep=SweepCfg(start=1.0, stop=2.0, expts=3, step=0.5))

    def get_writeback_items(
        self, req: WritebackRequest[SampleRun, SampleAnalysis]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback("r_f", "First", 6100.0),
            ParameterWriteback("rf_w", "Second", 2.0),
        ]


def register_all(registry) -> None:
    from tests.gui.app.measure._flux_adapter import (
        PhaseSpectrumAdapter,
        SpectrumAdapter,
    )

    registry.register("fake/freq", PairAdapter)
    registry.register("synthetic/magnitude", SpectrumAdapter)
    registry.register("synthetic/phase", PhaseSpectrumAdapter)
