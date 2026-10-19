"""Measure-GUI adapter for a MIST gain trace with averaged complex IQ."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepAnalyzeOptions,
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_lab.v2.mist.power_dep.single_trace.form import make_form

MistTraceRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


@dataclass
class MistTraceAnalyzeResult(AnalyzeResultBase):
    """Figure-only response analysis; no fitted scalars or writeback."""


@dataclass
class MistTraceAnalyzeParams:
    """IQ references as Python complex strings (e.g. 1+2j), or None to omit.

    g0=None uses the first measured IQ. e0 only sets the vertical display range.
    ac_coeff=None keeps gain on the x-axis; a supplied coefficient maps gain
    squared to photon number. Invalid complex strings fail analysis.
    """

    g0: Annotated[str | None, ParamMeta(label="Ground IQ (empty = first point)")] = None
    e0: Annotated[str | None, ParamMeta(label="Excited IQ (optional y-limit)")] = None
    ac_coeff: Annotated[
        float | None, ParamMeta(label="Photon number / gain squared", decimals=6)
    ] = None


class MistTraceAdapter(
    BaseAdapter[
        PowerDepCfg, MistTraceRunResult, MistTraceAnalyzeResult, MistTraceAnalyzeParams
    ]
):
    """Acquire averaged MIST IQ and plot displacement without population fitting."""

    experiment_tag: ClassVar[str | None] = PowerDepExp.persistence_tag()
    exp_cls = PowerDepExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
        """Create initial cfg from the fixed Open/Reset source capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: PowerDepCfg, *, context: RunContext
    ) -> MistTraceRunResult:
        """Reset, optionally initialize, probe and read out at each gain.

        Return hardware gain coordinates and averaged complex IQ with the exact
        accepted cfg. Acquisition errors follow the shared RunContext contract.
        """
        del req
        return RunRecord(cfg, PowerDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistTraceRunResult, MistTraceAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistTraceAnalyzeResult:
        """Plot source IQ displacement using the submitted reference/axis options.

        Accept loaded sources without cfg. Invalid complex reference strings
        raise ValueError; analysis does not read the current measurement form.
        """
        params = req.analyze_params
        options = PowerDepAnalyzeOptions(
            g0=None if params.g0 is None else complex(params.g0),
            e0=None if params.e0 is None else complex(params.e0),
            ac_coeff=params.ac_coeff,
        )
        PowerDepExp().analyze(req.run_result, options, plots=plots)
        return MistTraceAnalyzeResult()
