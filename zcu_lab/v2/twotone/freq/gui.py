from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.freq.core import (
    FreqAnalysis,
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.freq.form import make_form

FreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


class TwotoneSpectrumSummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    freq: float | None = Field(
        description="Owner freq in MHz; null retains an unavailable estimate."
    )

    freq_err: float | None = Field(
        description="Owner freq_err in MHz; null retains an unavailable estimate."
    )

    fwhm: float | None = Field(
        description="Owner fwhm in MHz; null retains an unavailable estimate."
    )

    fwhm_err: float | None = Field(
        description="Owner fwhm_err in MHz; null retains an unavailable estimate."
    )


@dataclass
class FreqAnalyzeParams:
    model_type: Annotated[Literal["lor", "sinc"], ParamMeta(label="Model type")] = "lor"
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


def _Freq_analyze(
    req: AnalyzeRequest[FreqRunResult, FreqAnalyzeOptions], *, plots: Plots
) -> FreqAnalysis:
    return FreqExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _Freq_get_writeback_items(
    req: WritebackRequest[FreqRunResult, FreqAnalysis],
) -> Sequence[WritebackItem]:
    """Propose fitted qubit frequency parameters and a captured-component sample.

    req supplies the source result and fixed Run context. Sample fields use
    MHz and target the Run's resolved $qubit, never the live selection. Without
    a parameter capture, retain parameter proposals only so loaded data can
    still be analyzed; Apply rejects the missing capture. Unbound $qubit
    with a capture raises RoleResolutionError. WritebackService supplies
    bias coordinates or omits the sample with its existing public reason.
    No write or device access occurs here; this Run leaves devices fixed.
    """
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.freq",
            description="Qubit frequency (MHz)",
            proposed_value=result.freq,
        ),
        ParameterWriteback(
            target_name="$qubit.kappa",
            description="Qubit linewidth FWHM (MHz)",
            proposed_value=result.fwhm,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None:
        items.append(
            SampleWriteback(
                target_name=parameters.resolve_path("$qubit"),
                description="Qubit frequency sample at the captured bias",
                proposed_value={
                    "freq_mhz": result.freq,
                    "freq_err_mhz": result.freq_err,
                },
            )
        )
    return items


def _Freq_make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: FreqCfg, *, context: RunContext) -> RunRecord[FreqCfg, FreqResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=FreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=_Freq_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_Freq_analyze,
        options=lambda source, env: FreqAnalyzeOptions(),
        writeback=_Freq_get_writeback_items,
    ),
)
