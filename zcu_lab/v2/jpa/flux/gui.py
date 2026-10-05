"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.ctx_helpers import md_has_key
from zcu_lab.v2._support.measure.jpa_shared import lower_jpa_flux_dev
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import custom
from zcu_lab.v2.jpa.flux.core import FluxCfg, FluxExp, FluxResult

JpaFluxRunResult: TypeAlias = RunRecord[FluxCfg, FluxResult]

_JPA_FLUX_SWEEP_EXPTS = 101
# Bring-up survey: ±5e-3 around the centre, taken from the notebook's JPA flux
# sweep (single_qubit.md, -5e-3..5e-3) and compressed to 101 points. These are
# inspectable starting bounds, NOT safety certification — the operator must
# review device and sweep.
_JPA_FLUX_SEED_SPAN = 5.0e-3


def jpa_flux_sweep_seed(
    ctx: SessionEnv, *, expts: int = _JPA_FLUX_SWEEP_EXPTS
) -> SweepValue:
    """JPA flux sweep seed: centred on ``best_jpa_flux`` when known, else the
    notebook-derived literal survey around zero.

    The sweep uses the neutral 'JPA flux device value' quantity of the selected
    device — the adapter never claims a physical-unit migration.
    """

    if md_has_key(ctx, "best_jpa_flux"):
        return SweepValue(
            start=EvalValue(expr=f"best_jpa_flux - {_JPA_FLUX_SEED_SPAN}"),
            stop=EvalValue(expr=f"best_jpa_flux + {_JPA_FLUX_SEED_SPAN}"),
            expts=expts,
        )
    return SweepValue(
        start=-_JPA_FLUX_SEED_SPAN,
        stop=_JPA_FLUX_SEED_SPAN,
        expts=expts,
    )


@dataclass
class JpaFluxAnalyzeResult(AnalyzeResultBase):
    best_flux: float


def _relabel_flux_figure(fig: Figure, best_flux: float) -> None:
    """Neutral JPA flux device value wording on the GUI review figure.

    The core analysis figure still labels its flux axis and the optimum legend
    with 'a.u.'; the GUI review figure must use the neutral 'JPA flux device
    value' vocabulary (A6) without claiming a physical-unit migration. The
    relabel happens at the adapter analysis boundary, in place, so the review
    figure wording is adapter-owned while the core figure stays untouched.
    """
    for ax in fig.axes:
        ax.set_xlabel("JPA flux device value")
        for line in ax.get_lines():
            label = line.get_label()
            if isinstance(label, str) and label.startswith("best JPA flux"):
                line.set_label(f"best JPA flux device value = {best_flux:.2g}")
        legend = ax.get_legend()
        if legend is not None:
            ax.legend()


class JpaFluxAdapter(
    BaseAdapter[FluxCfg, JpaFluxRunResult, JpaFluxAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = FluxExp
    ExpCfg_cls: ClassVar[Any] = FluxCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA flux-device calibration: with the qubit prepared in g and e "
            "(a pi pulse toggles it), sweeps the JPA flux device value and "
            "measures the g/e signal difference, so you can pick the flux "
            "device value that best enhances readout. Runs on real hardware. "
            "WARNING: review the selected JPA flux device and the sweep before "
            "running — the seeded bounds are bring-up defaults, not certified "
            "safety limits, and the run commands the selected device."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'best_jpa_flux' — a "
            "previously accepted JPA flux device value, preferred as the sweep "
            "centre; 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — "
            "cable time-of-flight for the trigger offset; 'q_f' / 'qub_ch' — "
            "qubit frequency / drive channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform named "
            "'ro_waveform' when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing JPA flux device value into "
            "MetaDict 'best_jpa_flux' as a draft — it never writes it back "
            "without your acceptance, never commands the device from the "
            "writeback, and never touches 'cur_jpa_A'."
        ),
        recommended=(
            "Review the selected flux device and the sweep bounds before every "
            "run; the seeded range around the current centre is only a "
            "starting point. Analysis picks the peak of the absolute signal "
            "difference."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(0.5)
            .device(
                "jpa_flux_dev",
                label="JPA flux device",
                default="",
                required=False,
            )
            .sweep(
                "jpa_flux",
                label="JPA flux device value",
                default=custom(
                    jpa_flux_sweep_seed,
                    description="jpa flux device value range",
                ),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(10000)
            .rounds(1)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FluxCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_flux_dev(cfg_raw, req.device_snapshot)
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        # Pure preflight over the detached request snapshot.
        lower_jpa_flux_dev(raw_cfg, req.device_snapshot)

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> JpaFluxRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FluxExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaFluxRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaFluxAnalyzeResult:
        answer = FluxExp().analyze(req.run_result, None, plots=plots)
        _relabel_flux_figure(plots["fit"], answer.best_flux)
        return JpaFluxAnalyzeResult(best_flux=answer.best_flux)

    def get_writeback_items(
        self, req: WritebackRequest[JpaFluxRunResult, JpaFluxAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            MetaDictWriteback(
                target_name="best_jpa_flux",
                description="Best JPA flux device value",
                proposed_value=req.analyze_result.best_flux,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_flux_{time.strftime('%m%d')}"
