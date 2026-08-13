"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_tools.experiment.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.jpa import FluxCfg, FluxExp
from zcu_tools.experiment.v2.jpa.jpa_flux import FluxResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    custom,
    md_has_key,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import EvalValue, SweepValue

from ._shared import cached_device_snapshot, lower_jpa_flux_dev

_JPA_FLUX_SWEEP_EXPTS = 101
# Bring-up survey: ±5e-3 around the centre, taken from the notebook's JPA flux
# sweep (single_qubit.md, -5e-3..5e-3) and compressed to 101 points. These are
# inspectable starting bounds, NOT safety certification — the operator must
# review device and sweep.
_JPA_FLUX_SEED_SPAN = 5.0e-3


def jpa_flux_sweep_seed(
    ctx: ExpContext, *, expts: int = _JPA_FLUX_SWEEP_EXPTS
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
    figure: Figure


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
    BaseAdapter[FluxCfg, FluxResult, JpaFluxAnalyzeResult, NoAnalyzeParams]
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
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FluxCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_flux_dev(cfg_raw, cached_device_snapshot())
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req
        # Pure preflight over cached/static data — never commands a live device.
        lower_jpa_flux_dev(raw_cfg, cached_device_snapshot())

    def analyze(
        self, req: AnalyzeRequest[FluxResult, NoAnalyzeParams]
    ) -> JpaFluxAnalyzeResult:
        best_flux, fig = FluxExp().analyze(req.run_result)
        _relabel_flux_figure(fig, best_flux)
        return JpaFluxAnalyzeResult(best_flux=best_flux, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[FluxResult, JpaFluxAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            MetaDictWriteback(
                target_name="best_jpa_flux",
                description="Best JPA flux device value",
                proposed_value=req.analyze_result.best_flux,
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_jpa_flux_{time.strftime('%m%d')}"
