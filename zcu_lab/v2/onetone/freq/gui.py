from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import (
    pulse_readout_module_writeback_items,
)
from zcu_lab.v2._support.resonator import finite_real, route_matched_delay
from zcu_lab.v2.onetone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.onetone.freq.form import make_form

OneToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]
SamplingMode: TypeAlias = Literal["linear", "homophasal"]
EDelayMode: TypeAlias = Literal["auto", "calibrated", "manual"]
EDelaySource: TypeAlias = Literal["global", "calibrated", "manual"]

_EDELAY_PARAMETER_PATH = "resonator.ext.edelay"
_DEFAULT_MAX_EDELAY_SEARCH_RADIUS = 100.0


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")] = (
        "hm"
    )
    fit_bg_amp_slope: Annotated[bool, ParamMeta(label="Fit amplitude background")] = (
        True
    )
    fit_bg_phase_curvature: Annotated[bool, ParamMeta(label="Fit phase curvature")] = (
        False
    )
    edelay_mode: Annotated[EDelayMode, ParamMeta(label="Electrical-delay mode")] = (
        "auto"
    )
    manual_edelay: Annotated[
        float | None, ParamMeta(label="Manual electrical delay")
    ] = None
    max_edelay_search_radius: Annotated[
        float, ParamMeta(label="Maximum electrical-delay search radius")
    ] = _DEFAULT_MAX_EDELAY_SEARCH_RADIUS


@dataclass
class OneToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    edelay: float | None = None
    edelay_source: EDelaySource = "global"
    edelay_persistable: bool = False


class OneToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        OneToneFreqRunResult,
        OneToneFreqAnalyzeResult,
        OneToneFreqAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def _readout_route(result: OneToneFreqRunResult) -> tuple[int, int] | None:
        cfg = result.cfg
        if cfg is None:
            return None
        res_ch = cfg.modules.readout.gen_ch
        ro_ch = cfg.modules.readout.ro_ch
        if (
            isinstance(res_ch, bool)
            or not isinstance(res_ch, Integral)
            or isinstance(ro_ch, bool)
            or not isinstance(ro_ch, Integral)
        ):
            return None
        return int(res_ch), int(ro_ch)

    @classmethod
    def _route_matched_prior(
        cls, parameters: ParameterContext | None, route: tuple[int, int] | None
    ) -> float | None:
        if parameters is None or route is None:
            return None
        lookup = parameters.lookup(f"${_EDELAY_PARAMETER_PATH}")
        return route_matched_delay(lookup.value, route) if lookup.present else None

    @classmethod
    def _resolve_edelay_seed(
        cls,
        params: OneToneFreqAnalyzeParams,
        parameters: ParameterContext | None,
        route: tuple[int, int] | None,
    ) -> tuple[float | None, EDelaySource]:
        if params.edelay_mode == "manual":
            if params.manual_edelay is None:
                raise ValueError("manual edelay mode requires manual_edelay")
            seed = finite_real(params.manual_edelay)
            if seed is None:
                raise ValueError("manual_edelay must be finite")
            return seed, "manual"

        prior = cls._route_matched_prior(parameters, route)
        if params.edelay_mode == "calibrated":
            if prior is None:
                raise ValueError(
                    "calibrated edelay mode requires a finite route-matched "
                    "resonator.ext.edelay prior"
                )
            return prior, "calibrated"
        if params.edelay_mode == "auto":
            if prior is not None:
                return prior, "calibrated"
            return None, "global"
        raise ValueError(f"Invalid electrical-delay mode: {params.edelay_mode!r}")

    @staticmethod
    def _has_nonuniform_fitting_grid(result: OneToneFreqRunResult) -> bool:
        fitting_freqs = result.result.freqs[1:-1]
        if len(fitting_freqs) < 3:
            return False
        steps = np.abs(np.diff(fitting_freqs))
        return not bool(np.allclose(steps, np.median(steps), rtol=1e-8, atol=0.0))

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> OneToneFreqRunResult:
        del req
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> OneToneFreqAnalyzeResult:
        params = req.analyze_params
        route = self._readout_route(req.run_result)
        edelay_seed, edelay_source = self._resolve_edelay_seed(
            params, req.parameters, route
        )
        max_search_radius: float | None = None
        if edelay_seed is None:
            max_search_radius = finite_real(params.max_edelay_search_radius)
            if max_search_radius is None or max_search_radius <= 0.0:
                raise ValueError(
                    "max_edelay_search_radius must be positive and finite, got "
                    f"{params.max_edelay_search_radius!r}"
                )
        answer = FreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(
                model_type=params.model_type,
                fit_bg_amp_slope=params.fit_bg_amp_slope,
                fit_bg_phase_curvature=params.fit_bg_phase_curvature,
                edelay_branch_seed=edelay_seed,
                edelay_max_search_radius=max_search_radius,
            ),
            plots=plots,
        )
        edelay = finite_real(answer.params.get("edelay"))
        if edelay is None:
            raise ValueError("one-tone fit result must contain a finite edelay")
        return OneToneFreqAnalyzeResult(
            freq=answer.freq,
            fwhm=answer.fwhm,
            params=answer.params,
            edelay=edelay,
            edelay_source=edelay_source,
            edelay_persistable=(
                route is not None
                and (
                    edelay_seed is not None
                    or self._has_nonuniform_fitting_grid(req.run_result)
                )
            ),
        )

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFreqRunResult, OneToneFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        theta0 = result.params.get("theta0")
        if isinstance(theta0, bool) or not isinstance(theta0, Real):
            raise ValueError(
                "OneToneFreqAnalyzeResult.params must contain numeric 'theta0' "
                "for writeback"
            )
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$resonator.freq",
                description="Resonator frequency (MHz)",
                proposed_value=result.freq,
            ),
            ParameterWriteback(
                target_name="$resonator.kappa",
                description="Resonator linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
            ParameterWriteback(
                target_name="$resonator.ext.theta0",
                description="Resonator circle phase offset (rad)",
                proposed_value=float(theta0),
            ),
        ]
        if result.edelay_persistable:
            route = self._readout_route(req.run_result)
            edelay = finite_real(result.edelay)
            if route is None or edelay is None:
                raise ValueError(
                    "persistable electrical delay requires a finite fitted edelay "
                    "and pulse-readout route"
                )
            res_ch, ro_ch = route
            items.append(
                ParameterWriteback(
                    target_name=f"${_EDELAY_PARAMETER_PATH}",
                    description="Route-qualified resonator electrical-delay calibration",
                    proposed_value={
                        "edelay": edelay,
                        "res_ch": res_ch,
                        "ro_ch": ro_ch,
                    },
                )
            )
        items.extend(
            pulse_readout_module_writeback_items(
                req.run_result.cfg,
                target="$resonator.readout.readout_rf",
                desc="Readout at fitted resonator frequency",
                field_updates=(("freq", float(result.freq)),),
                role_id="readout",
            )
        )
        return items
