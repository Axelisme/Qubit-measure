from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    ReferenceValue,
    ScalarSpec,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import custom, path_seed, res_freq_range
from zcu_lab.v2._support.measure.spec_helpers import make_pulse_readout_module_spec
from zcu_lab.v2._support.measure.writeback_helpers import (
    pulse_readout_module_writeback_items,
)
from zcu_lab.v2.onetone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
    HomophasalSamplingCfg,
)

OneToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]
SamplingMode: TypeAlias = Literal["linear", "homophasal"]
EDelayMode: TypeAlias = Literal["auto", "calibrated", "manual"]
EDelaySource: TypeAlias = Literal["global", "calibrated", "manual"]

_SAMPLING_MODE_CHOICES: list[SamplingMode] = ["linear", "homophasal"]
_EDELAY_PARAMETER_PATH = "resonator.ext.edelay"
_EDELAY_CALIBRATION_FIELDS = frozenset({"edelay", "res_ch", "ro_ch"})
_DEFAULT_MAX_EDELAY_SEARCH_RADIUS = 100.0


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    """Seed a fresh sweep-owned readout from the tab's captured resonator.

    Keep generator/ADC frequencies locked at zero, gain 0.05 and pulse length
    1 us. Resolve channels, optional ADC length minus 0.1 us, and trigger delay
    plus 0.05 us from resonator role paths; absent/null values use 0, 0.9 and
    0.55 respectively. Source and role errors propagate.
    """
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    value.with_field("pulse_cfg.freq", 0.0)
    value.with_field(
        "pulse_cfg.ch", path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.05)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", 0.0)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field(
        "ro_cfg.ro_length",
        path_seed(
            "$resonator.ext.default_probe_length",
            fallback=0.9,
            expr="$resonator.ext.default_probe_length - 0.1",
        ).resolve(ctx),
    )
    value.with_field(
        "ro_cfg.trig_offset",
        path_seed(
            "$resonator.ext.time_of_fly",
            fallback=0.55,
            expr="$resonator.ext.time_of_fly + 0.05",
        ).resolve(ctx),
    )
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


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
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "One-tone resonator spectroscopy: sweeps the readout frequency "
            "and fits the resonator response to extract its frequency and "
            "linewidth. Runs on real hardware. Typically run after a coarse "
            "resonator search has placed you near the right frequency."
        ),
        expects_md=(
            "Uses the tab's captured resonator parameters: freq and kappa "
            "seed the sweep; ext.default_probe_length and ext.time_of_fly "
            "seed readout timing; wiring.ch and wiring.ro_ch seed channels. "
            "Homophasal sampling requires explicit cfg calibration values "
            "for r_f, rf_w, and theta0, supplied directly or as expressions. "
            "Analysis optionally reads the route-matched resonator.ext.edelay "
            "prior from the same fixed capture."
        ),
        expects_ml=(
            "Starts with a fresh inline pulse readout and a const waveform. "
            "It does not adopt a calibrated library module or named waveform."
        ),
        typical_writeback=(
            "Proposes fitted values to resonator.freq, resonator.kappa, and "
            "resonator.ext.theta0 through the captured role binding. With a "
            "pulse-readout run snapshot, it also proposes "
            "resonator.readout.readout_rf from that template, changing only "
            "pulse/readout frequency while preserving gain, lengths, channels, "
            "and trigger timing. The compound resonator.ext.edelay proposal "
            "keeps delay and route together; it requires an identifiable "
            "absolute branch or a trusted prior/manual seed."
        ),
        recommended=(
            "Analysis defaults to the hanger-model fit ('hm') with "
            "multiplicative amplitude-background fitting on. A sweep spanning "
            "a few linewidths "
            "around the known resonator frequency usually captures the dip "
            "cleanly; widen it if the resonator has drifted. Use "
            "homophasal sampling after a fit has written 'theta0' when you "
            "want equal resonator-circle phase spacing instead of a linear "
            "frequency grid. The homophasal calibration fields start empty; "
            "fill them with fitted values or expressions before selecting that mode."
            " Electrical-delay mode 'auto' uses a route-matched calibration "
            "when available and otherwise runs bounded adaptive search; use "
            "'calibrated' to require that prior or 'manual' to supply a seed."
        ),
    )

    @staticmethod
    def _readout_route(result: OneToneFreqRunResult) -> tuple[int, int] | None:
        cfg = result.cfg
        if cfg is None:
            return None
        res_ch = cfg.modules.readout.pulse_cfg.ch
        ro_ch = cfg.modules.readout.ro_cfg.ro_ch
        if (
            isinstance(res_ch, bool)
            or not isinstance(res_ch, Integral)
            or isinstance(ro_ch, bool)
            or not isinstance(ro_ch, Integral)
        ):
            return None
        return int(res_ch), int(ro_ch)

    @staticmethod
    def _finite_real(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        converted = float(value)
        return converted if np.isfinite(converted) else None

    @staticmethod
    def _channel(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            return None
        return int(value)

    @classmethod
    def _route_matched_prior(
        cls, parameters: ParameterContext | None, route: tuple[int, int] | None
    ) -> float | None:
        if parameters is None or route is None:
            return None
        lookup = parameters.lookup(f"${_EDELAY_PARAMETER_PATH}")
        raw = lookup.value
        if not lookup.present or raw is None or not isinstance(raw, Mapping):
            return None
        fields: frozenset[str] = frozenset(raw)
        if fields != _EDELAY_CALIBRATION_FIELDS:
            return None
        prior = cls._finite_real(raw["edelay"])
        prior_res_ch = cls._channel(raw["res_ch"])
        prior_ro_ch = cls._channel(raw["ro_ch"])
        if prior is None or prior_res_ch is None or prior_ro_ch is None:
            return None
        if (prior_res_ch, prior_ro_ch) != route:
            return None
        return prior

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
            seed = cls._finite_real(params.manual_edelay)
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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            # No reset module — one-tone spectroscopy runs without a qubit
            # reset. The sweep owns both readout frequency leaves.
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
                default=custom(_blank_readout, description="inline readout"),
            )
            .relax_delay(1.0)
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=301),
            )
            .choice(
                "sampling_mode",
                label="Sampling mode",
                choices=_SAMPLING_MODE_CHOICES,
                default="linear",
            )
            .field(
                "homophasal.r_f",
                spec=ScalarSpec("Resonance frequency (MHz)", float, optional=True),
                default=None,
            )
            .field(
                "homophasal.rf_w",
                spec=ScalarSpec("Linewidth (MHz)", float, optional=True),
                default=None,
            )
            .field(
                "homophasal.theta0",
                spec=ScalarSpec("Phase offset (rad)", float, optional=True),
                default=None,
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    def _sampling_mode(self, raw_cfg: Mapping[str, ExperimentCfgValue]) -> SamplingMode:
        value = raw_cfg.get("sampling_mode", "linear")
        if value == "linear":
            return "linear"
        if value == "homophasal":
            return "homophasal"
        raise ValueError(
            "'sampling_mode' must be one of "
            f"{', '.join(_SAMPLING_MODE_CHOICES)}, got {value!r}"
        )

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> FreqCfg:
        cfg_raw = dict(raw_cfg)
        if self._sampling_mode(cfg_raw) == "homophasal":
            # Validate required fit values before the assembler can query devices.
            HomophasalSamplingCfg.model_validate(cfg_raw.get("homophasal"))
        else:
            cfg_raw.pop("homophasal", None)
        return super().build_exp_cfg(cfg_raw, req)

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> OneToneFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
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
            max_search_radius = self._finite_real(params.max_edelay_search_radius)
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
        edelay = self._finite_real(answer.params.get("edelay"))
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
            edelay = self._finite_real(result.edelay)
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
                field_updates=(
                    ("pulse_cfg.freq", float(result.freq)),
                    ("ro_cfg.ro_freq", float(result.freq)),
                ),
                role_id="readout",
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"
