from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.experiment.v2.onetone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    md,
    pulse_readout_module_writeback_items,
    res_freq_range,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)

OneToneFreqRunResult: TypeAlias = FreqResult
SamplingMode: TypeAlias = Literal["linear", "homophasal"]
EDelayMode: TypeAlias = Literal["auto", "calibrated", "manual"]
EDelaySource: TypeAlias = Literal["global", "calibrated", "manual"]

_SAMPLING_MODE_CHOICES: list[SamplingMode] = ["linear", "homophasal"]
_HOMOPHASAL_MD_KEYS = ("r_f", "rf_w", "theta0")
_EDELAY_MD_KEY = "res_edelay_calibration"
_EDELAY_CALIBRATION_FIELDS = frozenset({"edelay", "res_ch", "ro_ch"})
_DEFAULT_MAX_EDELAY_SEARCH_RADIUS = 100.0


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")] = (
        "hm"
    )
    fit_bg_amp_slope: Annotated[bool, ParamMeta(label="Fit amplitude background")] = (
        True
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
    figure: Figure
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
    legacy_migration_experiment: ClassVar[str | None] = "onetone/freq"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "One-tone resonator spectroscopy: sweeps the readout frequency "
            "and fits the resonator response to extract its frequency and "
            "linewidth. Runs on real hardware. Typically run after a coarse "
            "resonator search has placed you near the right frequency."
        ),
        expects_md=(
            "Reads from the MetaDict: 'r_f' seeds the default sweep centre; "
            "'rf_w' seeds the default span; 'res_probe_len' seeds the readout "
            "window; 'res_ch' / 'ro_ch' seed drive / readout channels; "
            "'timeFly' seeds the trigger offset. Homophasal sampling also "
            "requires fitted 'r_f', 'rf_w', and 'theta0'. Analysis optionally "
            "uses a route-matched 'res_edelay_calibration' value."
        ),
        expects_ml=(
            "Needs a pulse-readout module, and references a ModuleLibrary "
            "waveform named 'ro_waveform' when one exists (optional)."
        ),
        typical_writeback=(
            "Proposes the fitted resonator frequency, linewidth, and phase "
            "offset into MetaDict 'r_f' / 'rf_w' / 'theta0'. When the run "
            "result includes a pulse-readout cfg snapshot, it also proposes "
            "ModuleLibrary 'readout_rf' from that readout template, changing "
            "only pulse/readout frequency to the fitted 'r_f' while preserving "
            "gain, lengths, channels, and trigger timing. A route-qualified "
            "electrical-delay calibration is proposed only when its absolute "
            "branch is identifiable or a trusted prior/manual seed was used."
        ),
        recommended=(
            "Analysis defaults to the hanger-model fit ('hm') with "
            "multiplicative amplitude-background fitting on. A sweep spanning "
            "a few linewidths "
            "around the known resonator frequency usually captures the dip "
            "cleanly; widen it if the resonator has drifted. Use "
            "homophasal sampling after a fit has written 'theta0' when you "
            "want equal resonator-circle phase spacing instead of a linear "
            "frequency grid."
            " Electrical-delay mode 'auto' uses a route-matched calibration "
            "when available and otherwise runs bounded adaptive search; use "
            "'calibrated' to require that prior or 'manual' to supply a seed."
        ),
    )

    @staticmethod
    def _readout_route(result: OneToneFreqRunResult) -> tuple[int, int] | None:
        cfg = result.cfg_snapshot
        if cfg is None:
            return None
        try:
            res_ch = cfg.modules.readout.pulse_cfg.ch
            ro_ch = cfg.modules.readout.ro_cfg.ro_ch
        except AttributeError:
            return None
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
        cls, md_obj: object, route: tuple[int, int] | None
    ) -> float | None:
        if route is None:
            return None
        get = getattr(md_obj, "get")
        raw = get(_EDELAY_MD_KEY)
        if not isinstance(raw, Mapping) or set(raw) != _EDELAY_CALIBRATION_FIELDS:
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
        md_obj: object,
        route: tuple[int, int] | None,
    ) -> tuple[float | None, EDelaySource]:
        if params.edelay_mode == "manual":
            if params.manual_edelay is None:
                raise ValueError("manual edelay mode requires manual_edelay")
            seed = cls._finite_real(params.manual_edelay)
            if seed is None:
                raise ValueError("manual_edelay must be finite")
            return seed, "manual"

        prior = cls._route_matched_prior(md_obj, route)
        if params.edelay_mode == "calibrated":
            if prior is None:
                raise ValueError(
                    "calibrated edelay mode requires a finite route-matched "
                    "res_edelay prior"
                )
            return prior, "calibrated"
        if params.edelay_mode == "auto":
            if prior is not None:
                return prior, "calibrated"
            return None, "global"
        raise ValueError(f"Invalid electrical-delay mode: {params.edelay_mode!r}")

    @staticmethod
    def _has_nonuniform_fitting_grid(result: OneToneFreqRunResult) -> bool:
        fitting_freqs = result.freqs[1:-1]
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
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                locked={"pulse_cfg.freq": 0.0, "ro_cfg.ro_freq": 0.0},
                overrides={
                    "pulse_cfg.gain": 0.05,
                    "ro_cfg.ro_length": md(
                        "res_probe_len",
                        expr="res_probe_len - 0.1",
                        fallback=0.9,
                    ),
                },
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
            .reps(100)
            .rounds(100)
            .build()
        )

    def _sampling_mode(self, raw_cfg: dict[str, object]) -> SamplingMode:
        value = raw_cfg.get("sampling_mode", "linear")
        if value == "linear":
            return "linear"
        if value == "homophasal":
            return "homophasal"
        raise ValueError(
            "'sampling_mode' must be one of "
            f"{', '.join(_SAMPLING_MODE_CHOICES)}, got {value!r}"
        )

    def _homophasal_params_from_md(self, md: object) -> dict[str, float]:
        values: dict[str, float] = {}
        missing: list[str] = []
        invalid: list[str] = []
        get = getattr(md, "get")
        for key in _HOMOPHASAL_MD_KEYS:
            value = get(key)
            if value is None:
                missing.append(key)
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                invalid.append(f"{key}={value!r}")
                continue
            values[key] = float(value)

        problems: list[str] = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if invalid:
            problems.append(f"non-numeric: {', '.join(invalid)}")
        if problems:
            raise ValueError(
                "homophasal sampling requires numeric MetaDict keys "
                f"{', '.join(_HOMOPHASAL_MD_KEYS)} ({'; '.join(problems)})"
            )
        if values["r_f"] <= 0.0:
            raise ValueError(f"MetaDict r_f must be positive, got {values['r_f']}")
        if values["rf_w"] <= 0.0:
            raise ValueError(f"MetaDict rf_w must be positive, got {values['rf_w']}")
        return values

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        if self._sampling_mode(raw_cfg) == "homophasal":
            self._homophasal_params_from_md(req.md)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FreqCfg:
        cfg_raw = dict(raw_cfg)
        if self._sampling_mode(cfg_raw) == "homophasal":
            cfg_raw["homophasal"] = self._homophasal_params_from_md(req.md)
        else:
            cfg_raw.pop("homophasal", None)
        return super().build_exp_cfg(cfg_raw, req)

    def analyze(
        self, req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams]
    ) -> OneToneFreqAnalyzeResult:
        params = req.analyze_params
        route = self._readout_route(req.run_result)
        edelay_seed, edelay_source = self._resolve_edelay_seed(params, req.md, route)
        max_search_radius: float | None = None
        if edelay_seed is None:
            max_search_radius = self._finite_real(params.max_edelay_search_radius)
            if max_search_radius is None or max_search_radius <= 0.0:
                raise ValueError(
                    "max_edelay_search_radius must be positive and finite, got "
                    f"{params.max_edelay_search_radius!r}"
                )
        freq, fwhm, fit_params, figure = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            fit_bg_amp_slope=params.fit_bg_amp_slope,
            edelay_branch_seed=edelay_seed,
            edelay_max_search_radius=max_search_radius,
        )
        edelay = self._finite_real(fit_params.get("edelay"))
        if edelay is None:
            raise ValueError("one-tone fit result must contain a finite edelay")
        return OneToneFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=fit_params,
            figure=figure,
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
            MetaDictWriteback(
                target_name="r_f",
                description="Resonator frequency (MHz)",
                proposed_value=result.freq,
            ),
            MetaDictWriteback(
                target_name="rf_w",
                description="Resonator linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
            MetaDictWriteback(
                target_name="theta0",
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
                MetaDictWriteback(
                    target_name=_EDELAY_MD_KEY,
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
                req.run_result.cfg_snapshot,
                target="readout_rf",
                desc="Readout at fitted resonator frequency",
                field_updates=(
                    ("pulse_cfg.freq", float(result.freq)),
                    ("ro_cfg.ro_freq", float(result.freq)),
                ),
                role_id="readout",
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"
