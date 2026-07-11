from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

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

_SAMPLING_MODE_CHOICES: list[SamplingMode] = ["linear", "homophasal"]
_HOMOPHASAL_MD_KEYS = ("r_f", "rf_w", "theta0")


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")] = (
        "hm"
    )
    fit_bg_slope: Annotated[bool, ParamMeta(label="Fit background slope")] = True


@dataclass
class OneToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


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
            "requires fitted 'r_f', 'rf_w', and 'theta0'."
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
            "gain, lengths, channels, and trigger timing."
        ),
        recommended=(
            "Analysis defaults to the hanger-model fit ('hm') with "
            "background-slope fitting on. A sweep spanning a few linewidths "
            "around the known resonator frequency usually captures the dip "
            "cleanly; widen it if the resonator has drifted. Use "
            "homophasal sampling after a fit has written 'theta0' when you "
            "want equal resonator-circle phase spacing instead of a linear "
            "frequency grid."
        ),
    )

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
        freq, fwhm, fit_params, figure = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            fit_bg_slope=params.fit_bg_slope,
        )
        return OneToneFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=fit_params,
            figure=figure,
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
