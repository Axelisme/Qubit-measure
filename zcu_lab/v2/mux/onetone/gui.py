"""Measure attachment for common-offset mux spectroscopy."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import MuxReadoutSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.resonator import (
    ResonanceFit,
    ResonanceOptions,
    finite_real,
    fit_resonance,
    route_matched_delay,
)
from zcu_lab.v2.mux.onetone.core import (
    MuxOnetoneCfg,
    MuxOnetoneExp,
    MuxOnetoneResult,
)
from zcu_lab.v2.mux.onetone.form import make_form
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAnalyzeParams

MuxOnetoneRunResult: TypeAlias = RunRecord[MuxOnetoneCfg, MuxOnetoneResult]


class MuxOnetoneTone(BaseModel):
    """Original index/display label, native fit status and optional owner estimates.

    freq/kappa use MHz; theta0 uses rad and edelay inverse MHz. reason is the
    native failure reason. fit_quality is the fitting owner's diagnostic type.
    edelay_source records the seed source; persistable follows native policy.
    """

    index: int
    label: str
    status: Literal["success", "failed"]
    freq: float | None = None
    kappa: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None
    theta0: float | None = None
    edelay: float | None = None
    edelay_source: Literal["global", "calibrated", "manual"] | None = None
    edelay_persistable: bool | None = None


class MuxOnetoneSummary(BaseModel):
    """Native acquisition order and reason that scalar resonator writeback is unavailable."""

    tones: list[MuxOnetoneTone]
    resonators_diagnostic: str | None


@dataclass(frozen=True)
class ToneFitSuccess:
    """One active original tone's fit, never an active-column ordinal.

    index is the original tones[index]; label is display-only and may repeat.
    fit contains MHz frequency/linewidth, rad theta0, inverse-MHz delay and
    typed diagnostic fit_quality on this tone's trimmed absolute-frequency IQ.
    edelay_source records global search, captured route prior, or manual seed.
    edelay_persistable follows seeded-or-nonuniform single-tone policy.
    """

    index: int
    label: str
    fit: ResonanceFit
    edelay_source: Literal["global", "calibrated", "manual"]
    edelay_persistable: bool


@dataclass(frozen=True)
class ToneFitFailure:
    """An active original index and display label with its numerical/prior reason."""

    index: int
    label: str
    reason: str


@dataclass
class MuxOnetoneAnalyzeResult(AnalyzeResultBase):
    """Active outcomes in column order plus bindings from the analysis capture.

    resonators contains the complete N bindings, or None when absent/mismatched.
    resonators_diagnostic explains why parameter proposals are unavailable. It does
    not suppress successful module proposals. No live context is retained.
    """

    outcomes: tuple[ToneFitSuccess | ToneFitFailure, ...]
    resonators: tuple[str, ...] | None
    resonators_diagnostic: str | None

    def summary(self) -> MuxOnetoneSummary:
        """Return native fits/failures and the captured binding diagnostic."""
        tones: list[MuxOnetoneTone] = []
        for outcome in self.outcomes:
            if isinstance(outcome, ToneFitFailure):
                row = MuxOnetoneTone(
                    index=outcome.index,
                    label=outcome.label,
                    status="failed",
                    reason=outcome.reason,
                )
            else:
                row = MuxOnetoneTone(
                    index=outcome.index,
                    label=outcome.label,
                    status="success",
                    freq=outcome.fit.freq,
                    kappa=outcome.fit.fwhm,
                    fit_quality=outcome.fit.fit_quality,
                    theta0=outcome.fit.theta0,
                    edelay=outcome.fit.edelay,
                    edelay_source=outcome.edelay_source,
                    edelay_persistable=outcome.edelay_persistable,
                )
            tones.append(row)
        return MuxOnetoneSummary(
            tones=tones, resonators_diagnostic=self.resonators_diagnostic
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize only fields present in the native success/failure outcome."""
        return self.summary().model_dump(mode="json", exclude_unset=True)


def _validate_source(source: MuxOnetoneRunResult) -> MuxOnetoneCfg:
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Mux FIT requires the saved complete cfg; current form is not a substitute"
        )
    indices = cfg.participants
    if (
        data.tone_indices.ndim != 1
        or data.tone_indices.dtype.kind not in "iu"
        or not np.array_equal(data.tone_indices, indices)
    ):
        raise ValueError(
            "tone_indices must match saved cfg participants in original-index column order"
        )
    if data.dfs.ndim != 1 or data.signals.shape != (data.dfs.size, len(indices)):
        raise ValueError(
            "Mux FIT requires dfs (Ndf,), tone_indices (K,), signals (Ndf,K)"
        )
    if (
        len(data.dfs) < 6
        or not np.all(np.isfinite(data.dfs))
        or not (np.all(np.diff(data.dfs) > 0) or np.all(np.diff(data.dfs) < 0))
    ):
        raise ValueError(
            "Mux FIT nominal dfs must be finite, strictly monotonic, with at least six points"
        )
    return cfg


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _validate_options(params: OneToneFreqAnalyzeParams) -> None:
    if params.model_type not in ("hm", "t", "auto"):
        raise ValueError(f"Invalid model type: {params.model_type!r}")
    if not _is_bool(params.fit_bg_amp_slope) or not _is_bool(
        params.fit_bg_phase_curvature
    ):
        raise ValueError("Background fit options must be bool")
    if params.edelay is not None and finite_real(params.edelay) is None:
        raise ValueError("edelay must be a finite real scalar or None")


def _resonators(
    parameters: ParameterContext | None, count: int
) -> tuple[tuple[str, ...] | None, str | None]:
    if parameters is None:
        return None, "No Run/Load parameter capture; resonator parameters unavailable"
    lookup = parameters.lookup("$mux.resonators")
    raw = lookup.value
    if not lookup.present or raw is None:
        raise ValueError(
            "$mux.resonators is missing; expected an ordered list of component names"
        )
    if not isinstance(raw, list):
        raise ValueError("$mux.resonators must be an ordered list of component names")
    resonators: list[str] = []
    for resonator in raw:
        if not isinstance(resonator, str):
            raise ValueError(
                "$mux.resonators must be an ordered list of component names"
            )
        resonators.append(resonator)
    if len(raw) != count:
        return (
            None,
            f"$mux.resonators length {len(raw)} does not match complete readout tone count N={count}; parameter proposals omitted",
        )
    return tuple(resonators), None


def _fit_tone(
    source: MuxOnetoneRunResult,
    params: OneToneFreqAnalyzeParams,
    column: int,
    prior: float | None,
    plots: Plots,
) -> ToneFitSuccess | ToneFitFailure:
    # The entry point already validated cfg, indices, columns and shared options.
    cfg = source.cfg
    if cfg is None:
        raise ValueError("Mux FIT requires saved cfg")
    index = int(source.result.tone_indices[column])
    tone = cfg.readout.tones[index]
    freqs = tone.freq + source.result.dfs
    fig, _ = plots.subplots(f"fit_{index}", figsize=(9, 8))
    try:
        seed = prior
        delay_source: Literal["global", "calibrated", "manual"] = (
            "calibrated" if prior is not None else "global"
        )
        if params.edelay is not None:
            seed, delay_source = params.edelay, "manual"
        fit = fit_resonance(
            freqs,
            source.result.signals[:, column],
            ResonanceOptions(
                model_type=params.model_type,
                fit_bg_amp_slope=params.fit_bg_amp_slope,
                fit_bg_phase_curvature=params.fit_bg_phase_curvature,
                edelay_branch_seed=seed,
                edelay_max_search_radius=100.0 if seed is None else None,
            ),
            fig,
        )
        if not all(
            np.isfinite(value) for value in (fit.freq, fit.fwhm, fit.theta0, fit.edelay)
        ):
            raise ValueError("Resonance fit produced nonfinite calibration scalars")
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        reason = str(exc)
        fig.clear()
        axes = fig.subplots()
        axes.plot(freqs[1:-1], np.abs(source.result.signals[1:-1, column]))
        axes.set(xlabel="Nominal frequency (MHz)", ylabel="Magnitude (a.u.)")
        axes.text(0.02, 0.98, reason, transform=axes.transAxes, va="top", wrap=True)
        fig.suptitle(f"tone {index}: {tone.label or f'tone {index}'} — FIT failed")
        return ToneFitFailure(index, tone.label, reason)
    fig.suptitle(f"tone {index}: {tone.label or f'tone {index}'}")
    steps = np.abs(np.diff(freqs[1:-1]))
    nonuniform = not bool(np.allclose(steps, np.median(steps), rtol=1e-8, atol=0.0))
    return ToneFitSuccess(
        index, tone.label, fit, delay_source, seed is not None or nonuniform
    )


class MuxOnetoneAdapter(
    BaseAdapter[
        MuxOnetoneCfg,
        MuxOnetoneRunResult,
        MuxOnetoneAnalyzeResult,
        OneToneFreqAnalyzeParams,
    ]
):
    """Scan selected resonators, fit their spectra, and propose frequency updates."""

    exp_cls = MuxOnetoneExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
        """Start from the mux group's readout, or an editable blank readout."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MuxOnetoneCfg, *, context: RunContext
    ) -> MuxOnetoneRunResult:
        """Measure selected tones and keep the full base readout with their spectra.

        cfg supplies the tones and offset sweep; context supplies hardware, plots
        and cancellation. req is unused. Unmeasured points remain NaN; check the
        context cancel signal for acquisition errors. Invalid settings raise.
        """
        del req
        return RunRecord(cfg=cfg, result=MuxOnetoneExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxOnetoneRunResult, OneToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxOnetoneAnalyzeResult:
        """Fit each selected resonator at its saved base frequency plus MHz offset.

        req supplies measured IQ, the saved cfg, common fit options and captured
        resonator calibrations. plots receives a fit_i figure per original tone i.
        Return frequency, linewidth, phase and electrical delay, or a failure
        reason for each tone. Missing cfg, inconsistent data, invalid options or
        source errors raise; one tone's numerical or delay-prior failure does not
        prevent fitting the others.
        """
        # 1. Match measured columns to the saved tones and captured resonators.
        cfg = _validate_source(req.run_result)
        params = req.analyze_params
        _validate_options(params)
        resonators, diagnostic = _resonators(req.parameters, len(cfg.readout.tones))

        # 2. Fit independently, using only a delay prior for the same generator/ADC.
        outcomes: list[ToneFitSuccess | ToneFitFailure] = []
        for column, index in enumerate(req.run_result.result.tone_indices):
            prior = None
            if (
                resonators is not None
                and req.parameters is not None
                and params.edelay is None
            ):
                tone = cfg.readout.tones[index]
                raw = req.parameters.lookup(f"{resonators[index]}.ext.edelay")
                if raw.present:
                    prior = route_matched_delay(
                        raw.value, (cfg.readout.gen_ch, tone.ro_ch)
                    )
            outcomes.append(_fit_tone(req.run_result, params, column, prior, plots))
        return MuxOnetoneAnalyzeResult(tuple(outcomes), resonators, diagnostic)

    def get_writeback_items(
        self, req: WritebackRequest[MuxOnetoneRunResult, MuxOnetoneAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose fitted frequencies without changing inactive or failed tones.

        req supplies the saved run and its analysis. Return a full readout module
        for $mux.readout.readout_rf, a following readout slot row when capture
        is available, and separate resonator calibration rows when bindings are
        available. The slot points to the captured module root; alias and source
        failures propagate. All-failed fits return no proposals; missing or
        inconsistent saved cfg raises. Proposals do not write until Apply.
        """
        # 1. Only successful fits can propose changes.
        cfg = _validate_source(req.run_result)
        answer = req.analyze_result
        successes = [
            outcome
            for outcome in answer.outcomes
            if isinstance(outcome, ToneFitSuccess)
        ]
        if not successes:
            return []

        # 2. Replace successful frequencies in the complete readout, not a subset.
        readout = cfg.readout
        tones = list(readout.tones)
        changes: list[str] = []
        for outcome in answer.outcomes:
            if isinstance(outcome, ToneFitFailure):
                changes.append(
                    f"tone {outcome.index} ({outcome.label}): failed: {outcome.reason}; frequency unchanged"
                )
            else:
                old = tones[outcome.index]
                tones[outcome.index] = old.with_updates(freq=outcome.fit.freq)
                changes.append(
                    f"tone {outcome.index} ({outcome.label}): {old.freq:g} -> {outcome.fit.freq:g} MHz"
                )
        proposal = MuxReadoutSpec.model_validate(
            readout.with_updates(tones=tones).model_dump()
        )
        items: list[WritebackItem] = [
            ModuleWriteback(
                target_name="$mux.readout.readout_rf",
                description="; ".join(changes),
                proposal=proposal,
                role_id="readout",
            )
        ]

        if req.ctx.parameters is not None:
            items.append(
                ParameterWriteback(
                    target_name="$mux.module.readout",
                    description="Select fitted mux readout for downstream experiments",
                    proposed_value=req.ctx.parameters.resolve_path(
                        "$mux.readout.readout_rf"
                    ),
                )
            )

        # 3. Map scalar calibrations by original tone index to captured resonators.
        if answer.resonators is not None:
            for outcome in successes:
                resonator, fit = answer.resonators[outcome.index], outcome.fit
                for suffix, description, value in (
                    ("freq", "Resonator frequency (MHz)", fit.freq),
                    ("kappa", "Resonator linewidth FWHM (MHz)", fit.fwhm),
                    ("ext.theta0", "Resonator circle phase offset (rad)", fit.theta0),
                ):
                    items.append(
                        ParameterWriteback(
                            target_name=f"{resonator}.{suffix}",
                            description=description,
                            proposed_value=value,
                        )
                    )
                if outcome.edelay_persistable:
                    items.append(
                        ParameterWriteback(
                            target_name=f"{resonator}.ext.edelay",
                            description="Route-qualified resonator electrical-delay calibration",
                            proposed_value={
                                "edelay": fit.edelay,
                                "res_ch": readout.gen_ch,
                                "ro_ch": readout.tones[outcome.index].ro_ch,
                            },
                        )
                    )
        return items
