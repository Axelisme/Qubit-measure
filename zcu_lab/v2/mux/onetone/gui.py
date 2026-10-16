"""Measure attachment for common-offset mux spectroscopy."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
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


@dataclass(frozen=True)
class ToneFitSuccess:
    """One active original tone's fit, never an active-column ordinal.

    index is the original tones[index]; label is display-only and may repeat.
    fit contains MHz frequency/linewidth, rad theta0 and inverse-MHz delay.
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

    members contains the complete N bindings, or None when absent/mismatched.
    members_diagnostic explains why parameter proposals are unavailable. It does
    not suppress successful module proposals. No live context is retained.
    """

    outcomes: tuple[ToneFitSuccess | ToneFitFailure, ...]
    members: tuple[str, ...] | None
    members_diagnostic: str | None

    def to_summary_dict(self) -> dict[str, object]:
        """Expose each original tone's scalars or failure and binding diagnostic."""
        tones: list[dict[str, object]] = []
        for outcome in self.outcomes:
            row: dict[str, object] = {"index": outcome.index, "label": outcome.label}
            if isinstance(outcome, ToneFitFailure):
                row.update(status="failed", reason=outcome.reason)
            else:
                row.update(
                    status="success",
                    freq=outcome.fit.freq,
                    kappa=outcome.fit.fwhm,
                    theta0=outcome.fit.theta0,
                    edelay=outcome.fit.edelay,
                    edelay_source=outcome.edelay_source,
                    edelay_persistable=outcome.edelay_persistable,
                )
            tones.append(row)
        return {"tones": tones, "members_diagnostic": self.members_diagnostic}


def _validate_source(source: MuxOnetoneRunResult) -> MuxOnetoneCfg:
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Mux FIT requires the saved complete cfg; current form is not a substitute"
        )
    indices = (
        cfg.active_tones
        if cfg.active_tones is not None
        else list(range(len(cfg.modules.readout.tones)))
    )
    if (
        data.tone_indices.ndim != 1
        or data.tone_indices.dtype.kind not in "iu"
        or not np.array_equal(data.tone_indices, indices)
    ):
        raise ValueError(
            "tone_indices must match saved cfg active_tones in original-index column order"
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
    if params.edelay_mode not in ("auto", "calibrated", "manual"):
        raise ValueError(f"Invalid electrical-delay mode: {params.edelay_mode!r}")
    if params.edelay_mode == "manual" and finite_real(params.manual_edelay) is None:
        raise ValueError("manual_edelay must be a finite real scalar in manual mode")
    if params.edelay_mode == "auto":
        radius = finite_real(params.max_edelay_search_radius)
        if radius is None or radius <= 0:
            raise ValueError("max_edelay_search_radius must be positive and finite")


def _members(
    parameters: ParameterContext | None, count: int
) -> tuple[tuple[str, ...] | None, str | None]:
    if parameters is None:
        return None, "No Run/Load parameter capture; member parameters unavailable"
    lookup = parameters.lookup("$mux.members")
    raw = lookup.value
    if not lookup.present or raw is None:
        raise ValueError(
            "$mux.members is missing; expected an ordered list of component names"
        )
    if not isinstance(raw, list):
        raise ValueError("$mux.members must be an ordered list of component names")
    members: list[str] = []
    for member in raw:
        if not isinstance(member, str):
            raise ValueError("$mux.members must be an ordered list of component names")
        members.append(member)
    if len(raw) != count:
        return (
            None,
            f"$mux.members length {len(raw)} does not match complete readout tone count N={count}; parameter proposals omitted",
        )
    return tuple(members), None


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
    tone = cfg.modules.readout.tones[index]
    freqs = tone.freq + source.result.dfs
    fig, _ = plots.subplots(f"fit_{index}", figsize=(9, 8))
    try:
        seed = prior
        delay_source: Literal["global", "calibrated", "manual"] = (
            "calibrated" if prior is not None else "global"
        )
        if params.edelay_mode == "manual":
            seed, delay_source = finite_real(params.manual_edelay), "manual"
        elif params.edelay_mode == "calibrated" and prior is None:
            raise ValueError(
                "calibrated edelay mode requires a finite route-matched member.ext.edelay prior"
            )
        fit = fit_resonance(
            freqs,
            source.result.signals[:, column],
            ResonanceOptions(
                model_type=params.model_type,
                fit_bg_amp_slope=params.fit_bg_amp_slope,
                fit_bg_phase_curvature=params.fit_bg_phase_curvature,
                edelay_branch_seed=seed,
                edelay_max_search_radius=params.max_edelay_search_radius
                if seed is None
                else None,
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
    """Run, fit and propose updates for active columns of a complete mux readout."""

    experiment_tag: ClassVar[str | None] = MuxOnetoneExp.persistence_tag()
    exp_cls = MuxOnetoneExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
        """Build the mux draft from the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MuxOnetoneCfg, *, context: RunContext
    ) -> MuxOnetoneRunResult:
        """Acquire active columns; retain complete cfg and partial data.

        req is unused. Context owns hardware/plots/cancellation; caller checks its
        cancel signal for acquisition failure. Configuration errors propagate.
        """
        del req
        return RunRecord(cfg=cfg, result=MuxOnetoneExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxOnetoneRunResult, OneToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxOnetoneAnalyzeResult:
        """Fit active (Ndf,K) IQ at saved base frequencies + nominal MHz dfs.

        Reject absent cfg, inconsistent coordinates/shapes and shared options.
        Numerical/prior failures produce per-tone diagnostics and fit_i figures;
        source/alias errors propagate. Only the supplied Run/Load capture supplies
        full members and route-qualified delay priors. No form or global reads.
        """
        cfg = _validate_source(req.run_result)
        params = req.analyze_params
        _validate_options(params)
        members, diagnostic = _members(req.parameters, len(cfg.modules.readout.tones))
        outcomes: list[ToneFitSuccess | ToneFitFailure] = []
        for column, index in enumerate(req.run_result.result.tone_indices):
            prior = None
            if (
                members is not None
                and req.parameters is not None
                and params.edelay_mode != "manual"
            ):
                tone = cfg.modules.readout.tones[index]
                raw = req.parameters.lookup(f"{members[index]}.ext.edelay")
                if raw.present:
                    prior = route_matched_delay(
                        raw.value, (cfg.modules.readout.gen_ch, tone.ro_ch)
                    )
            outcomes.append(_fit_tone(req.run_result, params, column, prior, plots))
        return MuxOnetoneAnalyzeResult(tuple(outcomes), members, diagnostic)

    def get_writeback_items(
        self, req: WritebackRequest[MuxOnetoneRunResult, MuxOnetoneAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose full module and independent successful member scalar rows.

        Preserve inactive/failed tones; omit module when every fit failed. Fixed
        $mux module target is resolved by the existing draft capture, not by this
        adapter. Member destinations come only from analyze's captured full list.
        """
        cfg = _validate_source(req.run_result)
        answer = req.analyze_result
        successes = [
            outcome
            for outcome in answer.outcomes
            if isinstance(outcome, ToneFitSuccess)
        ]
        if not successes:
            return []
        readout = cfg.modules.readout
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
        if answer.members is not None:
            for outcome in successes:
                member, fit = answer.members[outcome.index], outcome.fit
                for suffix, description, value in (
                    ("freq", "Resonator frequency (MHz)", fit.freq),
                    ("kappa", "Resonator linewidth FWHM (MHz)", fit.fwhm),
                    ("ext.theta0", "Resonator circle phase offset (rad)", fit.theta0),
                ):
                    items.append(
                        ParameterWriteback(
                            target_name=f"{member}.{suffix}",
                            description=description,
                            proposed_value=value,
                        )
                    )
                if outcome.edelay_persistable:
                    items.append(
                        ParameterWriteback(
                            target_name=f"{member}.ext.edelay",
                            description="Route-qualified resonator electrical-delay calibration",
                            proposed_value={
                                "edelay": fit.edelay,
                                "res_ch": readout.gen_ch,
                                "ro_ch": readout.tones[outcome.index].ro_ch,
                            },
                        )
                    )
        return items
