"""Fixed-tone mux spectroscopy with a common nominal host-side offset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt, ValidationInfo, field_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, Reset, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2._support import live_line
from zcu_lab.v2._support.resonator import (
    ResonanceFit,
    ResonanceOptions,
    finite_real,
    fit_resonance,
)


@dataclass(frozen=True)
class MuxOnetoneAnalyzeOptions:
    """Fit controls shared by every active spectrum.

    model_type selects hanger (hm, default), transmission (t), or auto.
    fit_bg_amp_slope defaults True; fit_bg_phase_curvature defaults False.
    edelay is an optional finite inverse-MHz branch seed, not a fixed delay.
    None uses each tone's route prior, then a 100 us global search if absent.
    Invalid models, non-bool switches and nonfinite/non-real seeds raise ValueError.
    """

    model_type: Literal["hm", "t", "auto"] = "hm"
    fit_bg_amp_slope: bool = True
    fit_bg_phase_curvature: bool = False
    edelay: float | None = None

    def __post_init__(self) -> None:
        if self.model_type not in ("hm", "t", "auto"):
            raise ValueError(f"Invalid model type: {self.model_type!r}")
        if (
            type(self.fit_bg_amp_slope) is not bool
            or type(self.fit_bg_phase_curvature) is not bool
        ):
            raise ValueError("Background fit options must be bool")
        if self.edelay is not None and finite_real(self.edelay) is None:
            raise ValueError("edelay must be a finite real scalar or None")


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
    """An active original index and display label with its numerical fit reason."""

    index: int
    label: str
    reason: str


@dataclass
class MuxOnetoneAnalysis:
    """Active fits or numerical failures in saved participants/column order.

    outcomes retains each original index and label, never active ordinals.
    No capture, writeback destination, or live context is retained.
    """

    outcomes: tuple[ToneFitSuccess | ToneFitFailure, ...]


@dataclass(frozen=True)
class MuxOnetoneResult:
    """Averaged active spectra in selection order, including partial runs.

    dfs is float64 (Ndf,) of nominal common offsets in MHz, not actual DDS
    frequencies. tone_indices is int64 (K,) of original indices in 0..N-1;
    each index i refers to saved cfg.readout.tones[i], not its ADC
    channel. K is the active count and N the full cfg tone count.
    signals is complex128 (Ndf, K) of averaged I+jQ in a.u., retaining the
    program's ADC-length normalization. Unwritten entries remain NaN; acquired
    rows survive cancellation or failure.
    """

    dfs: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class MuxFrequencyOffsetCfg(ConfigBase):
    """Common nominal frequency-offset scan, without hardware-grid rounding.

    df is SweepCfg(start, stop, expts, step) in MHz, shared by every tone.
    start, stop and step must be finite. SweepCfg requires positive expts and
    stop = start + step * (expts - 1); one point requires start = stop, step = 0.
    Invalid values raise Pydantic validation errors at construction/assignment.
    """

    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]

    @field_validator("df")
    @classmethod
    def _validate_finite_df(cls, value: SweepCfg) -> SweepCfg:
        if not all(math.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("df start, stop and step must be finite MHz values")
        return value


class MuxOnetoneCfg(ConfigBase):
    """Scan every selected readout tone by the same frequency offset.

    readout supplies the base mux readout; reset optionally runs before it.
    Both fields are top-level. df supplies nominal offsets in MHz;
    each point uses base_freq[i] + df with no cumulative
    update. Each point's program quantizes these frequencies to the hardware grid.

    Explicit reps (default 1) is repetitions per round; rounds (default 1) is
    software acquisition rounds per offset. initial_delay and relax_delay are
    program delays in us (both default 1). No external devices are required.
    participants selects original readout indices in execution/column order.
    Lists must be nonempty, unique integers in 0..N-1;
    bool, floating point and string indices are rejected without coercion.
    Saved data retains the full base readout, including inactive tones.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Readout",
            description=(
                "Without a slot, mixer_freq comes from $mux.mixer_freq "
                "independently of kappa; missing mixer stays required and blank. "
                "ro_length=1.0 us; trig_offset=$mux.time_of_fly "
                "+5/(2*pi*min group kappa); length=trig_offset+ro_length. "
                "Missing group flight time or any kappa leaves trigger and length "
                "required and blank. "
                "Candidates are not device-confirmed (未經裝置確認); "
                "capability checks run at Run. With a slot, keep its mixer and "
                "ADC window, add ringup to its trigger and extend pulse coverage."
            ),
        ),
    ]
    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]

    @field_validator("df")
    @classmethod
    def _validate_finite_df(cls, value: SweepCfg) -> SweepCfg:
        if not all(math.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("df start, stop and step must be finite MHz values")
        return value

    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            label="Participants",
            description="Original tone indices in execution/column order.",
        ),
    ]

    @field_validator("participants")
    @classmethod
    def validate_participants(cls, value: list[int], info: ValidationInfo) -> list[int]:
        """Reject empty, duplicate or out-of-range selections; preserve order.

        Pydantic validates strict integer elements before this check. Range is
        checked against the full readout when its own validation succeeded.
        """
        if not value or len(set(value)) != len(value):
            raise ValueError("participants must be nonempty and contain unique indices")
        readout = info.data.get("readout")
        if isinstance(readout, MuxReadout.Cfg) and any(
            index < 0 or index >= len(readout.tones) for index in value
        ):
            raise ValueError(
                "participants indices must be in 0..N-1 of the full readout"
            )
        return value


def _mux_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder deep-copies modules; only the program-owned instance is initialized.
    for module in prog.modules:
        if module.name == "readout" and isinstance(module, MuxReadout):
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("Mux program is missing its named readout MuxReadout")


def validate_analysis_source(
    source: RunRecord[MuxOnetoneCfg, MuxOnetoneResult],
) -> MuxOnetoneCfg:
    """Return saved cfg after checking original indices and the shared scan grid.

    Missing cfg, mismatched selection/shape, or fewer than six finite strictly
    monotonic MHz offsets raise ValueError. IQ failures remain per-tone.
    GUI callers may use this before resolving capture-dependent inputs.
    """
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


def _fit_tone(
    source: RunRecord[MuxOnetoneCfg, MuxOnetoneResult],
    params: MuxOnetoneAnalyzeOptions,
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


class MuxOnetoneExp(Experiment[MuxOnetoneResult, MuxOnetoneCfg]):
    """Measure several resonator spectra with a common frequency-offset scan.

    Each point applies the optional reset, then plays and acquires the selected
    mux readout tones together. Results contain averaged complex IQ with shape
    (Ndf, K), where K is the selected tone count. Saved axes retain original tone
    indices and nominal offsets (MHz in memory, Hz on disk).
    """

    AXES_SPEC: ClassVar[AxesSpec[MuxOnetoneResult, MuxOnetoneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
            AxesSpec.Axis("dfs", "Frequency offset", "Hz", scale=1e6),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=MuxOnetoneResult,
        cfg_type=MuxOnetoneCfg,
        cfg_schema_version="3.0",
        tag="mux/onetone",
    )

    Options = MuxOnetoneAnalyzeOptions

    def analyze(
        self,
        source: RunRecord[MuxOnetoneCfg, MuxOnetoneResult],
        options: MuxOnetoneAnalyzeOptions,
        *,
        route_priors: Sequence[float | None] | None = None,
        plots: Plots,
    ) -> MuxOnetoneAnalysis:
        """Fit saved absolute-MHz spectra independently, preserving original indices.

        source must retain its complete cfg, matching participants and (Ndf,K)
        IQ. options controls the shared model/background and manual delay seed.
        route_priors supplies one parsed finite inverse-MHz seed or None per
        active column, in participants order; omitted means no route priors.
        Manual edelay overrides priors. This core never resolves capture sources.
        plots receives fit_i for each original tone i. Invalid source/priors raise
        ValueError. Numerical fit failures retain their reason without stopping
        other tones. Successful delay persists only if seeded or nonuniform.
        """
        validate_analysis_source(source)
        if route_priors is None:
            route_priors = [None] * len(source.result.tone_indices)
        if len(route_priors) != len(source.result.tone_indices) or any(
            prior is not None and finite_real(prior) is None for prior in route_priors
        ):
            raise ValueError(
                "route_priors must contain one finite real seed or None per active tone"
            )
        outcomes = tuple(
            _fit_tone(source, options, column, prior, plots)
            for column, prior in enumerate(route_priors)
        )
        return MuxOnetoneAnalysis(outcomes)

    def run(self, config: MuxOnetoneCfg, *, context: RunContext) -> MuxOnetoneResult:
        """Reset, read all selected tones together, and repeat at each offset.

        config supplies the base readout and common MHz offsets; it is unchanged.
        context supplies connected soc/soccfg, plots and cancellation. The live
        plot shows one magnitude curve per selected tone. Returned complex IQ is
        averaged over reps and rounds, in participants order; unmeasured points
        stay NaN after a stop. Call context.scope.stop.raise_if_error() to
        distinguish acquisition failure from a successful or stopped return.
        Invalid cfg or program settings raise.
        """
        # 1. Select tones without changing their original indices or base frequencies.
        cfg = deepcopy(config)
        base_readout = cfg.readout
        dfs = sweep2array(cfg.df, "none")
        indices = cfg.participants
        tone_indices = np.asarray(indices, dtype=np.int64)
        active_readout = base_readout.with_updates(
            tones=[base_readout.tones[index] for index in indices]
        )
        # 2. Give each selected spectrum its own figure and original tone identity.
        viewers = [
            live_line(
                context.plots,
                f"measurement_{index}",
                "Nominal frequency offset (MHz)",
                "Magnitude (a.u.)",
                title=f"{index}: {base_readout.tones[index].label or f'tone {index}'}",
            )
            for index in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for column, viewer in enumerate(viewers):
                viewer.update(dfs, np.abs(data[:, column]))

        signals_buffer = SignalBuffer((len(dfs), len(tone_indices)), on_update=update)

        # 3. Rebuild reset -> mux readout at each offset, always from the base tones.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for df, step in sched.scan("df", dfs.tolist()):
                readout = active_readout.with_updates(
                    tones=[
                        tone.with_updates(freq=tone.freq + df)
                        for tone in active_readout.tones
                    ]
                )
                builder = (
                    step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=step.cfg.reps,
                        initial_delay=step.cfg.initial_delay,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", step.cfg.reset)
                    .add(MuxReadout("readout", readout))
                )
                prog = builder.build()
                # Collect all selected ADCs and average reps/rounds into this scan row.
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_mux_signal, prog), rounds=cfg.rounds
                )

        # 4. Keep acquired rows and the original tone mapping, including partial scans.
        signals_buffer.trigger_update(flush=True)
        return MuxOnetoneResult(
            dfs=dfs,
            tone_indices=tone_indices,
            signals=signals_buffer.array,
        )
