"""Fixed-tone mux spectroscopy with a common nominal host-side offset."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting import HangerModel, TransmissionModel
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.resonator import (
    ResonanceFit,
    ResonanceOptions,
    fit_resonance,
)


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


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class MuxOnetoneAnalyzeOptions:
    """Fit controls shared by every active spectrum.

    model_type selects hanger (hm, default), transmission (t), or auto.
    fit_bg_amp_slope defaults True; fit_bg_phase_curvature defaults False.
    edelay_seeds supplies one finite inverse-MHz branch seed or None per active
    column, in participants order. None uses a 100 us global search.
    Invalid models, non-bool switches and nonfinite/non-real seeds raise ValueError.
    """

    edelay_seeds: tuple[
        Annotated[float, Field(strict=True, allow_inf_nan=False)] | None, ...
    ]
    model_type: Literal["hm", "t", "auto"] = "hm"
    fit_bg_amp_slope: bool = Field(default=True, strict=True)
    fit_bg_phase_curvature: bool = Field(default=False, strict=True)


@dataclass(frozen=True)
class ToneFitSuccess:
    """One active original tone's fit, never an active-column ordinal.

    index is the original tones[index]; label is display-only and may repeat.
    fit contains MHz frequency/linewidth, rad theta0, inverse-MHz delay and
    typed diagnostic fit_quality on this tone's trimmed absolute-frequency IQ.
    """

    index: int
    label: str
    fit: ResonanceFit


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


class MuxFrequencyOffsetCfg(ConfigBase):
    """Common nominal frequency-offset scan, without hardware-grid rounding.

    df is SweepCfg(start, stop, expts, step) in MHz, shared by every tone.
    start, stop and step must be finite. SweepCfg requires positive expts and
    stop = start + step * (expts - 1); one point requires start = stop, step = 0.
    Invalid values raise Pydantic validation errors at construction/assignment.
    """

    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]


class MuxOnetoneCfg(ConfigBase):
    """Scan every selected readout tone by the same frequency offset.

    readout supplies the base mux readout; reset optionally runs before it.
    Both fields are top-level. df supplies nominal offsets in MHz;
    each point uses base_freq[i] + df with no cumulative
    update. Each point's program quantizes these frequencies to the hardware grid.

    reps is the required repetitions per round; rounds defaults to one
    software acquisition round per offset. initial_delay defaults to 1 us;
    relax_delay is the required delay after each repetition in us.
    No external devices are required.
    participants selects original readout indices in execution/column order.
    Lists must be nonempty, unique integers in 0..N-1;
    bool, floating point and string indices are rejected without coercion.
    Saved data retains the full base readout, including inactive tones.
    """

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Readout",
            description=(
                "Full-group mux readout with editable, unconfirmed mixer and timing candidates when no slot exists."
            ),
        ),
    ]
    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]

    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            label="Participants",
            description="Original tone indices in execution/column order.",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @field_validator("participants")
    @classmethod
    def validate_participants(cls, value: list[int]) -> list[int]:
        """Reject empty, duplicate or negative selections; preserve order.

        Pydantic validates strict integer elements before this check.
        Run indexes the complete readout and rejects positive out-of-range values.

        """
        if not value or len(set(value)) != len(value):
            raise ValueError("participants must be nonempty and contain unique indices")
        if any(index < 0 for index in value):
            raise ValueError("participants indices must be nonnegative")
        return value


def validate_analysis_source(
    source: RunRecord[MuxOnetoneCfg, MuxOnetoneResult],
) -> MuxOnetoneCfg:
    """Return saved cfg; raise ValueError for invalid indices or shared scan grid.

    Missing cfg, mismatched selection/shape, or fewer than six finite strictly
    monotonic MHz offsets raise ValueError. IQ failures remain per-tone.
    GUI callers may use this before resolving capture-dependent inputs.

    source supplies the immutable Run/Load result and optional saved cfg. Return
    that same cfg, not a copy; do not acquire, draw or mutate the record.
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


class MuxOnetoneExp(Experiment[MuxOnetoneResult, MuxOnetoneCfg]):
    """Measure several resonator spectra with a common frequency-offset scan.

    Each point applies the optional reset, then plays and acquires the selected
    mux readout tones together. Results contain averaged complex IQ with shape
    (Ndf, K), where K is the selected tone count. Saved axes retain original tone
    indices and nominal offsets (MHz in memory, Hz on disk).
    """

    AXES_SPEC: ClassVar[AxesSpec[MuxOnetoneResult, MuxOnetoneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", label="Tone", unit="", dtype=np.int64),
            AxesSpec.Axis("dfs", label="Frequency offset", unit="Hz", scale=1e6),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=MuxOnetoneResult,
        cfg_type=MuxOnetoneCfg,
        cfg_schema_version="3.0",
        tag="mux/onetone",
    )

    Options = MuxOnetoneAnalyzeOptions

    def run(
        self,
        config: MuxOnetoneCfg,
        *,
        context: RunContext,
        on_acquire_stop: Callable[[], None] | None = None,
    ) -> MuxOnetoneResult:
        """Reset, read all selected tones together, and repeat at each offset.

        config supplies the base readout and common MHz offsets; it is unchanged.
        context supplies connected soc/soccfg, plots and cancellation. The live
        plot shows one magnitude curve per selected tone. Returned complex IQ is
        averaged over reps and rounds, in participants order; unmeasured points
        stay NaN after a stop. Acquisition, cfg and program errors propagate.
        on_acquire_stop optionally observes each externally stopped acquire once,
        after its scope and progress cleanup, before it returns. Pure data stop
        does not call it. Observer exceptions propagate without retry. The
        observer does not stop later offsets or change the persisted result.

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
        viewers = []
        for index in indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            ax.set(
                xlabel="Nominal frequency offset (MHz)",
                ylabel="Magnitude (a.u.)",
                title=f"{index}: {base_readout.tones[index].label or f'tone {index}'}",
            )
            viewers.append(live_line(ax))
            fig.show()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            return tuple(
                artist
                for row, viewer in enumerate(viewers)
                for artist in viewer.set(dfs, np.abs(data[:, row]))
            )

        signals_buffer = SignalBuffer((len(dfs), len(tone_indices)), on_update=update)

        # 3. Rebuild reset -> mux readout at each offset, always from the base tones.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for df, step in sched.scan("df", dfs.tolist()):
                readout = active_readout.with_updates(
                    tones=[
                        tone.with_updates(freq=tone.freq + df)
                        for tone in active_readout.tones
                    ]
                )
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=cfg.reps,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                readout_module = MuxReadout("readout", readout)
                builder.emit(readout_module)
                prog = builder.build()
                # Collect all selected ADCs and average reps/rounds into this scan row.
                _ = step.run_acquire(
                    prog,
                    context.soc,
                    rounds=cfg.rounds,
                    options=RunOptions(
                        converter=partial(averaged_mux_signal, prog, readout_module),
                        on_stop=on_acquire_stop,
                    ),
                )

        # 4. Keep acquired rows and the original tone mapping, including partial scans.
        signals_buffer.trigger_update(flush=True)
        return MuxOnetoneResult(
            dfs=dfs,
            tone_indices=tone_indices,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[MuxOnetoneCfg, MuxOnetoneResult],
        options: MuxOnetoneAnalyzeOptions,
        *,
        plots: Plots,
    ) -> MuxOnetoneAnalysis:
        """Fit saved absolute-MHz spectra independently, preserving original indices.

        source must retain its complete cfg, matching participants and (Ndf,K)
        IQ. options controls the shared model/background and supplies one delay
        seed or None per active column. This core never resolves capture sources
        or decides writeback. plots receives fit_i for each original tone i.
        Invalid source or seed count raises ValueError. Numerical fit failures
        retain their reason without stopping other tones.

        """
        # 1. Require saved tone identities and one electrical-delay seed per active spectrum.
        cfg = validate_analysis_source(source)
        if len(options.edelay_seeds) != len(source.result.tone_indices):
            raise ValueError(
                "edelay_seeds must contain one seed or None per active tone"
            )
        outcomes: list[ToneFitSuccess | ToneFitFailure] = []
        # 2. Fit each absolute-frequency spectrum with its selected background and delay branch.
        for column, seed in enumerate(options.edelay_seeds):
            index = int(source.result.tone_indices[column])
            tone = cfg.readout.tones[index]
            freqs = tone.freq + source.result.dfs
            signals = source.result.signals[:, column]
            fig, _ = plots.subplots(num=f"fit_{index}", figsize=(9, 8))
            try:
                fit = fit_resonance(
                    freqs,
                    signals,
                    ResonanceOptions(
                        model_type=options.model_type,
                        fit_bg_amp_slope=options.fit_bg_amp_slope,
                        fit_bg_phase_curvature=options.fit_bg_phase_curvature,
                        edelay_branch_seed=seed,
                        edelay_max_search_radius=100.0 if seed is None else None,
                    ),
                )
                params = fit.params
                fig.clear()
                if "Qc" in params:
                    HangerModel.visualize_fit(
                        freqs[1:-1],
                        signals[1:-1],
                        params,
                        figure=fig,
                        fit_bg_amp_slope=options.fit_bg_amp_slope,
                        fit_bg_phase_curvature=options.fit_bg_phase_curvature,
                    )
                else:
                    TransmissionModel.visualize_fit(
                        freqs[1:-1],
                        signals[1:-1],
                        params,
                        figure=fig,
                        fit_bg_amp_slope=options.fit_bg_amp_slope,
                        fit_bg_phase_curvature=options.fit_bg_phase_curvature,
                    )
                if not all(
                    np.isfinite(value)
                    for value in (fit.freq, fit.fwhm, fit.theta0, fit.edelay)
                ):
                    raise ValueError(
                        "Resonance fit produced nonfinite calibration scalars"
                    )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                # 3. Keep numerical failures visible as acquired magnitude without stopping other tones.
                reason = str(exc)
                fig.clear()
                axes = fig.subplots()
                axes.plot(freqs[1:-1], np.abs(signals[1:-1]))
                axes.set(xlabel="Nominal frequency (MHz)", ylabel="Magnitude (a.u.)")
                axes.text(
                    0.02, 0.98, reason, transform=axes.transAxes, va="top", wrap=True
                )
                fig.suptitle(
                    f"tone {index}: {tone.label or f'tone {index}'} — FIT failed"
                )
                outcomes.append(ToneFitFailure(index, tone.label, reason))
                continue
            fig.suptitle(f"tone {index}: {tone.label or f'tone {index}'}")
            outcomes.append(ToneFitSuccess(index, tone.label, fit))
        # 4. Keep resonance calibration or a fit failure attached to the original readout tone.
        return MuxOnetoneAnalysis(tuple(outcomes))
