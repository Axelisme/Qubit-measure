"""Conditional Ramsey acquisition with thermal spectator I/Xπ branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick import QickConfig
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse, Reset, SweepCfg, TimelineBuilder, TimelineProgramV2, shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._readout import JointRoles
from .._sums import readout_axes, sigma, terminal_counts, validate_classifiers
from .._timeline import emit_readout_prefix
from ._analysis import IdleBranchFit, IdleZZAnalysis, fit_branch, plot_analysis


class IdleZZCfg(ConfigBase):
    """Target f1/f2 Ramsey with nominal spectator I/Xπ branches and optional reset.

    target selects the role doing π/2→delay→phase projection. The other role
    receives I (branch 0) or Xπ (branch 1). reset=None emits no reset.
    roles are fixed capture names and original tone positions. Delay is µs,
    pulse frequency MHz, phase degrees. detune (MHz) is a required operator input:
    second π/2 phase = first.phase + 360*detune*actual_delay. Its magnitude must
    comfortably exceed expected |ζ| plus qubit frequency error so both nominal
    branches remain on its sign-selected side. No automatic detune adjustment
    occurs. dc_device=None leaves all sources untouched;
    otherwise only the named source is swept, with no compensation. Rounds
    accumulate class counts on the acquisition side. F1/F2 require circle
    classifiers with radii from GE Post; other tones remain in the full readout.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(default=1, gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
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
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before every shot."),
    ] = None
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="填入 F1 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="填入 F2 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain a.u. within [-1, 1] 與 timing us。",
        ),
    ]
    roles: JointRoles
    target: Literal["f1", "f2"] = "f1"
    detune: Annotated[
        float,
        Field(allow_inf_nan=False),
        FieldMeta(
            label="Detune",
            unit="MHz",
            description="填入 signed artificial phase ramp。先做 twotone frequency 與 T2 Ramsey；magnitude 必須大於預期 |ζ| 加 frequency error。",
        ),
    ]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    dc_device: Annotated[
        DeviceName | None,
        FieldMeta(
            label="Coupler DC source",
            description="Direct device name; None disables the DC sweep.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"], FieldMeta(label="DC source working unit")
    ] = "native"
    dc_values: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC source values",
            description="Inactive when source is None; no conversion or F1/F2 compensation.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require complete tones and consistent independent calibrated drive lines."""
        m = self
        if m.x180_f1.ch == m.x180_f2.ch:
            raise ValueError("F1/F2 require independent charge generators")
        if m.x90_f1.ch != m.x180_f1.ch or m.x90_f2.ch != m.x180_f2.ch:
            raise ValueError("Each role's Xπ and Xπ/2 must use the same generator")
        validate_classifiers(m.readout, self.roles)
        delays = sweep2array(self.delay)
        if not np.all(np.isfinite(delays)) or np.any(delays < 0):
            raise ValueError("Ramsey delay must be finite and nonnegative")
        if self.dc_device is not None:
            if not self.dc_device.strip():
                raise ValueError("DC source must name a connected device or be None")
            if not np.all(np.isfinite(sweep2array(self.dc_values))):
                raise ValueError("DC source coordinates must be finite")
        return self


@dataclass(frozen=True)
class IdleZZResult:
    """Unnormalized float64 sums (D,2,T,3,3), terminal F1/F2 g/e/other.

    dc_values is the D-point device coordinate, or [0] when DC is disabled.
    branches is int64 [0,1] for spectator I/Xπ. delays is actual quantized µs.
    Rounds accumulate without averaging. Unmeasured points stay NaN; completed
    rounds survive cancellation. No individual shots or IQ are retained.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


def build_program(
    cfg: IdleZZCfg,
    soccfg: QickConfig,
    branch: int,
    delay: float,
    *,
    heralded: bool,
) -> TimelineProgramV2:
    """Compile spectator I/Xπ (branch 0/1) and target Ramsey at delay in µs.

    Emit optional reset, optional herald, spectator preparation, target π/2,
    delay, phase-ramped projection and terminal. Fresh modules belong to each
    build; herald and terminal use the same mux object. No acquisition occurs.
    """
    if branch not in (0, 1):
        raise ValueError("branch must be 0 (I) or 1 (Xπ)")
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Ramsey delay must be finite and nonnegative")
    validate_classifiers(cfg.readout, cfg.roles)
    builder = TimelineBuilder(
        soccfg, reps=cfg.reps,
        initial_delay=cfg.initial_delay, relax_delay=cfg.relax_delay,
    )
    readout = emit_readout_prefix(builder, cfg.readout, cfg.reset, heralded=heralded)
    spectator = cfg.x180_f2 if cfg.target == "f1" else cfg.x180_f1
    pi2 = cfg.x90_f1 if cfg.target == "f1" else cfg.x90_f2
    if branch:
        builder.emit(Pulse("spectator_x180", spectator))
    builder.emit(Pulse("target_pi2", pi2))
    builder.delay_timeline(delay)
    builder.emit(Pulse(
        "projection", pi2.with_updates(phase=pi2.phase + 360 * cfg.detune * delay),
    ))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: IdleZZCfg, *, context: RunContext, heralded: bool,
) -> tuple[
    NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return DC, branch, actual delay axes and raw 9/81-class summed counts.

    Host scans DC/branch/delay; acquisition accumulates rounds. Live traces are
    target sigma per DC/branch, selecting herald (g,g) when requested. Only the
    named optional DC source is set, without compensation or restoration.
    Schedule owns the cfg copy and cancellation. The input cfg is unchanged.
    """
    validate_classifiers(config.readout, config.roles)
    delays = sweep2array(config.delay, "time", {"soccfg": context.soccfg})
    dc_values = np.array([0.0]) if config.dc_device is None else sweep2array(config.dc_values)
    viewer = live_line(
        context.plots, "measurement", "Delay (us)",
        "Target ⟨σ⟩" if not heralded else "Target ⟨σ⟩ | herald (g,g)",
        num_lines=len(dc_values) * 2,
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        traces = sigma(terminal_counts(sums, heralded=heralded), config.target)
        viewer.update(delays, traces.reshape(-1, len(delays)))

    classes = (3, 3, 3, 3) if heralded else (3, 3)
    buffer = SignalBuffer(
        (len(dc_values), 2, len(delays), *classes), dtype=np.float64,
        on_update=update_view,
    )
    with Schedule(config, buffer, scope=context.scope) as sched:
        cfg = sched.cfg
        for dc_value, dc_step in sched.scan("dc", dc_values.tolist()):
            if cfg.dc_device is not None:
                set_flux(
                    context.devices[cfg.dc_device], dc_value,
                    unit=cfg.dc_unit, stop=context.scope.stop,
                )
            for branch, branch_step in dc_step.scan("branch", [0, 1]):
                for delay, step in branch_step.scan("delay", delays.tolist()):
                    prog = build_program(
                        step.cfg, context.soccfg, branch, delay, heralded=heralded,
                    )
                    step.run_acquire_summed(
                        prog, context.soc,
                        processor=shot_sums(prog, readout_axes(cfg.roles, heralded=heralded)),
                        rounds=cfg.rounds, options=RunOptions(),
                    )
    return dc_values, np.arange(2, dtype=np.int64), delays, buffer.array


def analyze_counts(
    cfg: IdleZZCfg | None,
    result: IdleZZResult,
    *,
    complete: NDArray[np.bool_],
    plots: Plots,
) -> IdleZZAnalysis:
    """Fit saved target sigma for each complete branch; return f_Xπ−f_I in MHz.

    result holds terminal (D,2,T,3,3) counts, already herald-selected if needed.
    complete is (D,2,T), established from all saved classes before selection.
    Zero accepted shots cannot fit. Saved cfg is required; no live library is
    consulted. Free-phase decay cosine, detune-sign ambiguity, covariance errors
    and the six regular-delay requirement retain their existing definitions.
    """
    if cfg is None:
        raise ValueError("Idle ZZ analysis requires saved cfg, not current Form/library")
    dc_values = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
    ndelay = cfg.delay.expts
    if (
        not np.array_equal(result.dc_values, dc_values)
        or not np.array_equal(result.branches, np.arange(2))
        or result.delays.shape != (ndelay,)
        or not np.all(np.isfinite(result.delays))
        or np.any(result.delays < 0)
        or len(np.unique(result.delays)) != ndelay
        or result.sums.shape != (len(dc_values), 2, ndelay, 3, 3)
    ):
        raise ValueError("Idle ZZ requires saved (DC,2,delay,3,3) count axes")
    traces = sigma(result.sums, cfg.target)
    shots = result.sums.sum(axis=(-2, -1))
    fits = []
    zetas: list[float | None] = []
    errors: list[float | None] = []
    reasons: list[str | None] = []
    for dc_index in range(len(dc_values)):
        pair = [
            fit_branch(
                IdleBranchFit(
                    dc_index, branch, int(np.nansum(shots[dc_index, branch])),
                    bool(np.all(complete[dc_index, branch])), traces[dc_index, branch],
                ),
                delays=result.delays, detune=cfg.detune,
            )
            for branch in range(2)
        ]
        fits.extend(pair)
        a, b = pair
        if (
            a.signed_mhz is None or b.signed_mhz is None
            or a.error_mhz is None or b.error_mhz is None
        ):
            zetas.append(None)
            errors.append(None)
            reasons.append("; ".join(
                f"{label}: {fit.reason}"
                for label, fit in zip(("I", "Xπ"), pair, strict=True) if fit.reason
            ))
        else:
            zetas.append(b.signed_mhz - a.signed_mhz)
            errors.append(float(np.hypot(a.error_mhz, b.error_mhz)))
            reasons.append(None)
    answer = IdleZZAnalysis(tuple(fits), tuple(zetas), tuple(errors), tuple(reasons))
    dc_label = (
        "DC not swept (inactive coordinate)" if cfg.dc_device is None
        else f"{cfg.dc_device} ({cfg.dc_unit})"
    )
    plot_analysis(
        answer, result.delays, result.dc_values,
        title=f"Target {cfg.target}, detune={cfg.detune:g} MHz",
        dc_label=dc_label, plots=plots,
    )
    return answer


def complete_points(
    cfg: IdleZZCfg | None, sums: NDArray[np.float64], *, heralded: bool,
) -> NDArray[np.bool_]:
    """Validate raw counts and return points with all requested rounds acquired.

    All-NaN points are unmeasured. Finite nonnegative points may contain partial
    round accumulation. Completion uses all 9/81 classes, never herald acceptance.
    Saved cfg is mandatory; malformed counts raise ValueError.
    """
    if cfg is None:
        raise ValueError("Idle ZZ analysis requires saved cfg")
    nclass = 4 if heralded else 2
    if sums.ndim != 3 + nclass or sums.shape[-nclass:] != (3,) * nclass:
        raise ValueError("Idle ZZ requires DC × branch × delay × 9/81 class counts")
    axes = tuple(range(-nclass, 0))
    finite = np.all(np.isfinite(sums) & (sums >= 0), axis=axes)
    missing = np.all(np.isnan(sums), axis=axes)
    if not np.all(finite | missing):
        raise ValueError("Idle ZZ counts must be nonnegative finite or entirely NaN")
    return finite & (sums.sum(axis=axes) == cfg.reps * cfg.rounds)


def restore_result(data: GroupedLoadData[IdleZZCfg]) -> IdleZZResult:
    """Restore summed counts; old raw-IQ files missing sums raise ValueError."""
    sums = data.variable("sums")
    return IdleZZResult(
        sums.axes[4].astype(np.float64), sums.axes[3].astype(np.int64),
        sums.axes[2].astype(np.float64), sums.z.astype(np.float64),
    )


class IdleZZExp(Experiment[IdleZZResult, IdleZZCfg]):
    """Compare target sigma Ramsey fringes for nominal spectator I/Xπ branches."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZResult, IdleZZCfg]] = GroupedAxesSpec(
        variables=(VariableSpec(
            variable="sums",
            axes=(
                VariableAxisSpec.generated_arange("terminal_f2", ""),
                VariableAxisSpec.generated_arange("terminal_f1", ""),
                VariableAxisSpec("Delay", "s", field_name="delays", scale=1e-6),
                VariableAxisSpec("Spectator branch", "", field_name="branches", dtype=np.int64),
                VariableAxisSpec("DC source value (inactive if disabled)", "a.u.", field_name="dc_values"),
            ),
            z=VariableZSpec(field_name="sums", label="Class counts", unit="", dtype=np.float64),
        ),),
        result_type=IdleZZResult, cfg_type=IdleZZCfg, cfg_schema_version="4.0",
        tag="two_qubit/idle_zz", result_builder=restore_result,
    )

    def run(self, config: IdleZZCfg, *, context: RunContext) -> IdleZZResult:
        """Acquire optional reset → spectator I/Xπ → target Ramsey → terminal.

        Return raw (DC,2,delay,3,3) counts. Rounds accumulate without averaging;
        completed counts survive cancellation. No calibration is written.
        """
        return IdleZZResult(*acquire_counts(config, context=context, heralded=False))

    def analyze(
        self, source: RunRecord[IdleZZCfg, IdleZZResult], options: None, *, plots: Plots,
    ) -> IdleZZAnalysis:
        """Fit complete saved sigma traces, preserving unavailable ζ with reasons.

        Operator must ensure both branch frequencies stay on the detune-selected
        side. Finite fits do not prove a pure ZZ Hamiltonian or exclude thermal
        mixtures. No automatic detune/DC adjustment or writeback occurs.
        """
        del options
        return analyze_counts(
            source.cfg, source.result,
            complete=complete_points(source.cfg, source.result.sums, heralded=False),
            plots=plots,
        )
