"""Conditional Ramsey acquisition with thermal spectator I/Xπ branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
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
from zcu_tools.experiment.v2.utils.round_zcu import round_sweep_dict, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._readout import JointRoles
from .._sums import sigma, validate_classifiers
from ._analysis import (
    IdleBranchFit,
    complete_points,
    fit_traces,
    sigma_errors,
    summarize,
    validate_axes,
)


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
    The spectator axis is fixed [0,1] for I/Xπ. delays is actual quantized µs.
    Rounds accumulate without averaging. Unmeasured points stay NaN; completed
    rounds survive cancellation. No individual shots or IQ are retained.
    """

    dc_values: NDArray[np.float64]

    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(data: GroupedLoadData[IdleZZCfg]) -> IdleZZResult:
    """Restore summed counts; old raw-IQ files missing sums raise ValueError."""
    sums = data.variable("sums")
    return IdleZZResult(
        sums.axes[4].astype(np.float64),
        sums.axes[2].astype(np.float64),
        sums.z.astype(np.float64),
    )


class IdleZZExp(Experiment[IdleZZResult, IdleZZCfg]):
    """Compare target sigma Ramsey fringes for nominal spectator I/Xπ branches."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZResult, IdleZZCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                    VariableAxisSpec(
                        label="Delay", unit="s", field_name="delays", scale=1e-6
                    ),
                    VariableAxisSpec(
                        label="Spectator state (0=I, 1=Xπ)",
                        unit="",
                        values=(0, 1),
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        label="DC source value (inactive if disabled)",
                        unit="a.u.",
                        field_name="dc_values",
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Class counts", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=IdleZZResult,
        cfg_type=IdleZZCfg,
        cfg_schema_version="5.0",
        tag="two_qubit/idle_zz",
        result_builder=restore_result,
    )

    def run(self, config: IdleZZCfg, *, context: RunContext) -> IdleZZResult:
        """Acquire reset if set → spectator I/Xπ → target Ramsey → terminal.

        Hardware scans spectator_states and delay; DC alone stays on host
        because it changes an external instrument. Return unnormalized counts.
        """
        validate_classifiers(config.readout, config.roles)
        delay_sweep = round_sweep_dict(config.delay, "time", {"soccfg": context.soccfg})
        delays = sweep2array(delay_sweep)
        dc_values = (
            np.array([0.0])
            if config.dc_device is None
            else sweep2array(config.dc_values)
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Delay (us)", ylabel="Target ⟨σ⟩")
        viewer = live_line(ax, num_lines=len(dc_values) * 2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            return viewer.set(
                delays, sigma(sums, config.target).reshape(-1, len(delays))
            )

        buffer = SignalBuffer(
            (len(dc_values), 2, len(delays), 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # DC stays on host because it sets an external instrument.
            for dc_value, step in sched.scan("dc", dc_values.tolist()):
                if config.dc_device is not None:
                    set_flux(
                        context.devices[config.dc_device],
                        dc_value,
                        unit=config.dc_unit,
                        stop=context.scope.stop,
                    )
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("spectator_states", 2)  # 0 = I, 1 = Xπ (outer)
                delay = builder.add_sweep("delay", delay_sweep)
                spectator_cfg = (
                    config.x180_f2 if config.target == "f1" else config.x180_f1
                )
                pi2_cfg = config.x90_f1 if config.target == "f1" else config.x90_f2
                spectator = Pulse("spectator_x180", spectator_cfg)
                pi2 = Pulse("target_pi2", pi2_cfg)
                projection = Pulse(
                    "projection",
                    pi2_cfg,
                    phase=pi2_cfg.phase + 360 * config.detune * delay,
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                with (
                    builder.branch(by="spectator_states", align_timeline=True) as cases,
                    cases.case(1),
                ):
                    builder.emit(spectator)
                builder.emit(pi2)
                builder.timeline_inc(delay)
                builder.emit(projection)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
                    Classify("terminal", config.roles.f1_tone),
                    Classify("terminal", config.roles.f2_tone),
                ]
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        return IdleZZResult(dc_values, delays, buffer.array)

    def analyze(
        self,
        source: RunRecord[IdleZZCfg, IdleZZResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Fit full saved delay axes and plot signed frequency differences.

        Every required delay must contain all requested rounds. Other remains
        in the sigma denominator. Count errors weight the fringe fit.
        No calibration or DC writeback occurs.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("Idle ZZ analysis requires saved cfg")
        dc_values = (
            np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
        )
        validate_axes(
            result.dc_values,
            np.arange(2, dtype=np.int64),
            result.delays,
            result.sums,
            dc_values,
            (cfg.delay.expts, 3, 3),
        )
        complete = complete_points(result.sums, cfg.reps * cfg.rounds)
        traces = sigma(result.sums, cfg.target)
        shots = result.sums.sum(axis=(-2, -1))
        fits = []
        for dc_index in range(len(dc_values)):
            for state in np.arange(2, dtype=np.int64):
                fit = IdleBranchFit(
                    dc_index,
                    int(state),
                    int(np.nansum(shots[dc_index, state])),
                    bool(np.all(complete[dc_index, state])),
                    traces[dc_index, state],
                )
                fits.extend(
                    fit_traces(
                        (fit,),
                        (sigma_errors(result.sums[dc_index, state], cfg.target),),
                        delays=result.delays,
                        detune=cfg.detune,
                    )
                )
        answer = summarize(tuple(fits))
        dc_label = (
            "DC not swept (inactive coordinate)"
            if cfg.dc_device is None
            else f"{cfg.dc_device} ({cfg.dc_unit})"
        )
        for dc_index, dc_value in enumerate(dc_values):
            fig, ax = plots.subplots(num=f"ramsey_{dc_index}", figsize=(9, 6))
            reasons = []
            for fit in answer.branches[2 * dc_index : 2 * dc_index + 2]:
                label = "I" if fit.branch == 0 else "Xπ"
                ax.plot(
                    result.delays,
                    fit.trace,
                    ".",
                    label=f"Spectator {label} ({fit.acquired_shots} shots)",
                )
                if fit.fit_curve is not None:
                    ax.plot(
                        result.delays,
                        fit.fit_curve,
                        label=f"{label} fit: |f|={fit.magnitude_mhz:.5g} MHz",
                    )
                if fit.reason:
                    reasons.append(f"{label}: {fit.reason}")
                elif fit.signed_mhz is not None:
                    reasons.append(f"{label}: signed f={fit.signed_mhz:.5g} MHz")
            zeta = answer.zeta_mhz[dc_index]
            subtitle = f"ζ={zeta:.5g} MHz" if zeta is not None else "ζ unavailable"
            ax.set(
                xlabel="Delay (us)",
                ylabel="Target ⟨σ⟩ = (N_e − N_g) / N_total",
                title=f"Target {cfg.target}, detune={cfg.detune:g} MHz; {dc_label}={dc_value:g}\n{subtitle}; single-frequency model estimate\nNominal branches; mixture not excluded",
            )
            ax.text(
                0.01,
                -0.25,
                "\n".join(reasons),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        fig, ax = plots.subplots(num="zeta", figsize=(8, 5))
        for index, (value, error) in enumerate(
            zip(answer.zeta_mhz, answer.zeta_error_mhz, strict=True)
        ):
            if value is not None:
                ax.errorbar(
                    result.dc_values[index], value, yerr=error, marker="o", color="C0"
                )
        if not any(value is not None for value in answer.zeta_mhz):
            ax.text(
                0.5,
                0.5,
                "ζ unavailable: see branch reasons",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set(
            xlabel=dc_label,
            ylabel="ζ = f_Xπ − f_I (MHz)",
            title="Single-frequency model fit difference\nThermal mixture not excluded; not pure ZZ",
        )
        ax.grid(True)
        fig.tight_layout()
