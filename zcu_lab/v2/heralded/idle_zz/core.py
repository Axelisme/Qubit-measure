"""Herald-resolved conditional Ramsey acquisition and idle ZZ estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_sweep_dict, sweep2array
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.device import set_flux
from zcu_lab.v2.support.paired_counts import qubit_sigma, validate_qubit_tones

from ..counts import terminal_counts


@dataclass(frozen=True)
class IdleZZHeraldedResult:
    """Unnormalized float64 sums (D,2,T,3,3,3,3).

    The four class axes are herald Qubit1/Qubit2 then terminal Qubit1/Qubit2,
    each g/e/other. No class is dropped.

    dc_values is the D-point device coordinate, or [0] when DC is disabled.
    The nominal spectator preparation axis is fixed [0,1] for I/Xπ.
    delays stores actual quantized µs.
    Rounds accumulate without averaging. Unmeasured points stay NaN; completed
    rounds survive cancellation. No individual shots or IQ are retained.
    """

    dc_values: NDArray[np.float64]
    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


class IdleZZHeraldedCfg(ConfigBase):
    """Target qubit1/qubit2 Ramsey with nominal spectator I/Xπ branches after a visible herald.

    target selects the qubit doing π/2→delay→phase projection. The other qubit
    receives I (branch 0) or Xπ (branch 1) after the visible herald readout.
    reset=None omits preparation reset but not the visible herald.
    Both reads share the mux classifiers and readout.post_delay sets the wait
    after herald.
    The tone indices address the full readout. Delay is µs,
    pulse frequency MHz, phase degrees. detune (MHz) is a required operator input:
    second π/2 phase = first.phase + 360*detune*actual_delay. Its magnitude must
    comfortably exceed expected |ζ| plus qubit frequency error so both nominal
    branches remain on its sign-selected side. No automatic detune adjustment
    occurs. dc_device=None leaves all sources untouched;
    otherwise only the named source is swept, with no compensation. Rounds
    accumulate class counts on the acquisition side. Qubit1/Qubit2 require circle
    classifiers with radii from GE Post; other tones remain in the full readout.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an optional preparation reset before the visible herald.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the Rabi-calibrated Qubit1 flip used when Qubit1 is the spectator.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the Rabi-calibrated Qubit2 flip used when Qubit2 is the spectator.",
        ),
    ]
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Enter the Rabi-calibrated Qubit1 π/2 pulse for the Ramsey pair when Qubit1 is the target.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Enter the Rabi-calibrated Qubit2 π/2 pulse for the Ramsey pair when Qubit2 is the target.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Use the same full mux module for herald and terminal reads, with GE Post circle radii for both qubits.",
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    target: Annotated[
        Literal["qubit1", "qubit2"],
        FieldMeta(
            label="Ramsey target",
            description="Select the qubit that receives both π/2 pulses; the other qubit is the spectator.",
        ),
    ] = "qubit1"
    dc_device: Annotated[
        DeviceName | None,
        FieldMeta(
            label="Coupler DC source",
            description="Select the connected coupler-bias source by name; leave blank to keep all sources unchanged.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="DC source unit",
            description="Select amperes, volts or the selected source's native unit for the bias sweep.",
        ),
    ] = "native"
    dc: Annotated[
        SweepCfg,
        FieldMeta(
            label="Coupler DC bias sweep",
            description="Enter the sweep in DC source unit (A, V or native); leave the source blank to ignore this sweep.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Ramsey free-evolution delay",
            unit="µs",
            description="Enter the time sweep between the target's two π/2 pulses.",
        ),
    ]
    detune: Annotated[
        float,
        Field(allow_inf_nan=False),
        FieldMeta(
            label="Programmed Ramsey detuning",
            unit="MHz",
            description="Enter the signed phase ramp (MHz) of the second π/2 pulse; its magnitude must exceed the expected conditional shift plus frequency error.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter attempted shots per branch and delay per round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter repeated delay scans at each DC bias point.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial wait",
            unit="µs",
            description="Enter the wait before the first shot of each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot before the next preparation.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require complete tones and consistent independent calibrated drive lines."""
        m = self
        if m.x180_qubit1.ch == m.x180_qubit2.ch:
            raise ValueError("Qubit1/Qubit2 require independent charge generators")
        if m.x90_qubit1.ch != m.x180_qubit1.ch or m.x90_qubit2.ch != m.x180_qubit2.ch:
            raise ValueError("Each qubit's Xπ and Xπ/2 must use the same generator")
        validate_qubit_tones(m.readout, m.qubit1_tone, m.qubit2_tone)
        delays = sweep2array(self.delay)
        if np.any(delays < 0):
            raise ValueError("Ramsey delay must be nonnegative")
        if self.dc_device is not None and not self.dc_device.strip():
            raise ValueError("DC source must name a connected device or be None")
        return self


def restore_result(data: GroupedLoadData[IdleZZHeraldedCfg]) -> IdleZZHeraldedResult:
    """Restore saved DC coordinates, delays in µs and 81-class float64 counts.

    Read the grouped sums variable without mutating it. Missing variables and
    invalid stored data propagate load errors.
    """
    sums = data.variable("sums")
    return IdleZZHeraldedResult(
        dc_values=sums.axes[6].astype(np.float64),
        delays=sums.axes[4].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


_BRANCH_NAMES = ("I", "Xπ")


def _shot_sem(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Project terminal probability covariance onto the target's e-minus-g contrast."""
    marginal = counts.sum(axis=-1 if target == "qubit1" else -2)
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1], 0))


def _label_ramsey(fig: Figure, cfg: IdleZZHeraldedCfg, index: int) -> str:
    """Label the shared Run/Analyze data axes and return the figure title."""
    target = cfg.target.capitalize()
    ax = fig.axes[0]
    ax.set(
        xlabel="Ramsey free-evolution delay (µs)",
        ylabel=f"{target} contrast P(e) − P(g)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    title = f"Heralded idle ZZ · {target} · Ramsey contrast"
    if cfg.dc_device is not None:
        title += f" · Bias sweep point {index + 1}"
    return title


class IdleZZHeraldedExp(Experiment[IdleZZHeraldedResult, IdleZZHeraldedCfg]):
    """Acquire both reads for idle ZZ without filtering herald classes."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZHeraldedResult, IdleZZHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit1", unit=""
                        ),
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
                        field_name="sums",
                        label="Class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=IdleZZHeraldedResult,
            cfg_type=IdleZZHeraldedCfg,
            cfg_schema_version="1.0",
            tag="heralded/idle_zz",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: IdleZZHeraldedCfg, *, context: RunContext
    ) -> IdleZZHeraldedResult:
        """Acquire optional reset → visible herald → spectator I/Xπ → Ramsey → terminal.

        Hardware scans spectator_states and delay; DC alone stays on host
        because it changes an external instrument. Return all 81 joint class counts.

        """
        # 1. Pair hardware-rounded Ramsey delays with optional host DC settings.
        delay_sweep = round_sweep_dict(config.delay, "time", {"soccfg": context.soccfg})
        delays = sweep2array(delay_sweep)
        dc_values = (
            np.array([0.0]) if config.dc_device is None else sweep2array(config.dc)
        )
        # 2. Compare the same two spectator branches at each bias point.
        views = []
        for index in range(len(dc_values)):
            fig, ax = context.plots.subplots(
                num=f"ramsey_{index}", figsize=(12, 9), layout="constrained"
            )
            fig.suptitle(_label_ramsey(fig, config, index))
            viewer = live_errorbar(ax, num_lines=2)
            for line, name in zip(ax.get_lines(), _BRANCH_NAMES, strict=True):
                line.set(label=f"Herald gg · spectator {name}")
            fig.legend(loc="outside lower center", ncols=2)
            views.append((fig, viewer))
            fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Figure, ...]:
            gg = terminal_counts(sums, ("g", "g"))
            traces = qubit_sigma(gg, config.target)
            errors = _shot_sem(gg, config.target)
            for index, (_, viewer) in enumerate(views):
                viewer.set(delays, traces[index], errors[index])
            return tuple(fig for fig, _ in views)

        # 3. Set DC on the host; sweep spectator preparation and detuned target Ramsey delay in hardware.
        buffer = SignalBuffer(
            (len(dc_values), 2, len(delays), 3, 3, 3, 3),
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
                    config.x180_qubit2
                    if config.target == "qubit1"
                    else config.x180_qubit1
                )
                pi2_cfg = (
                    config.x90_qubit1
                    if config.target == "qubit1"
                    else config.x90_qubit2
                )
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
                builder.emit(readout, tag="herald")
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
                    Classify("herald", config.qubit1_tone),
                    Classify("herald", config.qubit2_tone),
                    Classify("terminal", config.qubit1_tone),
                    Classify("terminal", config.qubit2_tone),
                ]
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        # 4. Retain joint herald/terminal counts for completed rounds on every planned DC/delay slot.
        return IdleZZHeraldedResult(dc_values, delays, buffer.array)
