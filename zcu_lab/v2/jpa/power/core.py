from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_rf_power


@dataclass(frozen=True)
class PowerResult:
    """Pump powers in dBm and moment-based GE SNR in acquisition order."""

    powers: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class PowerAnalysis:
    best_power: float


class PowerCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated g→e pulse for the excited preparation only.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the calibrated resonator pulse and IQ integration for both preparations.",
        ),
    ]
    jpa_rf_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA pump RF source",
            description="Select the RF source whose JPA pump power is swept.",
        ),
    ] = ""
    jpa_power: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA pump source power",
            unit="dBm",
            description="Enter Start and Stop in pump source dBm and Expts as the number of power samples (count).",
        ),
    ]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="IQ shape penalty",
            unit="a.u.",
            decimals=3,
            description="Enter a nonnegative weight for unequal IQ widths and skew; 0 uses separation SNR alone.",
        ),
    ] = Field(default=0.0, ge=0.0)
    seed: Annotated[
        int | None,
        FieldMeta(
            label="Scan-order seed",
            description=(
                "Enter a nonnegative integer to repeat the shuffled interior order; "
                "leave blank for a new order."
            ),
        ),
    ] = None
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per preparation at each setting in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per setting.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after readout for relaxation before the next shot (µs).",
        ),
    ]


class PowerExp(Experiment[PowerResult, PowerCfg]):
    # powers stored in dBm on disk -> scale=IDENTITY (1.0); signals are real -> ZSpec dtype float64
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("powers", label="JPA Power", unit="dBm"),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.float64),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="jpa/power",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        """Measure qubit readout SNR versus JPA pump power.

        At each power, reset, select no pulse or a pi pulse, then read out
        both preparation branches. IQ moments give one real SNR per power,
        with the configured skew penalty, not an assignment fidelity.
        Return powers in dBm and SNR in the same acquisition order.

        Interior powers are randomized; endpoints stay in place.
        ``jpa_rf_dev`` must name the pump device in ``context.devices``.
        """
        # 1. Prepare pump powers without imposing a monotonic acquisition order.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_powers = sweep2array(cfg.jpa_power, allow_array=True)
        np.random.default_rng(cfg.seed).shuffle(jpa_powers[1:-1])

        # Show independent points rather than connecting randomized settings.
        fig, ax = context.plots.subplots(num="measurement")
        title = "JPA sweep · pump power · g/e separation score"
        ax.set(
            title=title,
            xlabel="JPA pump source power (dBm)",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        line = live_line(ax)
        ax.lines[0].set(linestyle="None", marker="o", markersize=4, color="C0")
        retained_shots = np.zeros(len(jpa_powers), dtype=np.int64)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> Iterable[Artist]:
            """Draw acquired SNR on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            progress = ax.set_title(
                f"{title}\n{np.count_nonzero(np.isfinite(data))}/{len(jpa_powers)} settings sampled"
                f" · {retained_shots.sum():,} retained shots"
            )
            return (*line.set(jpa_powers, np.abs(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_powers),),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Set the external pump power for each host-side point.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for index, (jpa_power, step) in enumerate(
                sched.scan("Pump source power", jpa_powers.tolist())
            ):
                set_rf_power(context.devices[cfg.jpa_rf_dev], jpa_power)

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.delay_by_timeline()
                # Empty g case has no pi-length padding; preserve each case's readout time.
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.delay_by_timeline()
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()

                def score(
                    sums: NDArray[np.float64], *, point: int = index
                ) -> NDArray[np.float64]:
                    """Track cumulative retained shots from both preparations and return SNR."""
                    retained_shots[point] = int(sums[..., 0].sum())
                    return snr_as_signal(sums, skew_penalty=cfg.skew_penalty)

                step.run_acquire_summed(
                    prog,
                    soc,
                    reducer=shot_sums(prog, [IQ("readout", order=3)]),
                    options=RunOptions(converter=score),
                    rounds=cfg.rounds,
                )

        # 4. Keep SNR paired with the randomized power coordinates.
        signals = signals_buffer.array

        return PowerResult(powers=jpa_powers, signals=signals)

    def analyze(
        self, source: RunRecord[PowerCfg, PowerResult], options: None, *, plots: Plots
    ) -> PowerAnalysis:
        """Plot measured SNR and return the sampled power with peak SNR in dBm.

        Use only ``source`` data, with no fit or interpolation; ``options``
        is None. All-NaN SNR fails rather than selecting a power.
        """
        # 1. Select the sampled pump power with peak SNR, ignoring missing points.
        del options
        result = source.result

        jpa_powers = result.powers
        signals = result.signals
        snrs = np.abs(signals)

        max_idx = np.nanargmax(snrs)
        best_jpa_power = jpa_powers[max_idx]

        # 2. Mark the best power without connecting randomized pump settings.
        _, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.set(
            title="JPA sweep · pump power · g/e separation score",
            xlabel="JPA pump source power (dBm)",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.plot(jpa_powers, snrs, "o", color="C0", markersize=4, label="Measured")
        ax.axvline(
            best_jpa_power,
            color="C3",
            linestyle="--",
            label=f"Peak pump source power = {best_jpa_power:.2f} dBm",
        )
        ax.margins(y=0.25)
        ax.legend(loc="upper center")
        return PowerAnalysis(best_power=float(best_jpa_power))
