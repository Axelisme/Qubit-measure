from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

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

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    jpa_rf_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA RF device",
            description="Connected RF device driven by the pump-power sweep.",
        ),
    ] = ""
    jpa_power: Annotated[SweepCfg, FieldMeta(label="JPA pump power", unit="dBm")]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)
    seed: Annotated[
        int | None,
        FieldMeta(
            label="Scan-order seed",
            description=(
                "Fixes the shuffled order of interior powers; empty draws a new "
                "order each run."
            ),
        ),
    ] = None
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


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

        def configure_axes(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        # Show independent points rather than connecting randomized settings.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Power (dBm)", ylabel="Signal Difference")
        line = live_line(ax)
        configure_axes(ax)

        @context.plots.on_plot_thread
        def update(data):
            return line.set(jpa_powers, np.abs(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_powers),),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Set the external pump power for each host-side point.
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            for jpa_power, step in sched.scan("power (dBm)", jpa_powers.tolist()):
                set_rf_power(context.devices[cfg.jpa_rf_dev], jpa_power)

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                tracker = MomentTracker()
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                builder.emit(Reset("reset", cfg.reset))
                builder.sync()
                # Empty g case has no pi-length padding; preserve each case's readout time.
                with builder.branch(by="ge", align_timeline=False) as cases:
                    with cases.case(1):
                        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.sync()
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(
                    prog, soc,
                    options=RunOptions(
                        converter=lambda _raw, tracker=tracker: snr_as_signal(
                            [tracker], ge_axis=1, skew_penalty=cfg.skew_penalty,
                        ),
                    ),
                    trackers=[tracker], rounds=cfg.rounds,
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
        del options
        result = source.result

        jpa_powers = result.powers
        signals = result.signals
        snrs = np.abs(signals)

        max_idx = np.nanargmax(snrs)
        best_jpa_power = jpa_powers[max_idx]

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.scatter(jpa_powers, snrs, label="signal difference", s=1)
        ax.axvline(
            best_jpa_power,
            color="r",
            ls="--",
            label=f"best JPA power = {best_jpa_power:.2g} dBm",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return PowerAnalysis(best_power=float(best_jpa_power))
