from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class PowerDepResult:
    """Hardware probe gains and mean complex readout IQ at each gain."""

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


@dataclass(frozen=True)
class PowerDepAnalyzeOptions:
    g0: complex | None = None
    e0: complex | None = None
    ac_coeff: float | None = None


class PowerDepCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Reset.Cfg | None = None
    init_pulse: Pulse.Cfg | None = None
    probe_pulse: Pulse.Cfg
    readout: Readout.Cfg
    gain: SweepCfg


class PowerDepExp(Experiment[PowerDepResult, PowerDepCfg]):
    Options: ClassVar[type[PowerDepAnalyzeOptions]] = PowerDepAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[PowerDepResult, PowerDepCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("gains", "Drive Power", "a.u."),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        cfg_schema_version="3.0",
        tag="mist/gain",
    )

    def run(
        self,
        config: PowerDepCfg,
        *,
        context: RunContext,
    ) -> PowerDepResult:
        """Measure a MIST response trace versus probe gain.

        Reset, apply the optional init pulse, apply the gain-swept probe,
        then read out. The board sweeps gain while the other pulse settings
        stay fixed. Return hardware gain coordinates and mean complex IQ
        per gain. IQ displacement reveals the probe-induced response; it
        is not a calibrated state population. ``config`` supplies the
        probe/readout modules and gain range.
        """
        # 1. Prepare gain coordinates for the probe's generator channel.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        viewer = live_line(context.plots, "measurement", "Pulse gain", "MIST")
        signals_buffer = SignalBuffer(
            (len(gains),),
            on_update=lambda data: viewer.update(gains, mist_signal2real(data)),
        )

        # 2. Bind the board gain sweep without changing the other probe settings.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            sched.cfg.probe_pulse.set_param("gain", sweep2param("gain", sched.cfg.gain))

            # 3. Reset and initialize before each probe/readout sequence.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", sched.cfg.reset),
                    Pulse("init_pulse", sched.cfg.init_pulse),
                    Pulse("probe_pulse", sched.cfg.probe_pulse),
                    Readout("readout", sched.cfg.readout),
                )
                .declare_sweep("gain", sched.cfg.gain)
                .build_and_acquire(
                    rounds=cfg.rounds, options=AcquireOptions(retry=cfg.acquire_retry)
                )
            )

        # 4. Preserve complex IQ so analysis can choose its own reference.
        return PowerDepResult(gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[PowerDepCfg, PowerDepResult],
        options: PowerDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the IQ displacement from ``options.g0`` using ``source`` data.

        If g0 is None, use the first measured IQ, not the live view's mean
        over the first 5% of gains. ``ac_coeff`` maps gain squared to photon
        number when supplied; ``e0`` only sets the vertical display range.
        Return no fit or calibrated population estimate.
        """
        result = source.result
        g0, e0, ac_coeff = options.g0, options.e0, options.ac_coeff

        gains, signals = result.gains, result.signals

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        _, ax = plots.subplots("fit", figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))
