from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)

from zcu_lab.v2._support.device import set_rf_power


@dataclass(frozen=True)
class PowerResult:
    """Pump powers in dBm and moment-based GE SNR in acquisition order."""

    powers: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class PowerAnalysis:
    best_power: float


class PowerModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[PulseCfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class PowerSweepCfg(ConfigBase):
    jpa_power: Annotated[SweepCfg, FieldMeta(label="JPA pump power", unit="dBm")]


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    jpa_rf_dev: Annotated[
        str,
        FieldMeta(
            label="JPA RF device",
            description="Connected RF device driven by the pump-power sweep.",
        ),
    ] = ""
    sweep: PowerSweepCfg
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    # powers stored in dBm on disk -> scale=IDENTITY (1.0); signals are real -> ZSpec dtype float64
    AXES_SPEC = AxesSpec(
        axes=(Axis("powers", "JPA Power", "dBm"),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
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
        jpa_powers = sweep2array(cfg.sweep.jpa_power, allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        def configure_axes(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        # Show independent points rather than connecting randomized settings.
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Power (dBm)",
            "Signal Difference",
            configure_axes=configure_axes,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_powers),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_powers, np.abs(data)),
        )

        # 2. Set the external pump power for each host-side point.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_power, step in sched.scan("power (dBm)", jpa_powers.tolist()):
                set_rf_power(context.devices[step.cfg.jpa_rf_dev], jpa_power)

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                modules = step.cfg.modules
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker, skew_penalty=step.cfg.skew_penalty: (
                            snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=skew_penalty,
                            )
                        ),
                        trackers=[tracker],
                    )
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

        fig, ax = plots.subplots("fit", figsize=config.figsize)
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
