from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, ProgramExpCfg, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Branch, Pulse, Readout, Reset, SweepCfg

from zcu_lab.v2._support.device import set_rf_frequency


@dataclass(frozen=True)
class FreqResult:
    """Pump frequencies in MHz and moment-based GE SNR in acquisition order."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class FreqAnalysis:
    best_freq: float


class FreqModuleCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]


class FreqSweepCfg(ConfigBase):
    jpa_freq: Annotated[SweepCfg, FieldMeta(label="JPA pump freq", unit="MHz")]


class FreqCfg(ProgramExpCfg):
    jpa_rf_dev: Annotated[
        str | None,
        FieldMeta(
            label="JPA RF device",
            description="Connected RF device name; None skips RF control.",
        ),
    ] = ""
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)


class FreqExp(Experiment[FreqResult, FreqCfg]):
    # jpa_freq stored as Hz on disk -> scale=1e6; signals are SNR (float64)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", "JPA Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="jpa/freq",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure qubit readout SNR versus JPA pump frequency.

        At each pump setting, reset, select no pulse or a pi pulse, then
        read out both preparation branches. Their IQ moments give the SNR,
        with the configured skew penalty; this is not assignment fidelity.
        Return pump frequencies in MHz and one real SNR per point.

        Interior frequencies are randomized; endpoints stay in place.
        ``jpa_rf_dev`` names a device in ``context.devices``; None skips
        pump tuning without changing the requested frequency coordinates.
        """
        # 1. Prepare pump settings and preserve randomized acquisition order.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_freqs = sweep2array(cfg.sweep.jpa_freq, allow_array=True)
        np.random.shuffle(jpa_freqs[1:-1])  # randomize permutation

        def configure_axes(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        # Show independent points because neighboring samples need not be nearby.
        viewer = context.plots.liveplot_1d(
            "measurement",
            "JPA Frequency (MHz)",
            "Signal Difference",
            configure_axes=configure_axes,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_freqs),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_freqs, np.abs(data)),
        )

        # 2. Tune the external pump once per host-side scan point.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for jpa_freq, step in sched.scan("JPA Frequency", jpa_freqs.tolist()):
                if step.cfg.jpa_rf_dev is not None:
                    set_rf_frequency(
                        context.devices[step.cfg.jpa_rf_dev], jpa_freq * 1e6
                    )

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

        # 4. Keep SNR paired with the actual acquisition-order coordinates.
        signals = signals_buffer.array

        return FreqResult(freqs=jpa_freqs, signals=signals)

    def analyze(
        self, source: RunRecord[FreqCfg, FreqResult], options: None, *, plots: Plots
    ) -> FreqAnalysis:
        """Plot measured SNR and return the sampled pump frequency with peak SNR.

        ``source`` supplies this run's frequencies and signals; no fit or
        interpolation is applied. ``options`` is None. All-NaN SNR fails.
        """
        del options
        result = source.result

        jpa_freqs = result.freqs
        signals = result.signals

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        fig, ax = plots.subplots("fit", figsize=config.figsize)
        ax.scatter(jpa_freqs, real_signals, label="signal difference", s=2)
        ax.axvline(
            best_jpa_freq,
            color="r",
            ls="--",
            label=f"best JPA frequency = {best_jpa_freq:.2g} MHz",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return FreqAnalysis(best_freq=float(best_jpa_freq))
