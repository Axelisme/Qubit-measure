from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_output


@dataclass(frozen=True)
class CheckResult:
    """Pump labels (0 off, 1 on), MHz frequencies, and complex IQ [output, freq]."""

    outputs: NDArray[np.int_]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return np.abs(signals)


class CheckCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Frequency-swept readout",
            description="Select the resonator pulse and IQ integration; the sweep sets the RF frequency.",
        ),
    ]
    jpa_rf_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA pump RF source",
            description="Select the RF source to switch the JPA pump off, then on.",
        ),
    ] = ""
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter Start and Stop in absolute probe RF MHz and Expts as the number of frequency samples (count).",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency at each pump state in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per pump state.",
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


class CheckExp(Experiment[CheckResult, CheckCfg]):
    OUTPUT_MAP: ClassVar[dict[int, Literal["off", "on"]]] = {0: "off", 1: "on"}

    # freqs stored as Hz on disk -> scale=1e6; outputs are int JPA labels.
    AXES_SPEC: ClassVar[AxesSpec[CheckResult, CheckCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("outputs", label="JPA Output", unit="a.u.", dtype=np.int_),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=CheckResult,
        cfg_type=CheckCfg,
        tag="jpa/check",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: CheckCfg, *, context: RunContext) -> CheckResult:
        """Compare the readout reflection spectrum with the JPA pump off and on.

        Disable the pump, acquire a reset/readout frequency sweep, then
        enable it and repeat the same sweep. Return frequencies in MHz,
        output labels 0/1, and mean complex IQ shaped [output, frequency].
        The spectra diagnose the pump's effect, not qubit-state SNR.

        ``jpa_rf_dev`` must name the RF device in ``context.devices``.
        Normal completion leaves its output ON; no off-state restoration
        is performed.
        """
        # 1. Prepare a shared hardware frequency grid for both pump states.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
        )

        fig, ax = context.plots.subplots(num="measurement")
        title = "JPA pump check · readout resonator · reflection"
        ax.set(
            title=title,
            xlabel="Readout RF frequency (MHz)",
            ylabel="Magnitude of mean IQ (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        line = live_line(ax, num_lines=2)
        self._style_pump_states(outputs, ax.lines)
        ax.legend(loc="upper center", ncols=2)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Draw acquired complex IQ on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            progress = ax.set_title(
                f"{title}\n{np.count_nonzero(np.isfinite(data))}/{data.size} spectrum points sampled"
            )
            return (*line.set(freqs, check_signal2real(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(outputs), len(freqs)),
            on_update=update,
        )

        # 2. Acquire OFF first, then ON; the final pump state is not restored.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for output, step in sched.scan("JPA on/off", outputs.tolist()):
                set_output(context.devices[cfg.jpa_rf_dev], self.OUTPUT_MAP[output])

                # 3. Reset before each frequency-swept readout pulse.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                freq = builder.add_sweep("freq", cfg.freq)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
                step.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        # 4. Retain complex spectra; magnitude is only a display transform.
        signals = signals_buffer.array

        return CheckResult(outputs=outputs, freqs=freqs, signals=signals)

    def analyze(
        self, source: RunRecord[CheckCfg, CheckResult], options: None, *, plots: Plots
    ) -> None:
        """Overlay pump-off/on IQ magnitudes from ``source`` in the fit figure.

        ``options`` is None. This diagnostic returns no calibration value.
        """
        # 1. Reduce pump-off/on IQ to magnitude for a reflection-spectrum comparison.
        del options
        result = source.result

        outputs = result.outputs
        freqs = result.freqs
        signals2D = result.signals
        real_signals = check_signal2real(signals2D)

        # 2. Overlay both pump states on the same readout-frequency axis.
        _, ax = plots.subplots(num="fit", figsize=config.figsize)
        for values in real_signals:
            ax.plot(freqs, values, linestyle="-")
        self._style_pump_states(outputs, ax.lines)

        ax.set(
            title="JPA pump check · readout resonator · reflection",
            xlabel="Readout RF frequency (MHz)",
            ylabel="Magnitude of mean IQ (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.margins(y=0.25)
        ax.legend(loc="upper center", ncols=2)

    def _style_pump_states(
        self, outputs: NDArray[np.int_], lines: Iterable[Line2D]
    ) -> None:
        """Give overlapping pump-on and pump-off data distinguishable markers."""
        for output, artist in zip(outputs, lines, strict=True):
            pump_off = output == 0
            artist.set(
                label=f"Pump {self.OUTPUT_MAP[output]}",
                color="C0" if pump_off else "C1",
                marker="o" if pump_off else "+",
                markersize=7 if pump_off else 5,
                markerfacecolor="none",
            )
