from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, sweep2param

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_output


@dataclass(frozen=True)
class CheckResult:
    """Pump labels (0 off, 1 on), MHz frequencies, and complex IQ [output, freq]."""

    outputs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class CheckCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    jpa_rf_dev: Annotated[
        str,
        FieldMeta(
            label="JPA RF device",
            description="Connected RF device whose output is toggled off then on.",
        ),
    ] = ""
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class CheckExp(Experiment[CheckResult, CheckCfg]):
    OUTPUT_MAP: ClassVar[dict[int, Literal["off", "on"]]] = {0: "off", 1: "on"}

    # freqs stored as Hz on disk -> scale=1e6; outputs are int JPA labels.
    AXES_SPEC: ClassVar[AxesSpec[CheckResult, CheckCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("outputs", "JPA Output", "a.u.", dtype=np.int_),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
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
        modules = cfg

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.gen_ch},
        )

        viewer = live_line(
            context.plots, "measurement", "Frequency (MHz)", "Magnitude", num_lines=2
        )
        signals_buffer = SignalBuffer(
            (len(outputs), len(freqs)),
            on_update=lambda data: viewer.update(freqs, check_signal2real(data)),
        )

        # 2. Acquire OFF first, then ON; the final pump state is not restored.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for output, step in sched.scan("JPA on/off", outputs.tolist()):
                set_output(
                    context.devices[step.cfg.jpa_rf_dev], self.OUTPUT_MAP[output]
                )

                # 3. Reset before each frequency-swept readout pulse.
                modules = step.cfg
                modules.readout.set_param("freq", sweep2param("freq", step.cfg.freq))
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", modules.reset),
                        PulseReadout("readout", modules.readout),
                    )
                    .declare_sweep("freq", step.cfg.freq)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        # 4. Retain complex spectra; magnitude is only a display transform.
        signals = signals_buffer.array

        return CheckResult(outputs=outputs, freqs=freqs, signals=signals)

    def analyze(
        self, source: RunRecord[CheckCfg, CheckResult], options: None, *, plots: Plots
    ) -> None:
        """Overlay pump-off/on IQ magnitudes from ``source`` in the fit figure.

        ``options`` is None. This diagnostic returns no calibration value.
        """
        del options
        result = source.result

        outputs = result.outputs
        freqs = result.freqs
        signals2D = result.signals
        real_signals = check_signal2real(signals2D)

        _, ax = plots.subplots("fit", figsize=config.figsize)
        for i, output in enumerate(outputs):
            ax.plot(
                freqs,
                real_signals[i, :],
                label=f"JPA {self.OUTPUT_MAP[output]}",
                marker="o",
                markersize=4,
                linestyle="-",
            )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Magnitude (a.u.)")
        ax.legend()
        ax.grid(True)
