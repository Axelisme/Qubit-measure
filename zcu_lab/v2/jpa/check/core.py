from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    MHZ_TO_HZ,
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
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)

from zcu_lab.v2._support.device import set_output


@dataclass(frozen=True)
class CheckResult:
    """Pump labels (0 off, 1 on), MHz frequencies, and complex IQ [output, freq]."""

    outputs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class CheckModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class CheckSweepCfg(ConfigBase):
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class CheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: CheckModuleCfg
    jpa_rf_dev: Annotated[
        str,
        FieldMeta(
            label="JPA RF device",
            description="Connected RF device whose output is toggled off then on.",
        ),
    ] = ""
    sweep: CheckSweepCfg


class CheckExp(PersistableExperiment[CheckResult, CheckCfg]):
    OUTPUT_MAP: ClassVar[dict[int, Literal["off", "on"]]] = {0: "off", 1: "on"}

    # freqs stored as Hz on disk -> scale=MHZ_TO_HZ; outputs are int JPA labels.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("outputs", "JPA Output", "a.u.", dtype=np.int_),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
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
        modules = cfg.modules

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.gen_ch},
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Magnitude", num_lines=2
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
                modules = step.cfg.modules
                modules.readout.set_param(
                    "freq", sweep2param("freq", step.cfg.sweep.freq)
                )
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        PulseReadout("readout", modules.readout),
                    )
                    .declare_sweep("freq", step.cfg.sweep.freq)
                    .build_and_acquire()
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
