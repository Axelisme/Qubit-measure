from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    BathReset,
    BathResetCfg,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal_nd


def _default_phase_values() -> NDArray[np.float64]:
    return np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)


@dataclass(frozen=True)
class FreqGainResult:
    """Mean IQ (phase, gain, frequency); phases are offsets in degrees, freqs in MHz."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    phases: NDArray[np.float64] = field(default_factory=_default_phase_values)


@dataclass(frozen=True)
class FreqGainAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class FreqGainAnalysis:
    gain: float
    freq: float


class FreqGainModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[BathResetCfg, FieldMeta(label="Tested reset")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class FreqGainSweepCfg(ConfigBase):
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class FreqGainCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqGainModuleCfg
    sweep: FreqGainSweepCfg


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return (
        np.abs(signals[2] - signals[0]) ** 2 + np.abs(signals[3] - signals[1]) ** 2
    )  # (gain, freq)


class FreqGainExp(PersistableExperiment[FreqGainResult, FreqGainCfg]):
    Options: ClassVar[type[FreqGainAnalyzeOptions]] = FreqGainAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Cavity Frequency", "Hz", scale=MHZ_TO_HZ, dtype=np.float64),
            Axis(
                "gains", "Cavity drive Gain", "a.u.", scale=IDENTITY, dtype=np.float64
            ),
            Axis("phases", "Pi2 Phase", "deg", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=FreqGainResult,
        cfg_type=FreqGainCfg,
        cfg_schema_version="2.0",
        tag="twotone/reset/bath/freq_gain",
    )

    def run(
        self,
        config: FreqGainCfg,
        *,
        context: RunContext,
    ) -> FreqGainResult:
        """Measure bath-reset tomography contrast versus cavity gain and frequency.

        After optional reset and initialization, start the cavity and qubit
        tones together, apply the final pi/2 pulse after the cavity tone,
        then read out. Sweep
        its phase by 0, 90, 180 and 270 degrees relative to the configured
        phase. Return mean complex IQ with shape (4, Ngain, Nfreq), normalized
        gains and frequencies in MHz. Opposite-phase differences probe
        transverse contrast, not reset fidelity.
        """
        # Prepare cavity-drive coordinates and four relative tomography phases.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        phases = _default_phase_values()
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )

        # Display opposite-phase contrast while retaining all four complex IQ planes.
        viewer = context.plots.liveplot_2d(
            "measurement", "Cavity Frequency (MHz)", "Cavity drive Gain (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (4, len(gains), len(freqs)),
            on_update=lambda data: viewer.update(
                freqs, gains, bathreset_signal2real(data).T
            ),
        )

        # Sweep both cavity knobs and the final pi/2 phase in one program.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            tested_reset = modules.tested_reset
            tested_reset.set_param(
                "res_gain", sweep2param("gain", sched.cfg.sweep.gain)
            )
            tested_reset.set_param(
                "res_freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            phase_param = QickSweep1D("phase", 0.0, 270.0) + tested_reset.pi2_cfg.phase
            tested_reset.set_param("pi2_phase", phase_param)

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    BathReset("tested_reset", tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("phase", 4)
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .declare_sweep("freq", sched.cfg.sweep.freq)
                .build_and_acquire()
            )

        return FreqGainResult(
            gains=gains,
            freqs=freqs,
            phases=phases,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[FreqGainCfg, FreqGainResult],
        options: FreqGainAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqGainAnalysis:
        """Select cavity gain and frequency from the strongest smoothed contrast.

        options controls smoothing along gain and frequency, not phase.
        Return normalized gain and MHz frequency from the four-phase contrast
        map, and publish it as fit; source.cfg may be None.
        """
        result = source.result

        gains, freqs, signals = result.gains, result.freqs, result.signals

        # Find peak in amplitude
        smooth_signals = smooth_signal_nd(
            signals,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axes=(1, 2),
        ).astype(np.complex128)
        real_signals = bathreset_signal2real(smooth_signals)

        gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
        freq_opt = freqs[np.argmax(np.max(real_signals, axis=0))]

        fig, ax = plots.subplots("fit")

        ax.imshow(
            real_signals,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
        )
        peak_label = f"({gain_opt:.2f} a.u., {freq_opt:.1f} MHz)"
        ax.scatter(freq_opt, gain_opt, color="r", s=40, marker="*", label=peak_label)
        ax.set_xlabel("Cavity Frequency (MHz)", fontsize="x-large")
        ax.set_ylabel("Cavity drive Gain (a.u.)", fontsize="x-large")
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return FreqGainAnalysis(gain=float(gain_opt), freq=float(freq_opt))
