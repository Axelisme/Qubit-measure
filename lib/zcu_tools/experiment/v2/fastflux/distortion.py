from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    NonBlocking,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (amps, freqs, signals2D)
DistortionResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def distortion_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class DistortionModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class DistortionCfg(ModularProgramCfg, TaskCfg):
    modules: DistortionModuleCfg
    readout_t: float
    sweep: dict[str, SweepCfg]


class DistortionExp(AbsExperiment):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> DistortionResult:
        _cfg = check_type(deepcopy(cfg), DistortionCfg)

        # force length be the outer loop
        _cfg["sweep"] = {
            "length": _cfg["sweep"]["length"],
            "phase": _cfg["sweep"]["phase"],
        }

        lengths = sweep2array(_cfg["sweep"]["length"])
        phases = sweep2array(_cfg["sweep"]["phase"])

        length_params = sweep2param("length", _cfg["sweep"]["length"])
        phase_params = sweep2param("phase", _cfg["sweep"]["phase"])

        with LivePlotter2D("Time (us)", "Phase (deg)") as viewer:

            def measure_fn(ctx: TaskState, update_hook: Optional[Callable]):
                nonlocal lengths, phases
                modules = ctx.cfg["modules"]
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        NonBlocking(
                            [
                                Pulse(
                                    "flux_pulse",
                                    modules["flux_pulse"],
                                    block_mode=False,
                                ),
                                SoftDelay("wait_time", delay=length_params),
                                Pulse(
                                    "pi2_pulse1", modules["pi2_pulse"], tag="pi2_pulse1"
                                ),
                            ]
                        ),
                        Delay("readout_t", ctx.cfg["readout_t"]),
                        Pulse(
                            name="pi2_pulse2",
                            cfg={  # type: ignore[dict-item]
                                **modules["pi2_pulse"],
                                "phase": phase_params,
                            },
                            tag="pi2_pulse2",
                        ),
                        Readout("readout", modules["readout"]),
                    ],
                )

                # get actual values after program generation, in case there are some adjustments
                true_ts = cast(
                    NDArray[np.float64],
                    prog.get_time_param("pi2_pulse1", "t", as_array=True),
                )
                lengths = true_ts - true_ts[0] + lengths[0]
                phases = cast(
                    NDArray[np.float64],
                    prog.get_pulse_param("pi2_pulse2", "phase", as_array=True),
                )

                return prog.acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(lengths), len(phases))
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, phases, distortion_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (lengths, phases, signals)

        return lengths, phases, signals

    def analyze(
        self, cfg: dict[str, Any], result: Optional[DistortionResult] = None
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, phases, signals2D = result

        real_signals = distortion_signal2real(signals2D)

        _cfg = check_type(cfg, DistortionCfg)
        modules = _cfg["modules"]

        # align to center of pi/2 pulse
        pi2_len = modules["pi2_pulse"]["waveform"]["length"]
        lengths = lengths + pi2_len / 2 - 0.025

        flux_pulse = modules["flux_pulse"]
        start_t = flux_pulse["pre_delay"]
        end_t = start_t + flux_pulse["waveform"]["length"]

        I_values = np.mean(real_signals * np.cos(np.deg2rad(phases))[None], axis=1)
        Q_values = np.mean(real_signals * np.sin(np.deg2rad(phases))[None], axis=1)
        init_phases = np.unwrap(np.arctan2(I_values, Q_values))
        detunes = np.gradient(init_phases, lengths) / (2 * np.pi)

        saturated_idxs = np.argsort(np.abs(detunes))[int(len(detunes) * 0.9) :]
        flux_detune = np.mean(detunes[saturated_idxs])
        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.zeros_like(ideal_lengths)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = flux_detune

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        ax1.imshow(
            real_signals.T,
            extent=(lengths[0], lengths[-1], phases[0], phases[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax1.set_ylabel("Phase (deg)")

        ax2.plot(ideal_lengths, ideal_curve, "g-", label="Ideal Detune")
        ax2.plot(lengths, detunes, ".-")
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Wait Time (us)")

        ax2.axvspan(
            start_t - pi2_len / 2, start_t + pi2_len / 2, color="gray", alpha=0.3
        )
        ax1.axvline(start_t, color="black", linestyle="--")
        ax2.axvspan(end_t - pi2_len / 2, end_t + pi2_len / 2, color="gray", alpha=0.3)
        ax1.axvline(end_t, color="black", linestyle="--")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[DistortionResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/distortion",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, phases, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Wait Time", "unit": "s", "values": lengths * 1e-6},
            y_info={"name": "Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DistortionResult:
        signals2D, lengths, phases = load_data(filepath, **kwargs)
        assert phases is not None and lengths is not None
        assert len(phases.shape) == 1 and len(lengths.shape) == 1
        assert signals2D.shape == (len(phases), len(lengths))

        lengths = lengths * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        phases = phases.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.T.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lengths, phases, signals2D)

        return lengths, phases, signals2D
