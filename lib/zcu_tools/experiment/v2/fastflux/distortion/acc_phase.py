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
    Join,
    ModularProgramCfg,
    ModularProgramV2,
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

# (lengths, phases, signals2D)
AccPhaseResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def acc_phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AccPhaseModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class AccPhaseCfg(ModularProgramCfg, TaskCfg):
    modules: AccPhaseModuleCfg
    readout_t: float
    sweep: dict[str, SweepCfg]


class AccPhaseExp(AbsExperiment[AccPhaseResult, AccPhaseCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> AccPhaseResult:
        _cfg = check_type(deepcopy(cfg), AccPhaseCfg)
        modules = _cfg["modules"]

        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {"soccfg": soccfg},
        )
        phases = sweep2array(
            _cfg["sweep"]["phase"],
            "phase",
            {"soccfg": soccfg, "gen_ch": modules["pi2_pulse"]["ch"]},
        )

        length_params = sweep2param("length", _cfg["sweep"]["length"])
        phase_params = sweep2param("phase", _cfg["sweep"]["phase"])

        with LivePlotter2D("Time (us)", "Phase (deg)") as viewer:

            def measure_fn(
                ctx: TaskState[NDArray[np.complex128], Any],
                update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
            ) -> list[NDArray[np.float64]]:
                cfg: AccPhaseCfg = cast(AccPhaseCfg, ctx.cfg)
                modules = cfg["modules"]

                return ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Join(
                            Pulse("flux_pulse", modules["flux_pulse"]),
                            [
                                SoftDelay("wait_time", delay=length_params),
                                Pulse(
                                    "pi2_pulse1", modules["pi2_pulse"], tag="pi2_pulse1"
                                ),
                            ],
                            SoftDelay("readout_t", ctx.cfg["readout_t"]),
                        ),
                        Pulse(
                            name="pi2_pulse2",
                            cfg={  # type: ignore[dict-item]
                                **modules["pi2_pulse"],
                                "phase": phase_params,
                            },
                        ),
                        Readout("readout", modules["readout"]),
                    ],
                    sweep=[
                        ("length", ctx.cfg["sweep"]["length"]),
                        ("phase", ctx.cfg["sweep"]["phase"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(lengths), len(phases))
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, phases, acc_phase_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (lengths, phases, signals)

        return lengths, phases, signals

    def analyze(
        self,
        cfg: Optional[AccPhaseCfg] = None,
        result: Optional[AccPhaseResult] = None,
    ) -> Figure:
        if cfg is None:
            cfg = self.last_cfg
        assert cfg is not None, "No config found"
        modules = cfg["modules"]

        flux_pulse = modules["flux_pulse"]
        pi2_len = float(modules["pi2_pulse"]["waveform"]["length"])

        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, phases, signals2D = result

        # align to center of pi/2 pulse
        lengths = lengths + pi2_len / 2

        real_signals = acc_phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        X_inv = np.linalg.pinv(X)
        coeffs = X_inv @ real_signals.T
        init_phases = np.unwrap(np.arctan2(coeffs[2], coeffs[1]))
        detunes = -np.gradient(init_phases, lengths) / (2 * np.pi)

        sort_idxs = np.argsort(np.abs(detunes))
        mean_background = np.mean(detunes[sort_idxs[: int(len(detunes) * 0.1)]])
        mean_topdetune = np.mean(detunes[sort_idxs[int(len(detunes) * 0.9) :]])

        start_t = float(flux_pulse["pre_delay"])
        end_t = start_t + float(flux_pulse["waveform"]["length"])

        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.full_like(ideal_lengths, mean_background)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = (
            mean_topdetune
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        ax1.imshow(
            real_signals.T,
            extent=(lengths[0], lengths[-1], phases[0], phases[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax1.set_ylabel("Phase (deg)")

        ax2.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")
        ax2.plot(lengths, detunes, ".-")
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Wait Time (us)")

        plot_kwargs = dict(color="gray", alpha=0.3)
        ax2.axvspan(start_t - pi2_len / 2, start_t + pi2_len / 2, **plot_kwargs)
        ax2.axvspan(end_t - pi2_len / 2, end_t + pi2_len / 2, **plot_kwargs)
        ax1.axvline(start_t, color="black", linestyle="--")
        ax1.axvline(end_t, color="black", linestyle="--")

        ax2.legend()

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[AccPhaseResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/distortion/acc_phase",
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

    def load(self, filepath: str, **kwargs) -> AccPhaseResult:
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
