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

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    set_freq_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    sweep: dict[str, SweepCfg]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FreqResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")
        _cfg = check_type(deepcopy(cfg), FreqCfg)

        jpa_freqs = sweep2array(_cfg["sweep"]["jpa_freq"], allow_array=True)
        np.random.shuffle(jpa_freqs[1:-1])  # randomize permutation

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any],
            update_hook: Optional[Callable[[int, list[MomentTracker]], None]],
        ) -> list[MomentTracker]:
            cfg: FreqCfg = cast(FreqCfg, ctx.cfg)
            setup_devices(cfg, progress=False)
            modules = cfg["modules"]

            assert update_hook is not None

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Branch("ge", [], Pulse("pi_pulse", modules["pi_pulse"])),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("ge", 2)],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                callback=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
            )
            return [tracker]

        with LivePlotScatter("JPA Frequency (MHz)", "Signal Difference") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "JPA Frequency",
                    jpa_freqs.tolist(),
                    before_each=lambda i, ctx, freq: set_freq_in_dev_cfg(
                        ctx.cfg["dev"], freq * 1e6, label="jpa_rf_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(jpa_freqs, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (jpa_freqs, signals)

        return jpa_freqs, signals

    def analyze(self, result: Optional[FreqResult] = None) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
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

        return float(best_jpa_freq), fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Frequency", "unit": "Hz", "values": jpa_freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, jpa_freqs, _ = load_data(filepath, **kwargs)
        assert jpa_freqs is not None
        assert len(jpa_freqs.shape) == 1 and len(signals.shape) == 1
        assert jpa_freqs.shape == signals.shape

        jpa_freqs = jpa_freqs * 1e-6  # Hz -> MHz

        jpa_freqs = jpa_freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_freqs, signals)

        return jpa_freqs, signals
