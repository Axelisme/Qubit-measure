from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D, make_plot_frame


@dataclass(frozen=True)
class FakeResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FakeCfg | None = None


class FakeCfg(ExpCfgModel): ...


def fake_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FakeExp(AbsExperiment[FakeResult, FakeCfg]):
    def run(self, cfg: FakeCfg) -> FakeResult:
        orig_cfg = deepcopy(cfg)

        # Predicted frequency points (before mapping to ADC domain)
        freqs = np.linspace(4.5, 5.5, 201)  # MHz

        round_n = 100

        def measure_fn(
            ctx: TaskState, update_hook: Callable | None
        ) -> NDArray[np.complex128]:
            signal_buffer = []
            for i in range(round_n):
                # Simulate the measurement of the signal at the given frequency
                raw_signal = (
                    np.exp(-((freqs - 5.0) ** 2) / (2 * 0.1**2))
                    + 0.1 * np.random.randn()
                    + 1j * 0.1 * np.random.randn()
                )
                signal_buffer.append(raw_signal)

                # Update the context with the new signal
                if update_hook is not None:
                    update_hook(i, np.mean(signal_buffer, axis=0))

                time.sleep(0.01)  # Simulate time delay for measurement
            return np.mean(signal_buffer, axis=0)

        # run experiment
        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda x: x,
                    result_shape=(len(freqs),),
                    pbar_n=round_n,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, fake_signal2real(ctx.root_data)
                ),
            )

        # record result
        self.last_result = FakeResult(
            freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(self, result: FakeResult | None = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals

        real_signals = fake_signal2real(signals)

        fig, ((ax,),) = make_plot_frame(1, 1)

        ax.plot(freqs, real_signals, label="Signal")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")

        return fig

    def save(
        self,
        filepath: str,
        result: FakeResult | None = None,
        comment: str | None = None,
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        comment = make_comment(cfg, comment)

    def load(self, filepath: str, **kwargs) -> FakeResult:
        freqs = np.linspace(4.5, 5.5, 201)  # MHz
        signals = (
            np.exp(-((freqs - 5.0) ** 2) / (2 * 0.1**2))
            + 0.1 * np.random.randn(len(freqs))
            + 1j * 0.1 * np.random.randn(len(freqs))
        )

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_result = FakeResult(freqs=freqs, signals=signals)

        return self.last_result
