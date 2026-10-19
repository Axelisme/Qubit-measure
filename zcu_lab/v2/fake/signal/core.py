from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.experiment import MHZ_TO_HZ, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg


@dataclass(frozen=True)
class FakeResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FakeCfg(ExpCfgModel):
    sweep: SweepCfg = Field(
        default_factory=lambda: SweepCfg(start=4.5, stop=5.5, expts=201, step=0.005)
    )
    rounds: int = Field(default=100, ge=1)
    noise_scale: float = Field(default=0.1, ge=0)
    round_delay: float = Field(default=0.01, ge=0)


def fake_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FakeExp(PersistableExperiment[FakeResult, FakeCfg]):
    """Generate and average a noisy Gaussian spectrum without hardware pulses."""

    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FakeResult,
        cfg_type=FakeCfg,
        tag="fake",
    )

    def run(self, config: FakeCfg, *, context: RunContext) -> FakeResult:
        """Generate a Gaussian centered at 5 MHz on the requested frequency grid.

        Each round adds one complex Gaussian offset shared by the whole trace,
        then updates the running mean. Return frequencies in MHz and mean
        complex signals, not a fitted resonance. Cancellation keeps the mean
        of completed rounds; no acquisition leaves the signal buffer as NaN.
        """
        # Prepare a synthetic frequency axis and a live view of its mean magnitude.
        cfg = deepcopy(config)
        freqs = np.linspace(cfg.sweep.start, cfg.sweep.stop, cfg.sweep.expts)
        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Amplitude"
        )
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, fake_signal2real(data)),
        )

        # Average independent noisy traces, with a cancellable pause between rounds.
        signal_buffer = []
        for _ in range(cfg.rounds):
            if context.cancel_signal.is_set():
                break
            # Each round adds scalar complex noise to the whole Gaussian trace.
            raw_signal = (
                np.exp(-((freqs - 5.0) ** 2) / (2 * 0.1**2))
                + cfg.noise_scale * np.random.randn()
                + 1j * cfg.noise_scale * np.random.randn()
            )
            signal_buffer.append(raw_signal)
            signals_buffer.set(np.mean(signal_buffer, axis=0))
            context.cancel_signal.event.wait(cfg.round_delay)

        # Publish the last mean even when the requested rounds were interrupted.
        signals_buffer.trigger_update(flush=True)
        return FakeResult(freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FakeCfg, FakeResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the mean spectrum magnitude; no fit or calibration is inferred."""
        del options
        _, ax = plots.subplots("fit")
        ax.plot(
            source.result.freqs, fake_signal2real(source.result.signals), label="Signal"
        )
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
