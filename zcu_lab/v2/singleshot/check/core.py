from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.plotting.singleshot import plot_with_classified
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Pulse, Readout, Reset

from zcu_lab.v2._support.singleshot.util import raw_shots_to_signal


@dataclass(frozen=True)
class CheckResult:
    shots: NDArray[np.int64]
    signals: NDArray[np.complex128]


class CheckCfg(ConfigBase):
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    shots: Annotated[int, FieldMeta(label="Shots")]


@dataclass(frozen=True)
class CheckAnalyzeOptions:
    g_center: complex
    e_center: complex
    radius: float
    max_point: int = 5000


class CheckExp(Experiment[CheckResult, CheckCfg]):
    Options: ClassVar[type[CheckAnalyzeOptions]] = CheckAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[CheckResult, CheckCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("shots", "shot", "point", dtype=np.int64),),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=CheckResult,
        cfg_type=CheckCfg,
        cfg_schema_version="3.0",
        tag="singleshot/check",
    )

    def run(self, cfg: CheckCfg, *, context: RunContext) -> CheckResult:
        """Collect an IQ cloud after a fixed qubit preparation and probe.

        Repeat optional reset, optional init pulse, probe pulse, then readout
        for cfg.shots shots. Return shot indices and one complex IQ per shot,
        not an averaged response. Analyze classifies this cloud with supplied
        G/E centers and radius to inspect preparation and readout separation.

        Acquisition uses one round and cfg.shots reps on a copied cfg, and
        raw-shot conversion requires exactly one readout channel.
        """

        # Keep individual shots with a separate raw-shot program configuration.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)

        signals_buffer = SignalBuffer((cfg.shots,))
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg

            # Prepare the same state for every shot before probing and readout.
            program = (
                sched.prog_builder(
                    soc,
                    soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                )
                .build()
            )

            # Retain raw IQ; an interrupted acquisition leaves missing data.
            try:
                program.acquire(soc, progress=True, cancel_flag=sched.stop)
            except StoppedPartialAcquireError:
                sched.set_stop()
            else:
                signals_buffer.set(raw_shots_to_signal(program))
        signals = signals_buffer.array

        shots = np.arange(cfg.shots, dtype=np.int64)
        return CheckResult(shots=shots, signals=signals)

    def analyze(
        self,
        source: RunRecord[CheckCfg, CheckResult],
        options: CheckAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        signals = source.result.signals
        g_center, e_center, radius = options.g_center, options.e_center, options.radius
        _, ax = plots.subplots("fit", figsize=(6, 6))

        mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)
        ng = mask_g.sum() / signals.shape[0]
        ne = mask_e.sum() / signals.shape[0]
        no = mask_o.sum() / signals.shape[0]

        plot_with_classified(
            ax, signals, g_center, e_center, radius, max_point=options.max_point
        )

        ax.set_title(
            f"Population: Ground: {ng:.1%}, Excited: {ne:.1%}, Other: {no:.1%}"
        )
