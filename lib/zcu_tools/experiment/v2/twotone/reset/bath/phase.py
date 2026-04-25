from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    BathReset,
    BathResetCfg,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.base import cosfunc, fitcos
from zcu_tools.utils.process import rotate2real

# (phases, signals)
PhaseResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class PhaseModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class PhaseSweepCfg(ConfigBase):
    phase: SweepCfg


class PhaseCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PhaseModuleCfg
    sweep: PhaseSweepCfg


class PhaseExp(AbsExperiment[PhaseResult, PhaseCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: PhaseCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> PhaseResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        phases = sweep2array(
            cfg.sweep.phase,
            "phase",
            {
                "soccfg": soccfg,
                "gen_ch": modules.tested_reset.pi2_cfg.ch,
            },
        )

        phase_param = sweep2param("phase", cfg.sweep.phase)
        modules.tested_reset.set_param("pi2_phase", phase_param)

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, PhaseCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules
            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("phase", cfg.sweep.phase)],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    BathReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Phase (deg)", "Signal (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    pbar_n=cfg.rounds,
                    measure_fn=measure_fn,
                    result_shape=(len(phases),),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    phases, bathreset_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (phases, signals)

        return phases, signals

    def analyze(
        self, result: Optional[PhaseResult] = None
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        real_signals = bathreset_signal2real(signals)

        pOpt, _ = fitcos(phases, real_signals, fixedparams=[None, None, 1 / 360, None])
        y_fit = cosfunc(phases, *pOpt)

        init_phase = float(pOpt[3])

        max_phase = -init_phase
        min_phase = 180 - init_phase

        while abs(max_phase) > abs(max_phase - 360):
            max_phase -= 360
        while abs(max_phase) > abs(max_phase + 360):
            max_phase += 360
        while abs(min_phase) > abs(min_phase - 360):
            min_phase -= 360
        while abs(min_phase) > abs(min_phase + 360):
            min_phase += 360

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.plot(phases, real_signals, ".-", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(
            max_phase, color="C1", linestyle="--", label=f"max: {max_phase:.2f} deg"
        )
        ax.axvline(
            min_phase, color="C2", linestyle="--", label=f"min: {min_phase:.2f} deg"
        )
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return max_phase, min_phase, fig

    def save(
        self,
        filepath: str,
        result: Optional[PhaseResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/phase",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PhaseResult:
        signals, phases, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert phases is not None
        assert len(phases.shape) == 1 and len(signals.shape) == 1
        assert phases.shape == signals.shape

        phases = phases.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = PhaseCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (phases, signals)

        return phases, signals
