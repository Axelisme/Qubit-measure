from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.utils.process import rotate2real

from .program import RBModuleCfg, RBSweepCfg, build_rb_program
from .sequence import make_seed_tables


@dataclass(frozen=True)
class RB_Result:
    sub_seeds: NDArray[np.int64]
    depths: NDArray[np.int64]
    signals2D: NDArray[np.complex128]


def rb_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    mask = np.any(np.isfinite(signals), axis=0)  # (depths, )
    mean_signals = np.full(
        (signals.shape[1],), np.nan, dtype=np.complex128
    )  # (depths,)
    mean_signals[mask] = np.nanmean(signals[..., mask], axis=0)
    return rotate2real(mean_signals).real


class RBCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RBModuleCfg
    sweep: RBSweepCfg
    seed: int
    n_seeds: int


@dataclass(frozen=True)
class RBAnalysis:
    epc: float
    fidelity: float


class RB_Exp(PersistableExperiment[RB_Result, RBCfg]):
    # depths/sub_seeds are integer sweeps on disk -> scale=IDENTITY (1.0).
    # axes inner-first [depths, sub_seeds] so native z == signals2D
    # (n_seeds, n_depths) with zero transpose.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("depths", "Depth", "a.u.", IDENTITY, np.int64),
            Axis("sub_seeds", "Entropy", "a.u.", IDENTITY, np.int64),
        ),
        z=ZSpec("signals2D", "Signal", "a.u."),
        result_type=RB_Result,
        cfg_type=RBCfg,
        tag="twotone/ge/rb",
    )

    def run(self, cfg: RBCfg, *, context: RunContext) -> RB_Result:
        run_cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            run_cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        depths = sweep2array(run_cfg.sweep.depth, allow_array=True).astype(np.int64)

        ss = np.random.SeedSequence(run_cfg.seed)
        entropys = np.array(
            [int(child.generate_state(1)[0]) for child in ss.spawn(run_cfg.n_seeds)],
            dtype=np.int64,
        )

        viewer = context.plots.liveplot_1d("measurement", "Depth", "Signal")

        signals_buffer = SignalBuffer(
            (len(entropys), len(depths)),
            on_update=lambda data: viewer.update(
                depths.astype(np.float64),
                rb_signal2real(data),
            ),
        )
        with Schedule(run_cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for _, step in sched.scan("seed", entropys.tolist()):
                tables = make_seed_tables(int(step.value), depths)
                builder = step.prog_builder(soc, soccfg)
                program = build_rb_program(builder, step.cfg.modules, tables)
                _ = builder.run_program(program)

        return RB_Result(
            sub_seeds=entropys,
            depths=depths,
            signals2D=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RB_Result],
        options: None,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        result = source.result
        del options

        depths = result.depths
        signals2D = result.signals2D

        real_signals_avg = rb_signal2real(signals2D)
        depths_f = depths.astype(np.float64)

        decay_time, decay_err, fit_signals, _ = fit_decay(depths_f, real_signals_avg)

        p = np.exp(-1.0 / decay_time)
        p_err = p / (decay_time**2) * decay_err
        epc = (1.0 - p) / 2.0  # (1 - p)(d - 1) / d, d = 2
        epc_err = p_err / 2.0
        fidelity = 1.0 - epc
        fidelity_err = epc_err

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        for si in range(signals2D.shape[0]):
            per_seed = rotate2real(signals2D[si]).real
            ax.plot(
                depths_f,
                per_seed,
                marker=".",
                linestyle="None",
                color="gray",
                alpha=0.3,
                markersize=3,
            )

        ax.plot(
            depths_f,
            real_signals_avg,
            marker="o",
            linestyle="None",
            label="Average",
            zorder=2,
        )
        ax.plot(depths_f, fit_signals, "-", color="red", label="Fit", zorder=3)

        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title(
            f"RB: EPC = {epc:.2e}, Fidelity = {fidelity:.6f} ± {fidelity_err:.6f}"
        )
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return RBAnalysis(epc=float(epc), fidelity=float(fidelity))
