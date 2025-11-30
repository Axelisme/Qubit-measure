from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    SoftTask,
    TaskConfig,
    TaskContext,
    run_task,
)
from zcu_tools.liveplot import (
    LivePlotter1D,
    LivePlotter2D,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    TriggerCfg,
    TriggerReadout,
)
from zcu_tools.utils.datasaver import save_data

LookbackOptimizeResultType = Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
]


class LookbackOptTaskConfig(TaskConfig, ModularProgramCfg):
    pre_pulse: PulseCfg
    main_pulse: PulseCfg
    post_pulse: PulseCfg
    ro_trigger: TriggerCfg


class LookbackOptimizeExperiment(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: LookbackOptTaskConfig
    ) -> LookbackOptimizeResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert cfg["readout"]["pulse_cfg"]["waveform"]["style"] == "padding"

        if cfg.setdefault("reps", 1) != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            cfg["reps"] = 1

        assert "sweep" in cfg
        cfg["sweep"] = {
            "pre_len": cfg["sweep"]["pre_len"],
            "post_len": cfg["sweep"]["post_len"],
        }

        pre_lens = sweep2array(cfg["sweep"]["pre_len"], allow_array=True)
        post_lens = sweep2array(cfg["sweep"]["post_len"], allow_array=True)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                TriggerReadout(
                    "ro_trigger", cfg["ro_trigger"], gen_ch=cfg["main_pulse"]["ch"]
                )
            ],
        )
        Ts = prog.get_time_axis(ro_index=0) + cfg["ro_trigger"]["trig_offset"]
        assert isinstance(Ts, np.ndarray)

        fig, axs = make_plot_frame(2, 1, figsize=(8, 5))
        ax2d, ax1d = axs[0]

        with MultiLivePlotter(
            fig,
            plotters=dict(
                plot_2d=LivePlotter2D(
                    "Pre-Length (us)", "Post-Length (us)", existed_axes=[[ax2d]]
                ),
                plot_1d=LivePlotter1D("Time (us)", "Amplitude", existed_axes=[[ax1d]]),
            ),
        ) as viewer:

            def update_pre_len(i: int, ctx: TaskContext, pre_len: float) -> None:
                Pulse.set_param(ctx.cfg["pre_pulse"], "length", pre_len)
                ctx.env_dict["pre_len"] = pre_len
                ctx.env_dict["pre_len_i"] = i

            def update_post_len(i: int, ctx: TaskContext, post_len: float) -> None:
                Pulse.set_param(ctx.cfg["post_pulse"], "length", post_len)
                ctx.env_dict["post_len"] = post_len
                ctx.env_dict["post_len_i"] = i

            def plot_fn(ctx: TaskContext) -> None:
                pre_len_i = ctx.env_dict["pre_len_i"]
                post_len_i = ctx.env_dict["post_len_i"]
                pre_len = ctx.env_dict["pre_len"]
                post_len = ctx.env_dict["post_len"]
                main_len = ctx.cfg["main_pulse"]["length"]

                real_signals = np.abs(ctx.data)

                step1 = np.where(Ts > pre_len)[0][0]
                step2 = np.where(Ts > pre_len + main_len + post_len)[0][0]

                step1_signals = real_signals[..., step1:step2]
                step2_signals = real_signals[..., step2:]

                loss_step1 = np.mean(
                    np.abs(step1_signals - np.mean(step1_signals, axis=-1)),
                    axis=-1,
                )
                loss_step2 = np.mean(
                    np.abs(step2_signals - np.mean(step2_signals, axis=-1)),
                    axis=-1,
                )

                loss = loss_step1 + loss_step2

                viwer_2d = viewer.get_plotter("plot_2d")
                viwer_1d = viewer.get_plotter("plot_1d")

                viwer_2d.update(pre_lens, post_lens, loss)

                cur_signals = real_signals[pre_len_i, post_len_i]
                viwer_1d.update(Ts, cur_signals)

            results = run_task(
                task=SoftTask(
                    sweep_name="pre_len",
                    sweep_values=pre_lens.tolist(),
                    update_cfg_fn=update_pre_len,
                    sub_task=SoftTask(
                        sweep_name="post_len",
                        sweep_values=post_lens.tolist(),
                        update_cfg_fn=update_post_len,
                        sub_task=HardTask(
                            measure_fn=lambda ctx, update_hook: (
                                ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        TriggerReadout(
                                            "ro_trigger",
                                            ctx.cfg["ro_trigger"],
                                            gen_ch=ctx.cfg["main_pulse"]["ch"],
                                        ),
                                        Pulse("pre_pulse", ctx.cfg["pre_pulse"]),
                                        Pulse("main_pulse", ctx.cfg["main_pulse"]),
                                        Pulse("post_pulse", ctx.cfg["post_pulse"]),
                                    ],
                                ).acquire(soc, progress=False, callback=update_hook)
                            ),
                            raw2signal_fn=lambda raw: raw[0].dot([1, 1j]),
                            result_shape=(len(Ts),),
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(results)

        plt.close(fig)

        self.last_cfg = cfg
        self.last_result = (pre_lens, post_lens, Ts, signals)

        return pre_lens, post_lens, Ts, signals

    def analyze(
        self, result: Optional[LookbackOptimizeResultType] = None
    ) -> Tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

    def save(
        self,
        filepath: str,
        result: Optional[LookbackOptimizeResultType] = None,
        comment: Optional[str] = None,
        tag: str = "lookback/optimize",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pre_lens, post_lens, Ts, signals = result
