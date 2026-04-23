from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D, MultiLivePlot, make_plot_frame
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
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

from ..util import calc_populations

FreqPowerResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]


class FreqPowerModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class FreqPowerSweepCfg(BaseModel):
    freq: SweepCfg
    gain: SweepCfg


class FreqPowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqPowerModuleCfg
    sweep: FreqPowerSweepCfg


class FreqPowerExp(AbsExperiment[FreqPowerResult, FreqPowerCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqPowerCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> FreqPowerResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FreqPowerCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.probe_pulse.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        fig, axs = make_plot_frame(3, 1, plot_instant=True, figsize=(12, 6))

        with MultiLivePlot(
            fig,
            dict(
                plot_2d_g=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[0][0]]],
                ),
                plot_2d_e=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[1][0]]],
                ),
                plot_2d_o=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[1][1]]],
                ),
            ),
        ) as viewer:

            def plot_fn(ctx) -> None:
                populations = calc_populations(np.asarray(ctx.root_data))

                viewer.get_plotter("plot_2d_g").update(
                    gains, freqs, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    gains, freqs, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    gains, freqs, populations[..., 2], refresh=False
                )

                viewer.refresh()

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(freqs), 2),
                    dtype=np.float64,
                    pbar_n=1,
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda i, ctx, gain: (
                        ctx.cfg.modules.probe_pulse.set_param("gain", gain)
                    ),
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = cfg
        self.last_result = (gains, freqs, signals)

        return self.last_result

    def analyze(
        self,
        result: Optional[FreqPowerResult] = None,
        *,
        ac_coeff=None,
        log_scale=False,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, populations = result

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        fig, (ax_g, ax_e, ax_o) = plt.subplots(3, 1, figsize=(8, 10))

        im_g = ax_g.imshow(
            populations[..., 0],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_g.set_title("Ground State Population")
        ax_g.set_xlabel("Drive Gain (a.u.)")
        ax_g.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_g, ax=ax_g, label="Population (a.u.)")

        im_e = ax_e.imshow(
            populations[..., 1],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_e.set_title("Excited State Population")
        ax_e.set_xlabel("Drive Gain (a.u.)")
        ax_e.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_e, ax=ax_e, label="Population (a.u.)")

        im_o = ax_o.imshow(
            populations[..., 2],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_o.set_title("Other States Population")
        ax_o.set_xlabel("Drive Gain (a.u.)")
        ax_o.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_o, ax=ax_o, label="Population (a.u.)")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqPowerResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/mist/gain_freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _filepath = Path(filepath)

        gains, freqs, populations = result

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_population")),
            x_info={"name": "Drive gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Drive freq", "unit": "Hz", "values": 1e6 * freqs},
            z_info={
                "name": "Population",
                "unit": "a.u.",
                "values": populations[..., 0].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_e_population")),
            x_info={"name": "Drive gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Drive freq", "unit": "Hz", "values": 1e6 * freqs},
            z_info={
                "name": "Population",
                "unit": "a.u.",
                "values": populations[..., 1].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: list[str], **kwargs) -> FreqPowerResult:
        g_filepath, e_filepath = filepath

        # Load ground populations
        g_pop, gains, freqs, cfg = load_data(g_filepath, return_cfg=True, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert g_pop.shape == (len(gains), len(freqs))

        # Load excited populations
        e_pop, gains_e, Ts_e = load_data(e_filepath, **kwargs)
        assert gains_e is not None and Ts_e is not None
        assert e_pop.shape == (len(gains_e), len(Ts_e))
        assert np.array_equal(gains, gains_e) and np.array_equal(freqs, Ts_e)

        freqs = freqs * 1e-6  # Hz to MHz

        # Reconstruct signals shape: (gains, ts, 2)
        populations = np.stack([g_pop, e_pop], axis=-1)

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        self.last_cfg = FreqPowerCfg.validate_or_warn(cfg, source=g_filepath)
        self.last_result = (gains, freqs, populations)

        return gains, freqs, populations
