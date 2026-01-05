from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
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
    Readout,
    ReadoutCfg,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

from .util import calc_populations

# (gains, freqs, populations)
AcStarkResultType = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


class AcStarkTaskConfig(TaskConfig, ModularProgramCfg):
    stark_pulse1: PulseCfg
    stark_pulse2: PulseCfg
    readout: ReadoutCfg


class AcStarkExp(AbsExperiment[AcStarkResultType, AcStarkTaskConfig]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        if cfg["stark_pulse1"].get("block_mode", True):
            raise ValueError("Stark pulse 1 must be in block mode")

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        gain_sweep = cfg["sweep"].pop("gain")

        # uniform in square space
        freqs = sweep2array(cfg["sweep"]["freq"])  # predicted frequencies
        gains = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )

        freq_param = sweep2param("freq", cfg["sweep"]["freq"])
        Pulse.set_param(cfg["stark_pulse2"], "freq", freq_param)

        fig, axs = make_plot_frame(4, 2, figsize=(12, 10))
        axs[3][0].set_ylim(0.0, 1.0)
        axs[3][1].set_ylim(0.0, 1.0)

        def make_plotter2d(ax: Axes) -> LivePlotter2D:
            return LivePlotter2D(
                "Stark Pulse Gain (a.u.)",
                "Probe Frequency (MHz)",
                uniform=False,
                existed_axes=[[ax]],
                segment_kwargs=dict(vmin=0.0, vmax=1.0),
            )

        def make_plotter1d(ax: Axes) -> LivePlotter1D:
            return LivePlotter1D(
                "Probe Frequency (MHz)",
                "Population",
                existed_axes=[[ax]],
                segment_kwargs=dict(
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            )

        with MultiLivePlotter(
            fig,
            dict(
                g_2d=make_plotter2d(axs[0][0]),
                e_2d=make_plotter2d(axs[1][0]),
                o_2d=make_plotter2d(axs[2][0]),
                cur_1d=make_plotter1d(axs[3][0]),
            ),
        ) as viewer:

            def update_fn(i, ctx, gain) -> None:
                Pulse.set_param(ctx.cfg["stark_pulse1"], "gain", gain)
                ctx.env_dict["idx"] = i

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("g_2d").update(
                    gains, freqs, populations[..., 0], refresh=False
                )
                viewer.get_plotter("e_2d").update(
                    gains, freqs, populations[..., 1], refresh=False
                )
                viewer.get_plotter("o_2d").update(
                    gains, freqs, populations[..., 2], refresh=False
                )
                viewer.get_plotter("cur_1d").update(
                    freqs, populations[i, :].T, refresh=False
                )

                viewer.refresh()

            def measure_fn(ctx, update_hook):
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                        Pulse("stark_pulse1", ctx.cfg["stark_pulse1"]),
                        Pulse("stark_pulse2", ctx.cfg["stark_pulse2"]),
                        Readout("readout", ctx.cfg["readout"]),
                    ],
                ).acquire(
                    soc,
                    progress=False,
                    callback=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    population_radius=radius,
                )

            signals = run_task(
                task=SoftTask(
                    sweep_name="resonator gain",
                    sweep_values=gains.tolist(),
                    update_cfg_fn=update_fn,
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        raw2signal_fn=lambda raw: raw[0][0],
                        result_shape=(len(freqs), 2),
                        dtype=np.float64,
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        ac_coeff: float,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        cutoff: Optional[float] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, populations = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        gains2 = gains**2

        # plot the data and the fitted polynomial
        photons = ac_coeff * gains2

        # fig, ax1 = plt.subplots(figsize=config.figsize)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

        # Use NonUniformImage for better visualization with pdr^2 as x-axis
        im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 0].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        ax1.add_image(im)
        ax1.set_xlim(photons[0], photons[-1])
        ax1.set_ylim(freqs[0], freqs[-1])
        ax1.set_aspect("auto")
        ax1.set_title("Ground")
        ax1.set_xlabel(r"$\bar n$", fontsize=14)
        ax1.set_ylabel("Frequency (MHz)", fontsize=14)

        im = NonUniformImage(ax2, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 1].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        ax2.add_image(im)
        ax2.set_xlim(photons[0], photons[-1])
        ax2.set_ylim(freqs[0], freqs[-1])
        ax2.set_aspect("auto")
        ax2.set_title("Excited")
        ax2.set_xlabel(r"$\bar n$", fontsize=14)

        im = NonUniformImage(ax3, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 2].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        ax3.add_image(im)
        ax3.set_xlim(photons[0], photons[-1])
        ax3.set_ylim(freqs[0], freqs[-1])
        ax3.set_aspect("auto")
        ax3.set_title("Other")
        ax3.set_xlabel(r"$\bar n$", fontsize=14)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[AcStarkResultType] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/ac_stark",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        _filepath = Path(filepath)

        gains, freqs, populations = result

        # Ground state population
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_pop")),
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": populations[..., 0].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # Excited state population
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_e_pop")),
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": populations[..., 1].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: List[str], **kwargs) -> AcStarkResultType:
        g_filepath, e_filepath = filepath

        # Load ground populations
        g_pop, gains, freqs = load_data(g_filepath, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert g_pop.shape == (len(gains), len(freqs))

        # Load ground populations
        e_pop, gains, freqs = load_data(e_filepath, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert e_pop.shape == (len(gains), len(freqs))

        populations = np.stack((g_pop, e_pop), axis=-1)

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        self.last_cfg = None
        self.last_result = (gains, freqs, populations)

        return gains, freqs, populations
