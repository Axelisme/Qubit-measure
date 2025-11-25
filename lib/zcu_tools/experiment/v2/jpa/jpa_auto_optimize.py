from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    make_ge_sweep,
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
    set_power_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import (
    HardTask,
    SoftTask,
    TaskConfig,
    TaskContext,
    run_task,
)
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.liveplot import LivePlotterScatter, MultiLivePlotter, instant_plot
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data


class JPAOptimizer:
    """
    Optimizer for JPA parameters using a two-phase approach:
    1. Flux-sliced 2D optimization: each flux grid point runs a dedicated freq/power search
    2. Fine 3D optimization: refine around the best flux slice with scipy-like sampling

    The schedule still minimizes flux switching overhead by grouping evaluations per flux
    and only changing flux when the 2D search for the current slice completes.
    """

    def __init__(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        total_points: int,
    ) -> None:
        self.total_points = total_points
        self.grid_phase_points = max(1, int(total_points * 0.6))
        self.fine_phase_points = max(0, total_points - self.grid_phase_points)

        # Convert sweeps to arrays for bounds
        self.flx_arr = sweep2array(flx_sweep)
        self.fpt_arr = sweep2array(fpt_sweep)
        self.pdr_arr = sweep2array(pdr_sweep)

        # Store bounds for fine optimization
        self.flx_bounds = (self.flx_arr.min(), self.flx_arr.max())
        self.fpt_bounds = (self.fpt_arr.min(), self.fpt_arr.max())
        self.pdr_bounds = (self.pdr_arr.min(), self.pdr_arr.max())

        # Calculate phase-1 schedule (flux slices + local 2D searches)
        self._setup_stage1_optimization()

        # State tracking
        self.phase = 1  # 1 = grid search, 2 = fine optimization
        self.current_idx = 0
        self.results: List[
            Tuple[float, float, float, float]
        ] = []  # (flx, fpt, pdr, snr)

        # Phase 2 state
        self.fine_opt_queue: List[Tuple[float, float, float]] = []
        self.fine_opt_initialized = False
        self.best_grid_point: Optional[Tuple[float, float, float]] = None
        self.best_grid_snr: Optional[float] = None

    def _setup_stage1_optimization(self) -> None:
        """
        Setup phase-1 parameters.
        Each flux slice gets a queue of 2D (freq, power) optimization points so that the
        outer loop maintains a low flux-switching rate while still refining rf settings.
        """
        n_total = self.grid_phase_points

        self.stage1_points: List[Tuple[float, float, float]] = []
        if n_total <= 0:
            self.n_flx_grid = 0
            self.n_points_per_flux = 0
            self.flx_grid_spacing = 0.0
            fpt_range = self.fpt_bounds[1] - self.fpt_bounds[0]
            pdr_range = self.pdr_bounds[1] - self.pdr_bounds[0]
            self.fpt_grid_spacing = fpt_range * 0.1
            self.pdr_grid_spacing = pdr_range * 0.1
            return

        base = max(1, int(np.cbrt(n_total)))
        n_flx = max(2, min(len(self.flx_arr), base))
        n_points_per_flux = max(1, int(np.ceil(n_total / n_flx)))

        flx_grid = np.linspace(self.flx_bounds[0], self.flx_bounds[1], n_flx)

        self.n_flx_grid = n_flx
        self.n_points_per_flux = n_points_per_flux

        self.flx_grid_spacing = (
            (self.flx_bounds[1] - self.flx_bounds[0]) / (n_flx - 1)
            if n_flx > 1
            else 0.0
        )

        n_per_dim = int(np.sqrt(n_points_per_flux))
        self.fpt_grid_spacing = (
            (self.fpt_bounds[1] - self.fpt_bounds[0]) / n_per_dim
            if n_per_dim > 1
            else (self.fpt_bounds[1] - self.fpt_bounds[0]) * 0.1
        )
        self.pdr_grid_spacing = (
            (self.pdr_bounds[1] - self.pdr_bounds[0]) / n_per_dim
            if n_per_dim > 1
            else (self.pdr_bounds[1] - self.pdr_bounds[0]) * 0.1
        )

        fpt_scale = max(
            (self.fpt_bounds[1] - self.fpt_bounds[0]) * 0.1,
            self.fpt_grid_spacing * 1.5,
        )
        pdr_scale = max(
            (self.pdr_bounds[1] - self.pdr_bounds[0]) * 0.1,
            self.pdr_grid_spacing * 1.5,
        )

        fpt_center = float((self.fpt_bounds[0] + self.fpt_bounds[1]) / 2)
        pdr_center = float((self.pdr_bounds[0] + self.pdr_bounds[1]) / 2)
        golden = (np.sqrt(5) - 1) / 2

        for idx, flx in enumerate(flx_grid):
            # Use deterministic offsets so each flux slice explores slightly different seeds
            offset = ((idx * golden) % 1.0) - 0.5
            init_fpt = float(
                np.clip(fpt_center + offset * self.fpt_grid_spacing, *self.fpt_bounds)
            )
            init_pdr = float(
                np.clip(pdr_center - offset * self.pdr_grid_spacing, *self.pdr_bounds)
            )

            fpt_pdr_seq = self._generate_2d_optimization_points(
                init_fpt=init_fpt,
                init_pdr=init_pdr,
                n_points=n_points_per_flux,
                fpt_scale=fpt_scale,
                pdr_scale=pdr_scale,
            )

            for fpt, pdr in fpt_pdr_seq:
                self.stage1_points.append((float(flx), fpt, pdr))
                if len(self.stage1_points) >= n_total:
                    break

            if len(self.stage1_points) >= n_total:
                break

        # Truncate/extend to exactly grid_phase_points
        if len(self.stage1_points) >= n_total:
            self.stage1_points = self.stage1_points[:n_total]
        elif self.stage1_points:
            deficit = n_total - len(self.stage1_points)
            self.stage1_points.extend(self.stage1_points[-1:] * deficit)
        else:
            # Fallback: repeat the center point to keep scheduling logic simple
            default_point = (
                float(flx_grid[0]) if len(flx_grid) else float(self.flx_bounds[0]),
                fpt_center,
                pdr_center,
            )
            self.stage1_points = [default_point for _ in range(max(1, n_total))]

    def _generate_2d_optimization_points(
        self,
        init_fpt: float,
        init_pdr: float,
        n_points: int,
        fpt_scale: float,
        pdr_scale: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate 2D local-search points (freq/power) around a seed using a simplex-like
        stencil followed by a shrinking golden-spiral pattern for exploration.
        """
        points: List[Tuple[float, float]] = []
        if n_points <= 0:
            return points

        # Initial stencil covers ±directions to quickly probe gradients
        stencil = [
            (init_fpt, init_pdr),
            (init_fpt + fpt_scale, init_pdr),
            (init_fpt - fpt_scale, init_pdr),
            (init_fpt, init_pdr + pdr_scale),
            (init_fpt, init_pdr - pdr_scale),
        ]

        for fpt, pdr in stencil:
            fpt_clamped = float(np.clip(fpt, *self.fpt_bounds))
            pdr_clamped = float(np.clip(pdr, *self.pdr_bounds))
            points.append((fpt_clamped, pdr_clamped))
            if len(points) >= n_points:
                return points

        golden = (np.sqrt(5) - 1) / 2
        remaining = n_points - len(points)

        for i in range(remaining):
            shrink = golden ** (i // 5)
            angle = 2 * np.pi * (i + 1) * golden

            fpt_offset = fpt_scale * shrink * np.cos(angle)
            pdr_offset = pdr_scale * shrink * np.sin(angle)

            fpt_new = float(np.clip(init_fpt + fpt_offset, *self.fpt_bounds))
            pdr_new = float(np.clip(init_pdr + pdr_offset, *self.pdr_bounds))
            points.append((fpt_new, pdr_new))

        return points[:n_points]

    def _setup_fine_optimization(self) -> None:
        """
        Setup fine optimization phase based on grid search results.
        Start from the best point found in phase 1 and perform full 3D optimization.
        """
        if not self.results:
            self.fine_opt_queue = []
            self.fine_opt_initialized = True
            return

        # Find the best point from grid search
        best_idx = max(range(len(self.results)), key=lambda i: self.results[i][3])
        best_flx, best_fpt, best_pdr, best_snr = self.results[best_idx]

        self.best_grid_point = (best_flx, best_fpt, best_pdr)
        self.best_grid_snr = best_snr

        # Generate 3D optimization points starting from the best point
        self.fine_opt_queue = self._generate_3d_optimization_points(
            best_flx, best_fpt, best_pdr, self.fine_phase_points
        )

        self.fine_opt_initialized = True

    def _generate_3d_optimization_points(
        self,
        init_flx: float,
        init_fpt: float,
        init_pdr: float,
        n_points: int,
    ) -> List[Tuple[float, float, float]]:
        """
        Generate 3D optimization trajectory points using Nelder-Mead-like simplex method.
        Performs full optimization in (flux, freq, power) space.

        The flux search range is set to cover at least the adjacent grid points from phase 1,
        ensuring flux precision can improve beyond the grid resolution.
        """
        points: List[Tuple[float, float, float]] = []

        # Scale factors for search region
        # For flux: use at least 1.5x the grid spacing to cover adjacent points
        # For freq/power: use 10% of total range or grid spacing, whichever is larger
        flx_scale = max(
            (self.flx_bounds[1] - self.flx_bounds[0]) * 0.1,
            self.flx_grid_spacing * 1.5,  # Cover adjacent grid points
        )
        fpt_scale = max(
            (self.fpt_bounds[1] - self.fpt_bounds[0]) * 0.1,
            self.fpt_grid_spacing * 1.5,
        )
        pdr_scale = max(
            (self.pdr_bounds[1] - self.pdr_bounds[0]) * 0.1,
            self.pdr_grid_spacing * 1.5,
        )

        # Initial simplex vertices (4 points for 3D)
        # Use a regular tetrahedron pattern
        simplex = [
            (init_flx, init_fpt, init_pdr),  # Center point
            (init_flx + flx_scale, init_fpt, init_pdr),  # +flux
            (
                init_flx - flx_scale,
                init_fpt,
                init_pdr,
            ),  # -flux (explore both directions)
            (init_flx, init_fpt + fpt_scale, init_pdr),  # +freq
            (init_flx, init_fpt, init_pdr + pdr_scale),  # +power
        ]

        # Add simplex points first
        for flx, fpt, pdr in simplex:
            flx_clamped = float(np.clip(flx, *self.flx_bounds))
            fpt_clamped = float(np.clip(fpt, *self.fpt_bounds))
            pdr_clamped = float(np.clip(pdr, *self.pdr_bounds))
            points.append((flx_clamped, fpt_clamped, pdr_clamped))

        # Generate additional points using 3D Fibonacci/golden spiral pattern
        golden = (np.sqrt(5) - 1) / 2  # Golden ratio
        n_remaining = n_points - len(points)

        # Use progressively shrinking search radius
        for i in range(n_remaining):
            # Shrink scale progressively but slower to maintain exploration
            # Start shrinking after exploring the initial volume
            shrink = golden ** (i // 6)  # Slower shrinking than before

            # 3D spherical coordinates with golden angle distribution
            # This gives quasi-uniform distribution on a sphere
            theta = 2 * np.pi * i * golden  # Azimuthal angle
            phi = np.arccos(1 - 2 * ((i * golden) % 1))  # Polar angle

            # Convert to Cartesian offsets
            flx_offset = flx_scale * shrink * np.sin(phi) * np.cos(theta)
            fpt_offset = fpt_scale * shrink * np.sin(phi) * np.sin(theta)
            pdr_offset = pdr_scale * shrink * np.cos(phi)

            flx_new = float(np.clip(init_flx + flx_offset, *self.flx_bounds))
            fpt_new = float(np.clip(init_fpt + fpt_offset, *self.fpt_bounds))
            pdr_new = float(np.clip(init_pdr + pdr_offset, *self.pdr_bounds))
            points.append((flx_new, fpt_new, pdr_new))

        return points[:n_points]

    def next_params(
        self, i: int, last_snr: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get the next parameter combination to evaluate.

        Args:
            i: Current iteration index
            last_snr: SNR result from the previous iteration (None for first call)

        Returns:
            Tuple of (flux, freq, power) or None if optimization is complete
        """
        # Store result from previous iteration
        if last_snr is not None and i > 0:
            prev_params = self._get_previous_params(i - 1)
            if prev_params is not None:
                self.results.append((*prev_params, last_snr))

        # Check if optimization is complete
        if i >= self.total_points:
            return None

        # Phase 1: Grid search
        if i < self.grid_phase_points:
            if i < len(self.stage1_points):
                return self.stage1_points[i]
            else:
                # Fallback: repeat last point if schedule is shorter
                return self.stage1_points[-1] if self.stage1_points else None

        # Transition to phase 2
        if self.phase == 1:
            self.phase = 2
            # Store last grid result before transitioning
            if last_snr is not None and len(self.results) < self.grid_phase_points:
                prev_params = self._get_previous_params(i - 1)
                if (
                    prev_params is not None
                    and (*prev_params, last_snr) not in self.results
                ):
                    self.results.append((*prev_params, last_snr))
            self._setup_fine_optimization()

        # Phase 2: Fine optimization
        fine_idx = i - self.grid_phase_points
        if fine_idx < len(self.fine_opt_queue):
            return self.fine_opt_queue[fine_idx]

        # All points exhausted
        return None

    def _get_previous_params(self, idx: int) -> Optional[Tuple[float, float, float]]:
        """Get parameters that were used at the given index."""
        if idx < len(self.stage1_points):
            return self.stage1_points[idx]
        fine_idx = idx - self.grid_phase_points
        if 0 <= fine_idx < len(self.fine_opt_queue):
            return self.fine_opt_queue[fine_idx]
        return None

    def get_optimal_params(self) -> Tuple[float, float, float]:
        """
        Get the optimal parameters found during optimization.
        Uses scipy.optimize.minimize with collected data for final refinement.

        Returns:
            Tuple of (optimal_flux, optimal_freq, optimal_power)
        """
        if not self.results:
            raise ValueError("No results available. Run optimization first.")

        # Find best result from collected data
        best_idx = max(range(len(self.results)), key=lambda i: self.results[i][3])
        best_flx, best_fpt, best_pdr, best_snr = self.results[best_idx]

        # Use scipy.optimize for final refinement based on interpolated surface
        # Create interpolation from collected data points
        try:
            from scipy.interpolate import RBFInterpolator

            points = np.array([(r[0], r[1], r[2]) for r in self.results])
            snrs = np.array([r[3] for r in self.results])

            # Build interpolator (negative because we minimize)
            rbf = RBFInterpolator(points, -snrs, kernel="thin_plate_spline")

            # Optimize using interpolated surface
            result = minimize(
                lambda x: rbf(x.reshape(1, -1))[0],
                x0=np.array([best_flx, best_fpt, best_pdr]),
                bounds=[self.flx_bounds, self.fpt_bounds, self.pdr_bounds],
                method="L-BFGS-B",
            )

            if result.success:
                return tuple(result.x)  # type: ignore
        except Exception:
            pass  # Fall back to best measured point

        return (best_flx, best_fpt, best_pdr)


JPAOptimizeResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class JPAOptTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAAutoOptimizeExperiment(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: JPAOptTaskConfig, num_points: int
    ) -> JPAOptimizeResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        flx_sweep = cfg["sweep"]["jpa_flux"]
        fpt_sweep = cfg["sweep"]["jpa_freq"]
        pdr_sweep = cfg["sweep"]["jpa_power"]

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        optimizer = JPAOptimizer(flx_sweep, fpt_sweep, pdr_sweep, num_points)

        # (num_points, [flux, freq, power])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)

        def update_fn(i, ctx, _):
            ctx.env_dict["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.data[i - 1])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # TODO: Better way to early stop
                raise RuntimeError("No more parameters to optimize.")

            params[i, :] = cur_params
            set_flux_in_dev_cfg(ctx.cfg["dev"], params[i, 0], label="jpa_flux_dev")
            set_freq_in_dev_cfg(ctx.cfg["dev"], 1e6 * params[i, 1], label="jpa_rf_dev")
            set_power_in_dev_cfg(ctx.cfg["dev"], params[i, 2], label="jpa_rf_dev")

        # initialize figure and axes
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        instant_plot(fig, figsize)  # show the figure immediately

        with MultiLivePlotter(
            fig,
            plotters=dict(
                iter_scatter=LivePlotterScatter(
                    "Iteration", "SNR (a.u.)", existed_axes=[[ax_iter]]
                ),
                flux_scatter=LivePlotterScatter(
                    "JPA Flux value (a.u.)", "SNR (a.u.)", existed_axes=[[ax_flux]]
                ),
                freq_scatter=LivePlotterScatter(
                    "JPA Frequency (MHz)", "SNR (a.u.)", existed_axes=[[ax_freq]]
                ),
                power_scatter=LivePlotterScatter(
                    "JPA Power (dBm)", "SNR (a.u.)", existed_axes=[[ax_power]]
                ),
            ),
        ) as viewer:
            colors = np.full(num_points, "b")

            def plot_fn(ctx: TaskContext) -> None:
                idx: int = ctx.env_dict["index"]
                snrs = np.abs(ctx.data)  # (num_points, )

                cur_flx, cur_fpt, cur_pdr = params[idx, :]

                fig.suptitle(
                    f"Iteration {idx}, Flux: {1e3 * cur_flx:.2g} (mA), Freq: {1e-3 * cur_fpt:.4g} (GHz), Power: {cur_pdr:.2g} (dBm)"
                )

                if optimizer.phase == 2:
                    colors[idx] = "r"

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, refresh=False
                )
                viewer.get_plotter("flux_scatter").update(
                    params[:, 0], snrs, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 1], snrs, refresh=False
                )
                viewer.get_plotter("power_scatter").update(
                    params[:, 2], snrs, refresh=False
                )
                viewer.refresh()

            results = run_task(
                task=SoftTask(
                    sweep_name="Iteration",
                    sweep_values=list(range(num_points)),
                    update_cfg_fn=update_fn,
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            (
                                prog := ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                )
                            )
                            and (
                                prog.acquire(
                                    soc,
                                    progress=False,
                                    callback=update_hook,
                                    record_stderr=True,
                                ),
                                prog.get_stderr(),
                            )
                        ),
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(results)

        plt.close(fig)

        self.last_cfg = cfg
        self.last_result = (params, signals)

        return params, signals

    def analyze(
        self, result: Optional[JPAOptimizeResultType] = None
    ) -> Tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, signals = result
        snrs = np.abs(signals)

        max_id = np.argmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config.figsize)
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        ax_iter.scatter(np.arange(len(snrs)), snrs, s=1)
        ax_iter.axhline(max_snr, color="r", ls="--", label=f"best = {max_snr:.2g}")
        ax_iter.scatter([max_id], [max_snr], color="r", marker="*")
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("SNR")
        ax_iter.legend()
        ax_iter.grid(True)

        def plot_ax(ax, param_idx, label_name) -> None:
            ax.scatter(params[:, param_idx], snrs, s=1)
            best_value = best_params[param_idx]
            ax.axvline(best_value, color="r", ls="--", label=f"best = {best_value:.2g}")
            ax.scatter([best_value], [max_snr], color="r", marker="*")
            ax.set_xlabel(label_name)
            ax.set_ylabel("SNR")
            ax.legend()
            ax.grid(True)

        plot_ax(ax_flux, 0, "JPA Flux value (a.u.)")
        plot_ax(ax_freq, 1, "JPA Frequency (MHz)")
        plot_ax(ax_power, 2, "JPA Power (dBm)")

        return float(best_params[0]), float(best_params[1]), float(best_params[2]), fig

    def save(
        self,
        filepath: str,
        result: Optional[JPAOptimizeResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/auto_optimize",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, signals = result

        filepath = Path(filepath)

        x_info = {
            "name": "Iteration",
            "unit": "a.u.",
            "values": np.arange(params.shape[0]),
        }

        save_data(
            filepath=str(filepath.with_name(filepath.name + "_params")),
            x_info=x_info,
            y_info={
                "name": "Parameter Type",
                "unit": "a.u.",
                "values": ["Flux", "Frequency", "Power"],
            },
            z_info={"name": "Parameters", "unit": "a.u.", "values": params},
            comment=comment,
            tag=tag + "/params",
            **kwargs,
        )

        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag + "/signals",
            **kwargs,
        )


if __name__ == "__main__":
    # Test JPAOptimizer with a synthetic SNR function
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Define parameter ranges
    flx_sweep: SweepCfg = {"start": -1.0, "stop": 1.0, "expts": 21, "step": 0.1}
    fpt_sweep: SweepCfg = {"start": 5000.0, "stop": 6000.0, "expts": 101, "step": 10.0}
    pdr_sweep: SweepCfg = {"start": -20.0, "stop": 0.0, "expts": 21, "step": 1.0}

    # True optimal point (unknown to the optimizer)
    # Note: Due to sinusoidal modulation, the actual optimal may shift slightly
    TRUE_OPT_FLX = 0.3
    TRUE_OPT_FPT = 5500.0
    TRUE_OPT_PDR = -10.0
    BASE_MAX_SNR = 40.0  # Base SNR at optimal point

    # Sinusoidal modulation parameters
    SIN_PARAMS = {
        # Flux modulation: creates periodic structure in flux space
        "flx_freq1": 2.5,  # Primary frequency (cycles per unit flux)
        "flx_amp1": 8.0,  # Amplitude of primary oscillation
        "flx_freq2": 7.0,  # Secondary frequency (higher harmonic)
        "flx_amp2": 3.0,  # Amplitude of secondary oscillation
        # Frequency modulation: resonance-like structure
        "fpt_freq1": 0.02,  # Oscillation frequency in MHz^-1
        "fpt_amp1": 6.0,  # Amplitude
        "fpt_freq2": 0.005,  # Slower modulation
        "fpt_amp2": 4.0,
        # Power modulation: saturation-like behavior
        "pdr_freq1": 0.3,  # Oscillation frequency in dBm^-1
        "pdr_amp1": 5.0,
        "pdr_freq2": 0.8,  # Higher frequency ripple
        "pdr_amp2": 2.0,
        # Cross-coupling terms
        "flx_fpt_coupling": 0.01,  # Flux-frequency coupling strength
        "flx_pdr_coupling": 0.05,  # Flux-power coupling strength
    }

    # Noise parameters
    NOISE_PARAMS = {
        "white_noise_std": 1.5,  # Gaussian measurement noise
        "multiplicative_noise_std": 0.03,  # Proportional to signal (3%)
        "drift_amplitude": 0.5,  # Slow drift amplitude
        "drift_freq": 0.01,  # Drift frequency (per measurement)
    }

    # Global counter for drift simulation
    measurement_counter = [0]

    def synthetic_snr(flx: float, fpt: float, pdr: float) -> float:
        """
        Synthetic SNR function with composite sinusoidal structure and realistic noise.

        The function includes:
        1. Gaussian envelope centered at the optimal point
        2. Multi-frequency sinusoidal modulation in each parameter
        3. Cross-coupling between parameters
        4. White noise (measurement uncertainty)
        5. Multiplicative noise (signal-dependent)
        6. Slow drift (systematic error)

        This creates a challenging optimization landscape with:
        - Multiple local maxima
        - Narrow global maximum
        - Realistic noise that obscures fine structure
        """
        # Normalized coordinates
        norm_flx = (flx - TRUE_OPT_FLX) / 0.5
        norm_fpt = (fpt - TRUE_OPT_FPT) / 200.0
        norm_pdr = (pdr - TRUE_OPT_PDR) / 5.0

        # Base Gaussian envelope
        gaussian_envelope = np.exp(-(norm_flx**2 + norm_fpt**2 + norm_pdr**2))

        # Sinusoidal modulations for each parameter
        # Flux modulation (creates periodic sweet spots)
        flx_mod = (
            SIN_PARAMS["flx_amp1"]
            * np.sin(2 * np.pi * SIN_PARAMS["flx_freq1"] * flx)
            * np.cos(np.pi * norm_flx)  # Envelope to keep modulation bounded
            + SIN_PARAMS["flx_amp2"]
            * np.sin(2 * np.pi * SIN_PARAMS["flx_freq2"] * flx + 0.7)
        )

        # Frequency modulation (resonance-like structure)
        fpt_centered = fpt - TRUE_OPT_FPT
        fpt_mod = SIN_PARAMS["fpt_amp1"] * np.sin(
            2 * np.pi * SIN_PARAMS["fpt_freq1"] * fpt_centered
        ) + SIN_PARAMS["fpt_amp2"] * np.cos(
            2 * np.pi * SIN_PARAMS["fpt_freq2"] * fpt_centered + 1.2
        )

        # Power modulation (saturation-like with ripples)
        pdr_centered = pdr - TRUE_OPT_PDR
        pdr_mod = (
            SIN_PARAMS["pdr_amp1"]
            * np.sin(2 * np.pi * SIN_PARAMS["pdr_freq1"] * pdr_centered)
            * (1 - 0.3 * np.tanh(pdr_centered / 3))  # Asymmetric saturation
            + SIN_PARAMS["pdr_amp2"]
            * np.sin(2 * np.pi * SIN_PARAMS["pdr_freq2"] * pdr_centered + 0.5)
        )

        # Cross-coupling terms (flux affects frequency response, etc.)
        coupling = SIN_PARAMS["flx_fpt_coupling"] * flx * fpt_centered * np.sin(
            np.pi * norm_flx
        ) + SIN_PARAMS["flx_pdr_coupling"] * flx * pdr_centered * np.cos(
            2 * np.pi * norm_pdr
        )

        # Combine all components
        base_snr = (
            BASE_MAX_SNR * gaussian_envelope
            + flx_mod * gaussian_envelope**0.5  # Modulation decays slower than base
            + fpt_mod * gaussian_envelope**0.7
            + pdr_mod * gaussian_envelope**0.6
            + coupling
        )

        # Add noise components
        # 1. White noise (measurement uncertainty)
        white_noise = np.random.normal(0, NOISE_PARAMS["white_noise_std"])

        # 2. Multiplicative noise (signal-dependent)
        mult_noise = base_snr * np.random.normal(
            0, NOISE_PARAMS["multiplicative_noise_std"]
        )

        # 3. Slow drift (systematic error that changes over time)
        measurement_counter[0] += 1
        drift = NOISE_PARAMS["drift_amplitude"] * np.sin(
            2 * np.pi * NOISE_PARAMS["drift_freq"] * measurement_counter[0]
        )

        # Final SNR with all noise components
        noisy_snr = base_snr + white_noise + mult_noise + drift

        # Ensure SNR is positive (physical constraint)
        return max(0.1, noisy_snr)

    def get_true_snr_surface(n_points: int = 50):
        """Generate the true (noiseless) SNR surface for visualization."""
        flx_vals = np.linspace(-1.0, 1.0, n_points)
        fpt_vals = np.linspace(5000.0, 6000.0, n_points)

        # Save current counter and temporarily disable noise
        saved_counter = measurement_counter[0]
        saved_noise = NOISE_PARAMS.copy()

        # Set noise to zero for true surface
        for key in NOISE_PARAMS:
            NOISE_PARAMS[key] = 0.0

        # Calculate SNR at each point (fix power at optimal for 2D slice)
        snr_flx_fpt = np.zeros((n_points, n_points))
        for i, flx in enumerate(flx_vals):
            for j, fpt in enumerate(fpt_vals):
                snr_flx_fpt[i, j] = synthetic_snr(flx, fpt, TRUE_OPT_PDR)

        # Restore noise parameters
        NOISE_PARAMS.update(saved_noise)
        measurement_counter[0] = saved_counter

        return flx_vals, fpt_vals, snr_flx_fpt

    # Create optimizer
    total_points = 200
    optimizer = JPAOptimizer(flx_sweep, fpt_sweep, pdr_sweep, total_points)

    print(f"Total points: {total_points}")
    print(f"Grid phase points: {optimizer.grid_phase_points}")
    print(f"Fine phase points: {optimizer.fine_phase_points}")
    print(
        f"Phase-1 structure: {optimizer.n_flx_grid} flux slices x {optimizer.n_points_per_flux} 2D-opt samples each"
    )
    print(f"Actual stage-1 points: {len(optimizer.stage1_points)}")
    print()

    # Run optimization
    all_params: List[Tuple[float, float, float]] = []
    all_snrs: List[float] = []
    phases: List[int] = []

    last_snr: Optional[float] = None
    for i in range(total_points):
        params = optimizer.next_params(i, last_snr)
        if params is None:
            print(f"Optimization completed early at iteration {i}")
            break

        flx, fpt, pdr = params
        snr = synthetic_snr(flx, fpt, pdr)

        all_params.append(params)
        all_snrs.append(snr)
        phases.append(optimizer.phase)

        last_snr = snr

        if i % 50 == 0:
            print(
                f"Iteration {i}: phase={optimizer.phase}, "
                f"params=({flx:.3f}, {fpt:.1f}, {pdr:.1f}), SNR={snr:.2f}"
            )

    # Get final results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)

    # Best measured point
    best_idx = np.argmax(all_snrs)
    best_measured = all_params[best_idx]
    print(
        f"Best measured point: flux={best_measured[0]:.4f}, "
        f"freq={best_measured[1]:.2f}, power={best_measured[2]:.2f}"
    )
    print(f"Best measured SNR: {all_snrs[best_idx]:.2f}")

    # Optimal point from scipy refinement
    try:
        opt_params = optimizer.get_optimal_params()
        opt_snr = synthetic_snr(*opt_params)
        print(
            f"\nScipy refined point: flux={opt_params[0]:.4f}, "
            f"freq={opt_params[1]:.2f}, power={opt_params[2]:.2f}"
        )
        print(f"SNR at refined point: {opt_snr:.2f}")
    except Exception as e:
        print(f"\nScipy refinement failed: {e}")
        opt_params = best_measured

    print(
        f"\nTrue optimal: flux={TRUE_OPT_FLX}, freq={TRUE_OPT_FPT}, power={TRUE_OPT_PDR}"
    )
    print(f"Base max SNR: {BASE_MAX_SNR}")

    # Convert to numpy arrays for plotting
    params_arr = np.array(all_params)
    snrs_arr = np.array(all_snrs)
    phases_arr = np.array(phases)

    # Generate true SNR surface for visualization
    print("\nGenerating true SNR surface for visualization...")
    flx_surface, fpt_surface, snr_surface = get_true_snr_surface(80)

    # Create visualization - now with 3x3 layout
    fig = plt.figure(figsize=(18, 16))

    # Plot 1: SNR evolution over iterations
    ax1 = fig.add_subplot(3, 3, 1)
    colors = ["blue" if p == 1 else "red" for p in phases_arr]
    ax1.scatter(range(len(snrs_arr)), snrs_arr, c=colors, s=10, alpha=0.7)
    ax1.axhline(BASE_MAX_SNR, color="green", ls="--", label="Base max SNR")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("SNR")
    ax1.set_title("SNR Evolution (Blue=Grid, Red=Fine)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: SNR vs Flux with sinusoidal structure visible
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.scatter(params_arr[:, 0], snrs_arr, c=colors, s=15, alpha=0.7)
    ax2.axvline(TRUE_OPT_FLX, color="green", ls="--", lw=2, label="True optimal")
    ax2.axvline(opt_params[0], color="red", ls="-", lw=2, label="Found optimal")
    ax2.set_xlabel("Flux (a.u.)")
    ax2.set_ylabel("SNR")
    ax2.set_title("SNR vs Flux (with sinusoidal modulation)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: SNR vs Frequency
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.scatter(params_arr[:, 1], snrs_arr, c=colors, s=15, alpha=0.7)
    ax3.axvline(TRUE_OPT_FPT, color="green", ls="--", lw=2, label="True optimal")
    ax3.axvline(opt_params[1], color="red", ls="-", lw=2, label="Found optimal")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("SNR")
    ax3.set_title("SNR vs Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: SNR vs Power
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(params_arr[:, 2], snrs_arr, c=colors, s=15, alpha=0.7)
    ax4.axvline(TRUE_OPT_PDR, color="green", ls="--", lw=2, label="True optimal")
    ax4.axvline(opt_params[2], color="red", ls="-", lw=2, label="Found optimal")
    ax4.set_xlabel("Power (dBm)")
    ax4.set_ylabel("SNR")
    ax4.set_title("SNR vs Power")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: True SNR surface (Flux vs Frequency at optimal power)
    ax5 = fig.add_subplot(3, 3, 5)
    im = ax5.pcolormesh(
        fpt_surface,
        flx_surface,
        snr_surface,
        cmap="viridis",
        shading="auto",
    )
    # Overlay sampled points
    ax5.scatter(
        params_arr[:, 1],
        params_arr[:, 0],
        c="white",
        s=5,
        alpha=0.5,
        edgecolors="none",
    )
    ax5.scatter(
        [TRUE_OPT_FPT],
        [TRUE_OPT_FLX],
        color="red",
        s=150,
        marker="*",
        label="True optimal",
        edgecolors="white",
        linewidths=1,
    )
    ax5.scatter(
        [opt_params[1]],
        [opt_params[0]],
        color="lime",
        s=150,
        marker="^",
        label="Found optimal",
        edgecolors="white",
        linewidths=1,
    )
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Flux (a.u.)")
    ax5.set_title(f"True SNR Surface @ Power={TRUE_OPT_PDR} dBm")
    ax5.legend(loc="upper right")
    fig.colorbar(im, ax=ax5, label="SNR")

    # Plot 6: 3D scatter of all evaluated points
    ax6 = fig.add_subplot(3, 3, 6, projection="3d")
    scatter = ax6.scatter(
        params_arr[:, 0],
        params_arr[:, 1],
        params_arr[:, 2],
        c=snrs_arr,
        cmap="viridis",
        s=20,
        alpha=0.7,
    )
    ax6.scatter(
        [TRUE_OPT_FLX],
        [TRUE_OPT_FPT],
        [TRUE_OPT_PDR],
        color="red",
        s=200,
        marker="*",
        label="True optimal",
    )
    ax6.scatter(
        [opt_params[0]],
        [opt_params[1]],
        [opt_params[2]],
        color="lime",
        s=200,
        marker="^",
        label="Found optimal",
    )
    ax6.set_xlabel("Flux")
    ax6.set_ylabel("Frequency")
    ax6.set_zlabel("Power")
    ax6.set_title("3D Parameter Space")
    ax6.legend()
    fig.colorbar(scatter, ax=ax6, label="SNR", shrink=0.5)

    # Plot 7: Sampling density per phase (Freq vs Power)
    ax7 = fig.add_subplot(3, 3, 7)
    phase1_mask = phases_arr == 1
    phase2_mask = phases_arr == 2
    ax7.scatter(
        params_arr[phase1_mask, 1],
        params_arr[phase1_mask, 2],
        c="blue",
        s=25,
        alpha=0.5,
        label=f"Grid search ({phase1_mask.sum()} pts)",
    )
    ax7.scatter(
        params_arr[phase2_mask, 1],
        params_arr[phase2_mask, 2],
        c="red",
        s=25,
        alpha=0.5,
        label=f"Fine opt ({phase2_mask.sum()} pts)",
    )
    ax7.scatter(
        [TRUE_OPT_FPT],
        [TRUE_OPT_PDR],
        color="green",
        s=200,
        marker="*",
        label="True optimal",
    )
    ax7.set_xlabel("Frequency (MHz)")
    ax7.set_ylabel("Power (dBm)")
    ax7.set_title("Freq-Power Sampling Distribution")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Histogram of SNR values by phase
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(
        snrs_arr[phase1_mask],
        bins=20,
        alpha=0.6,
        color="blue",
        label="Grid search",
        density=True,
    )
    ax8.hist(
        snrs_arr[phase2_mask],
        bins=20,
        alpha=0.6,
        color="red",
        label="Fine opt",
        density=True,
    )
    ax8.axvline(all_snrs[best_idx], color="green", ls="--", lw=2, label="Best found")
    ax8.set_xlabel("SNR")
    ax8.set_ylabel("Density")
    ax8.set_title("SNR Distribution by Phase")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Error analysis - distance from true optimal over iterations
    ax9 = fig.add_subplot(3, 3, 9)
    # Calculate normalized error for each measurement
    errors_flx = np.abs(params_arr[:, 0] - TRUE_OPT_FLX) / 2.0  # normalize by range
    errors_fpt = np.abs(params_arr[:, 1] - TRUE_OPT_FPT) / 1000.0
    errors_pdr = np.abs(params_arr[:, 2] - TRUE_OPT_PDR) / 20.0
    total_errors = np.sqrt(errors_flx**2 + errors_fpt**2 + errors_pdr**2)

    # Running minimum error
    running_min_error = np.minimum.accumulate(total_errors)

    ax9.plot(range(len(total_errors)), total_errors, ".", alpha=0.3, label="Each point")
    ax9.plot(
        range(len(running_min_error)),
        running_min_error,
        "r-",
        lw=2,
        label="Running best",
    )
    ax9.axvline(
        optimizer.grid_phase_points,
        color="gray",
        ls="--",
        label="Phase transition",
    )
    ax9.set_xlabel("Iteration")
    ax9.set_ylabel("Normalized Distance from True Optimal")
    ax9.set_title("Convergence Analysis")
    ax9.set_yscale("log")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("jpa_optimizer_test.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPlot saved as 'jpa_optimizer_test.png'")

    # Additional statistics
    print("\n" + "=" * 60)
    print("Additional Statistics")
    print("=" * 60)
    print(
        f"Grid phase SNR: mean={snrs_arr[phase1_mask].mean():.2f}, "
        f"max={snrs_arr[phase1_mask].max():.2f}"
    )
    print(
        f"Fine phase SNR: mean={snrs_arr[phase2_mask].mean():.2f}, "
        f"max={snrs_arr[phase2_mask].max():.2f}"
    )
    print(f"Final error (flux): {abs(opt_params[0] - TRUE_OPT_FLX):.4f}")
    print(f"Final error (freq): {abs(opt_params[1] - TRUE_OPT_FPT):.2f} MHz")
    print(f"Final error (power): {abs(opt_params[2] - TRUE_OPT_PDR):.2f} dBm")
