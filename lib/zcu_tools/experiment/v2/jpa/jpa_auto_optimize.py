from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skopt import Optimizer
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
    Optimizer for JPA parameters using a two-phase approach with scikit-optimize:
    1. Flux-sliced 2D optimization (80% points): fixed flux, optimize freq/power.
       - For each flux value, 80% grid search + 20% Bayesian optimization.
    2. Fine 3D optimization (20% points + savings): full 3D bayesian optimization
       starting from phase 1 data within a restricted flux range.
    """

    def __init__(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        total_points: int,
    ) -> None:
        self.total_points = total_points

        # Extract bounds from sweeps
        self.flx_bounds = (flx_sweep["start"], flx_sweep["stop"])
        self.fpt_bounds = (fpt_sweep["start"], fpt_sweep["stop"])
        self.pdr_bounds = (pdr_sweep["start"], pdr_sweep["stop"])

        # Budget allocation
        self.phase1_total_budget = int(0.8 * total_points)
        self.phase2_base_budget = total_points - self.phase1_total_budget
        self.phase2_extra_budget = 0  # accumulated savings from phase 1

        # Determine flux grid points (cube root of phase1 budget)
        self.num_flx_points = max(2, int(round(self.phase1_total_budget ** (1 / 3))))
        self.flx_grid = np.linspace(
            self.flx_bounds[0], self.flx_bounds[1], self.num_flx_points
        )
        self.flx_interval = (
            (self.flx_bounds[1] - self.flx_bounds[0]) / (self.num_flx_points - 1)
            if self.num_flx_points > 1
            else (self.flx_bounds[1] - self.flx_bounds[0])
        )

        # Budget per flux slice
        self.budget_per_flx = self.phase1_total_budget // self.num_flx_points

        # Calculate grid dimensions for 2D search (80% of budget_per_flx)
        grid_budget = int(0.8 * self.budget_per_flx)
        self.grid_size = max(2, int(np.sqrt(grid_budget)))
        self.fpt_grid = np.linspace(
            self.fpt_bounds[0], self.fpt_bounds[1], self.grid_size
        )
        self.pdr_grid = np.linspace(
            self.pdr_bounds[0], self.pdr_bounds[1], self.grid_size
        )

        # Grid spacings for convergence check
        self.fpt_spacing = (
            (self.fpt_bounds[1] - self.fpt_bounds[0]) / (self.grid_size - 1)
            if self.grid_size > 1
            else (self.fpt_bounds[1] - self.fpt_bounds[0])
        )
        self.pdr_spacing = (
            (self.pdr_bounds[1] - self.pdr_bounds[0]) / (self.grid_size - 1)
            if self.grid_size > 1
            else (self.pdr_bounds[1] - self.pdr_bounds[0])
        )

        # State tracking
        self.phase = 1
        self.current_flx_idx = 0
        self.current_slice_iter = 0
        self.slice_grid_done = False

        # History storage
        self.history_X: List[List[float]] = []  # [flx, fpt, pdr]
        self.history_y: List[float] = []  # SNR values

        # Current slice data (for 2D optimizer)
        self.slice_X: List[List[float]] = []  # [fpt, pdr]
        self.slice_y: List[float] = []

        # Pre-generate grid points for current slice
        self._generate_slice_grid()

        # 2D optimizer for current flux slice (initialized after grid search)
        self.opt_2d: Optional[Optimizer] = None

        # 3D optimizer for phase 2
        self.opt_3d: Optional[Optimizer] = None
        self.phase2_iter = 0

        # Track actual iteration count
        self._iter_count = 0

        # Flag to track if slice ended due to early stop (convergence)
        self._slice_early_stop = False

    def _generate_slice_grid(self) -> None:
        """Generate grid points for the current flux slice."""
        self.slice_grid_points: List[List[float]] = []
        for fpt in self.fpt_grid:
            for pdr in self.pdr_grid:
                self.slice_grid_points.append([fpt, pdr])
        self.slice_grid_idx = 0

    def _check_phase1_convergence(self) -> bool:
        """
        Check convergence for phase 1 (2D optimization).
        Calculate per-dimension std of last 30 points.
        If std < grid spacing for any dimension, consider converged.
        """
        if len(self.slice_X) < 30:
            return False

        recent_points = np.array(self.slice_X[-30:])
        std_fpt = np.std(recent_points[:, 0])
        std_pdr = np.std(recent_points[:, 1])

        return std_fpt < self.fpt_spacing or std_pdr < self.pdr_spacing

    def _check_phase2_convergence(self) -> bool:
        """
        Check convergence for phase 2 (3D optimization).
        Calculate std of last 50 SNR values from phase 2 only.
        If std < mean / 5, consider converged.
        """
        # Only check convergence based on phase 2 points
        if self.phase2_iter < 50:
            return False

        # Get only phase 2 SNR values (last phase2_iter points in history_y)
        recent_snrs = np.array(self.history_y[-min(50, self.phase2_iter) :])
        std_snr = np.std(recent_snrs)
        mean_snr = np.mean(recent_snrs)

        if mean_snr <= 0:
            return False

        return std_snr < mean_snr / 5

    def _init_phase2(self) -> None:
        """Initialize phase 2 with restricted flux range and warm start."""
        self.phase = 2
        self.phase2_iter = 0

        # Find best flux from phase 1
        best_idx = int(np.argmax(self.history_y))
        best_flx = self.history_X[best_idx][0]

        # Restricted flux range: best_flx ± 1.25 * flx_interval
        flx_range = 1.25 * self.flx_interval
        flx_lo = max(self.flx_bounds[0], best_flx - flx_range)
        flx_hi = min(self.flx_bounds[1], best_flx + flx_range)

        # Create 3D optimizer
        self.opt_3d = Optimizer(
            dimensions=[
                (flx_lo, flx_hi),  # flux
                self.fpt_bounds,  # freq
                self.pdr_bounds,  # power
            ],
            acq_func="EI",
            n_initial_points=0,  # we'll warm start with historical data
        )

        # Filter historical points within the new flux range
        init_X = []
        init_y = []
        for x, y in zip(self.history_X, self.history_y):
            if flx_lo <= x[0] <= flx_hi:
                init_X.append(x)
                init_y.append(-y)  # skopt minimizes, we want to maximize SNR

        # Warm start with filtered historical data (fit=False to save time)
        if init_X:
            # Add all points except the last one without fitting
            for x, y in zip(init_X[:-1], init_y[:-1]):
                self.opt_3d.tell(x, y, fit=False)
            # Fit once on the last point
            self.opt_3d.tell(init_X[-1], init_y[-1], fit=True)

    def _get_phase1_point(self) -> Optional[Tuple[float, float, float]]:
        """Get next point for phase 1 optimization."""
        current_flx = self.flx_grid[self.current_flx_idx]

        # Check if we're still in grid search phase
        if self.slice_grid_idx < len(self.slice_grid_points):
            fpt, pdr = self.slice_grid_points[self.slice_grid_idx]
            self.slice_grid_idx += 1
            return (current_flx, fpt, pdr)

        # Grid search done, switch to 2D Bayesian optimization
        if not self.slice_grid_done:
            self.slice_grid_done = True
            # Initialize 2D optimizer
            self.opt_2d = Optimizer(
                dimensions=[self.fpt_bounds, self.pdr_bounds],
                acq_func="EI",
                n_initial_points=0,
            )
            # Tell optimizer about grid search results (fit=False for speed)
            if self.slice_X and self.slice_y:
                for x, y in zip(self.slice_X, self.slice_y):
                    self.opt_2d.tell(x, -y, fit=False)  # minimize negative SNR
                # Fit once at the end
                self.opt_2d.tell(self.slice_X[-1], -self.slice_y[-1], fit=True)

        # Check if we've used up budget for this slice
        remaining_budget = self.budget_per_flx - self.current_slice_iter
        if remaining_budget <= 0:
            self._slice_early_stop = False  # Budget exhausted, not early stop
            return None  # Signal to move to next slice

        # Check convergence
        if self._check_phase1_convergence():
            self._slice_early_stop = True  # Converged, save remaining for phase 2
            return None  # Signal to move to next slice

        # Get next point from 2D optimizer
        if self.opt_2d is not None:
            next_2d = self.opt_2d.ask()
            return (current_flx, next_2d[0], next_2d[1])

        return None

    def _get_phase2_point(self) -> Optional[Tuple[float, float, float]]:
        """Get next point for phase 2 optimization."""
        if self.opt_3d is None:
            return None

        # Check budget
        total_phase2_budget = self.phase2_base_budget + self.phase2_extra_budget
        if self.phase2_iter >= total_phase2_budget:
            return None

        # Check convergence
        if self._check_phase2_convergence():
            return None

        # Get next point from 3D optimizer
        next_3d = self.opt_3d.ask()
        return (next_3d[0], next_3d[1], next_3d[2])

    def _advance_to_next_slice(self, early_stop: bool = False) -> None:
        """Move to the next flux slice.

        Args:
            early_stop: If True, remaining budget is saved for phase 2.
                        If False (budget exhausted), no extra budget is added.
        """
        # Only save remaining budget if early stopped (converged)
        if early_stop:
            remaining = self.budget_per_flx - self.current_slice_iter
            if remaining > 0:
                self.phase2_extra_budget += remaining

        self.current_flx_idx += 1
        self.current_slice_iter = 0
        self.slice_grid_done = False
        self.slice_X = []
        self.slice_y = []
        self.opt_2d = None

        if self.current_flx_idx < self.num_flx_points:
            self._generate_slice_grid()

    @property
    def phase(self) -> int:
        return self._phase

    @phase.setter
    def phase(self, value: int) -> None:
        self._phase = value

    def next_params(
        self, i: int, last_snr: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        if i >= self.total_points:
            return None

        # Record last result
        if last_snr is not None and i > 0:
            # Update history
            if len(self.history_X) > 0:
                self.history_y.append(last_snr)

                # Update slice data for phase 1
                if self.phase == 1 and len(self.slice_X) > 0:
                    self.slice_y.append(last_snr)
                    # Tell 2D optimizer if active
                    if self.opt_2d is not None and len(self.slice_y) > len(
                        self.slice_grid_points
                    ):
                        self.opt_2d.tell(self.slice_X[-1], -last_snr)

                # Tell 3D optimizer if in phase 2
                if self.phase == 2 and self.opt_3d is not None:
                    self.opt_3d.tell(self.history_X[-1], -last_snr)

        # Get next point based on current phase
        if self.phase == 1:
            point = self._get_phase1_point()

            if point is None:
                # Move to next slice or phase 2
                self._advance_to_next_slice(early_stop=self._slice_early_stop)

                if self.current_flx_idx >= self.num_flx_points:
                    # All flux slices done, move to phase 2
                    self._init_phase2()
                    return self.next_params(i, None)
                else:
                    # Try again with new slice
                    return self.next_params(i, None)

            # Record point
            self.history_X.append(list(point))
            self.slice_X.append([point[1], point[2]])  # [fpt, pdr]
            self.current_slice_iter += 1
            self._iter_count += 1
            return point

        else:  # phase == 2
            point = self._get_phase2_point()

            if point is None:
                # Phase 2 converged or budget exhausted
                return None

            # Record point
            self.history_X.append(list(point))
            self.phase2_iter += 1
            self._iter_count += 1
            return point


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

        def update_fn(i, ctx, _) -> None:
            ctx.env_dict["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.data[i - 1])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # TODO: Better way to early stop
                raise KeyboardInterrupt("No more parameters to optimize.")

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

        instant_plot(fig)  # show the figure immediately

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
            y_info={"name": "Parameter Type", "unit": "a.u.", "values": [0, 1, 2]},
            z_info={"name": "Parameters", "unit": "a.u.", "values": params.T},
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
    # Test JPAOptimizer with a simulated SNR function

    def simulate_snr(
        flx: float, fpt: float, pdr: float, noise_std: float = 0.1
    ) -> float:
        """
        Simulate SNR as a combination of Gaussian and sinusoidal functions with noise.

        The SNR landscape has:
        - A Gaussian peak centered at (flx_0, fpt_0, pdr_0)
        - Sinusoidal modulation in the flux direction
        - Measurement noise
        """
        # True optimal point
        flx_0, fpt_0, pdr_0 = 0.5, 7000.0, -10.0

        # Gaussian component (main peak)
        sigma_flx, sigma_fpt, sigma_pdr = 0.15, 300.0, 3.0
        gauss = np.exp(
            -((flx - flx_0) ** 2) / (2 * sigma_flx**2)
            - ((fpt - fpt_0) ** 2) / (2 * sigma_fpt**2)
            - ((pdr - pdr_0) ** 2) / (2 * sigma_pdr**2)
        )

        # Sinusoidal modulation in flux (simulates flux-dependent behavior)
        sin_mod = 0.3 * np.sin(2 * np.pi * (flx - 0.2) / 0.4)

        # Sinusoidal modulation in frequency
        sin_fpt = 0.2 * np.sin(2 * np.pi * (fpt - 6800) / 500)

        # Base SNR with modulations
        snr_clean = 10.0 * gauss * (1 + sin_mod) * (1 + sin_fpt) + 1.0

        # Add measurement noise
        noise = np.random.normal(0, noise_std * snr_clean)
        snr = max(0.1, snr_clean + noise)

        return snr

    # Define sweep ranges
    flx_sweep: SweepCfg = {"start": 0.0, "stop": 1.0, "expts": 50, "step": 0.02}
    fpt_sweep: SweepCfg = {"start": 6500.0, "stop": 7500.0, "expts": 50, "step": 20.0}
    pdr_sweep: SweepCfg = {"start": -20.0, "stop": 0.0, "expts": 50, "step": 0.4}

    total_points = 500

    print("=" * 60)
    print("JPAOptimizer Test")
    print("=" * 60)
    print(f"Total points: {total_points}")
    print(f"Flux range: [{flx_sweep['start']}, {flx_sweep['stop']}]")
    print(f"Frequency range: [{fpt_sweep['start']}, {fpt_sweep['stop']}] MHz")
    print(f"Power range: [{pdr_sweep['start']}, {pdr_sweep['stop']}] dBm")
    print("True optimal: flx=0.5, fpt=7000 MHz, pdr=-10 dBm")
    print("=" * 60)

    # Create optimizer
    optimizer = JPAOptimizer(flx_sweep, fpt_sweep, pdr_sweep, total_points)

    print(f"Phase 1 budget: {optimizer.phase1_total_budget}")
    print(f"Phase 2 base budget: {optimizer.phase2_base_budget}")
    print(f"Number of flux slices: {optimizer.num_flx_points}")
    print(f"Budget per flux slice: {optimizer.budget_per_flx}")
    print(f"Grid size: {optimizer.grid_size}x{optimizer.grid_size}")
    print("=" * 60)

    # Run optimization
    params_list: List[Tuple[float, float, float]] = []
    snrs_list: List[float] = []
    phases_list: List[int] = []

    last_snr: Optional[float] = None
    for i in range(total_points):
        params = optimizer.next_params(i, last_snr)
        if params is None:
            print(f"Optimization stopped at iteration {i}")
            break

        flx, fpt, pdr = params
        snr = simulate_snr(flx, fpt, pdr, noise_std=0.05)

        params_list.append(params)
        snrs_list.append(snr)
        phases_list.append(optimizer.phase)

        last_snr = snr

        if i % 50 == 0:
            print(
                f"Iter {i:4d} | Phase {optimizer.phase} | "
                f"flx={flx:.4f}, fpt={fpt:.1f}, pdr={pdr:.2f} | SNR={snr:.3f}"
            )

    # Convert to arrays
    params_arr = np.array(params_list)
    snrs_arr = np.array(snrs_list)
    phases_arr = np.array(phases_list)

    # Find best result
    best_idx = np.argmax(snrs_arr)
    best_params = params_arr[best_idx]
    best_snr = snrs_arr[best_idx]

    print("=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Total iterations: {len(params_list)}")
    print(f"Phase 1 points: {np.sum(phases_arr == 1)}")
    print(f"Phase 2 points: {np.sum(phases_arr == 2)}")
    print(f"Phase 2 extra budget (savings): {optimizer.phase2_extra_budget}")
    print(f"Best SNR: {best_snr:.4f} at iteration {best_idx}")
    print(
        f"Best params: flx={best_params[0]:.4f}, "
        f"fpt={best_params[1]:.1f} MHz, pdr={best_params[2]:.2f} dBm"
    )
    print("True optimal: flx=0.5, fpt=7000 MHz, pdr=-10 dBm")
    print(
        f"Error: flx={abs(best_params[0] - 0.5):.4f}, "
        f"fpt={abs(best_params[1] - 7000):.1f} MHz, pdr={abs(best_params[2] + 10):.2f} dBm"
    )
    print("=" * 60)

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("JPAOptimizer Test Results", fontsize=14)

    # 1. SNR vs Iteration
    ax1 = fig.add_subplot(2, 3, 1)
    phase1_mask = phases_arr == 1
    phase2_mask = phases_arr == 2
    ax1.scatter(
        np.arange(len(snrs_arr))[phase1_mask],
        snrs_arr[phase1_mask],
        c="blue",
        s=5,
        alpha=0.6,
        label="Phase 1",
    )
    ax1.scatter(
        np.arange(len(snrs_arr))[phase2_mask],
        snrs_arr[phase2_mask],
        c="red",
        s=5,
        alpha=0.6,
        label="Phase 2",
    )
    ax1.axhline(best_snr, color="green", ls="--", label=f"Best={best_snr:.2f}")
    ax1.scatter([best_idx], [best_snr], c="green", s=100, marker="*", zorder=5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("SNR")
    ax1.set_title("SNR vs Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SNR vs Flux
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(
        params_arr[phase1_mask, 0],
        snrs_arr[phase1_mask],
        c="blue",
        s=5,
        alpha=0.6,
        label="Phase 1",
    )
    ax2.scatter(
        params_arr[phase2_mask, 0],
        snrs_arr[phase2_mask],
        c="red",
        s=5,
        alpha=0.6,
        label="Phase 2",
    )
    ax2.axvline(best_params[0], color="green", ls="--")
    ax2.axvline(0.5, color="orange", ls=":", label="True optimal")
    ax2.scatter([best_params[0]], [best_snr], c="green", s=100, marker="*", zorder=5)
    ax2.set_xlabel("Flux (a.u.)")
    ax2.set_ylabel("SNR")
    ax2.set_title("SNR vs Flux")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. SNR vs Frequency
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(
        params_arr[phase1_mask, 1],
        snrs_arr[phase1_mask],
        c="blue",
        s=5,
        alpha=0.6,
        label="Phase 1",
    )
    ax3.scatter(
        params_arr[phase2_mask, 1],
        snrs_arr[phase2_mask],
        c="red",
        s=5,
        alpha=0.6,
        label="Phase 2",
    )
    ax3.axvline(best_params[1], color="green", ls="--")
    ax3.axvline(7000, color="orange", ls=":", label="True optimal")
    ax3.scatter([best_params[1]], [best_snr], c="green", s=100, marker="*", zorder=5)
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("SNR")
    ax3.set_title("SNR vs Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. SNR vs Power
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(
        params_arr[phase1_mask, 2],
        snrs_arr[phase1_mask],
        c="blue",
        s=5,
        alpha=0.6,
        label="Phase 1",
    )
    ax4.scatter(
        params_arr[phase2_mask, 2],
        snrs_arr[phase2_mask],
        c="red",
        s=5,
        alpha=0.6,
        label="Phase 2",
    )
    ax4.axvline(best_params[2], color="green", ls="--")
    ax4.axvline(-10, color="orange", ls=":", label="True optimal")
    ax4.scatter([best_params[2]], [best_snr], c="green", s=100, marker="*", zorder=5)
    ax4.set_xlabel("Power (dBm)")
    ax4.set_ylabel("SNR")
    ax4.set_title("SNR vs Power")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 2D scatter: Flux vs Frequency
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(
        params_arr[:, 0],
        params_arr[:, 1],
        c=snrs_arr,
        s=10,
        cmap="viridis",
        alpha=0.7,
    )
    ax5.scatter([0.5], [7000], c="red", s=100, marker="x", label="True optimal")
    ax5.scatter(
        [best_params[0]], [best_params[1]], c="green", s=100, marker="*", label="Found"
    )
    ax5.set_xlabel("Flux (a.u.)")
    ax5.set_ylabel("Frequency (MHz)")
    ax5.set_title("Sampled Points (Flux vs Freq)")
    ax5.legend()
    plt.colorbar(scatter, ax=ax5, label="SNR")

    # 6. 2D scatter: Frequency vs Power
    ax6 = fig.add_subplot(2, 3, 6)
    scatter2 = ax6.scatter(
        params_arr[:, 1],
        params_arr[:, 2],
        c=snrs_arr,
        s=10,
        cmap="viridis",
        alpha=0.7,
    )
    ax6.scatter([7000], [-10], c="red", s=100, marker="x", label="True optimal")
    ax6.scatter(
        [best_params[1]], [best_params[2]], c="green", s=100, marker="*", label="Found"
    )
    ax6.set_xlabel("Frequency (MHz)")
    ax6.set_ylabel("Power (dBm)")
    ax6.set_title("Sampled Points (Freq vs Power)")
    ax6.legend()
    plt.colorbar(scatter2, ax=ax6, label="SNR")

    plt.tight_layout()
    plt.savefig("jpa_optimizer_test.png", dpi=150)
    print("Figure saved to jpa_optimizer_test.png")
    plt.show()
