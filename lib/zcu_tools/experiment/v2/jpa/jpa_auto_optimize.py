from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skopt import Optimizer
from skopt.space import Real
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
    1. Flux-sliced 2D optimization (70% points): fixed flux, optimize freq/power.
    2. Fine 3D optimization (30% points): full 3D bayesian optimization starting from phase 1 data.
    """

    def __init__(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        total_points: int,
    ) -> None:
        self.total_points = total_points
        self.phase1_total_points = int(0.7 * total_points)
        self.phase2_total_points = total_points - self.phase1_total_points

        # Convert sweeps to arrays for bounds
        self.flx_arr = sweep2array(flx_sweep)
        self.fpt_arr = sweep2array(fpt_sweep)
        self.pdr_arr = sweep2array(pdr_sweep)

        # Bounds
        self.flx_bounds = (self.flx_arr.min(), self.flx_arr.max())
        self.fpt_bounds = (self.fpt_arr.min(), self.fpt_arr.max())
        self.pdr_bounds = (self.pdr_arr.min(), self.pdr_arr.max())

        # Setup Phase 1 Schedule
        self._setup_phase1_schedule()

        # State
        self.results: List[
            Tuple[float, float, float, float]
        ] = []  # (flx, fpt, pdr, snr)
        self.current_slice_idx = 0
        self.points_in_current_slice = 0

        # Optimizers
        self.opt_2d: Optional[Optimizer] = None
        self.opt_3d: Optional[Optimizer] = None

        self.last_suggested_params: Optional[Tuple[float, float, float]] = None

    def _setup_phase1_schedule(self) -> None:
        """
        Determine flux slices and points per slice for Phase 1.
        """
        if self.phase1_total_points <= 0:
            self.flx_grid = []
            self.points_per_slice = []
            return

        # Heuristic: Balance number of slices vs points per slice
        # We want at least ~10 points per 2D optimization to be useful
        min_points_per_slice = 10
        max_slices = max(1, self.phase1_total_points // min_points_per_slice)

        # Also limit slices by available flux points resolution if it's small
        n_flx_available = len(self.flx_arr)
        n_slices = min(
            max_slices, n_flx_available, int(np.sqrt(self.phase1_total_points))
        )
        n_slices = max(1, n_slices)

        # Select flux points linearly spaced from the available array (or range)
        if n_flx_available > 1:
            # Pick n_slices indices evenly spaced
            indices = np.linspace(0, n_flx_available - 1, n_slices, dtype=int)
            self.flx_grid = self.flx_arr[indices]
        else:
            self.flx_grid = np.array([self.flx_arr[0]])

        # Distribute points
        base_points = self.phase1_total_points // n_slices
        remainder = self.phase1_total_points % n_slices

        self.points_per_slice = [base_points] * n_slices
        for i in range(remainder):
            self.points_per_slice[i] += 1

    @property
    def phase(self) -> int:
        """Current optimization phase (1 or 2)."""
        return 1 if len(self.results) < self.phase1_total_points else 2

    def _init_2d_optimizer(self) -> None:
        """Initialize a new 2D optimizer for the current flux slice."""
        n_points = self.points_per_slice[self.current_slice_idx]
        # Heuristic for initial random points in 2D: 30% or at least 3
        n_initial = max(3, min(n_points, int(0.3 * n_points)))

        self.opt_2d = Optimizer(
            dimensions=[
                Real(*self.fpt_bounds, name="freq"),
                Real(*self.pdr_bounds, name="power"),
            ],
            base_estimator="GP",
            acq_func="gp_hedge",
            n_initial_points=n_initial,
        )

    def _init_3d_optimizer(self) -> None:
        """Initialize 3D optimizer for Phase 2, pre-fed with Phase 1 data."""
        # For Phase 2, we might want fewer initial points if we already have data,
        # but skopt uses n_initial_points for random exploration.
        # Since we feed existing points, we can reduce random exploration or keep it to explore new regions.
        # Let's set a small number of random points to ensure we don't get stuck immediately if Phase 1 was bad.
        n_initial = max(3, min(self.phase2_total_points, 5))

        self.opt_3d = Optimizer(
            dimensions=[
                Real(*self.flx_bounds, name="flux"),
                Real(*self.fpt_bounds, name="freq"),
                Real(*self.pdr_bounds, name="power"),
            ],
            base_estimator="GP",
            acq_func="gp_hedge",
            n_initial_points=n_initial,
        )

        # Feed all Phase 1 results
        # skopt minimizes, so use -snr
        try:
            points = [[r[0], r[1], r[2]] for r in self.results]
            values = [-r[3] for r in self.results]
            if points:
                self.opt_3d.tell(points, values)
        except Exception:
            # In case of duplicates or other issues, just continue
            pass

    def next_params(
        self, i: int, last_snr: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        if i >= self.total_points:
            return None

        # 1. Handle Result from Previous Iteration
        if last_snr is not None and self.last_suggested_params is not None:
            flx, fpt, pdr = self.last_suggested_params
            self.results.append((flx, fpt, pdr, last_snr))

            # Tell the appropriate optimizer
            if self.phase == 1:
                # Note: We might have just switched phases if i == phase1_total_points
                # But the result belongs to the optimizer that generated it.
                # Since we process result before generating next, we need to know which optimizer generated it.
                # Simplification: If we are in Phase 1, update 2D.
                # If we just transitioned to Phase 2 (i.e. this is the first call of Phase 2),
                # the previous result was Phase 1, so we should technically update the last 2D optimizer.
                # However, since we re-feed ALL results to 3D optimizer anyway, strict updating of the discarded 2D optimizer isn't strictly necessary for future points,
                # BUT it is good practice if we wanted to inspect it.
                # For simplicity: Update opt_2d if it exists and we are still within its "scope" or just finished it.
                if self.opt_2d:
                    try:
                        self.opt_2d.tell([fpt, pdr], -last_snr)
                    except ValueError:
                        pass
            else:
                # Phase 2
                if self.opt_3d:
                    try:
                        self.opt_3d.tell([flx, fpt, pdr], -last_snr)
                    except ValueError:
                        pass

        # 2. Generate Next Parameters

        # Check if we are in Phase 1
        if i < self.phase1_total_points:
            # Check if we need to move to next slice or init first slice
            if self.opt_2d is None:
                self._init_2d_optimizer()
            elif (
                self.points_in_current_slice
                >= self.points_per_slice[self.current_slice_idx]
            ):
                self.current_slice_idx += 1
                self.points_in_current_slice = 0
                self._init_2d_optimizer()

            # Ask 2D optimizer
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The objective has been evaluated.*"
                )
                fpt, pdr = self.opt_2d.ask()  # type: ignore
            flx = float(self.flx_grid[self.current_slice_idx])

            self.points_in_current_slice += 1
            self.last_suggested_params = (flx, fpt, pdr)
            return (flx, fpt, pdr)

        # Phase 2
        else:
            if self.opt_3d is None:
                self._init_3d_optimizer()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The objective has been evaluated.*"
                )
                suggested = self.opt_3d.ask()  # type: ignore
            self.last_suggested_params = tuple(suggested)  # type: ignore
            return self.last_suggested_params  # type: ignore

    def get_optimal_params(self) -> Tuple[float, float, float]:
        """
        Get the optimal parameters found.
        """
        if not self.results:
            raise ValueError("No results available.")

        # If Phase 2 ran, ask the 3D optimizer model for the best point
        if self.opt_3d and self.opt_3d.Xi:
            best_idx = np.argmin(self.opt_3d.yi)
            return tuple(self.opt_3d.Xi[best_idx])  # type: ignore

        # Fallback to best measured point
        best_idx = max(range(len(self.results)), key=lambda i: self.results[i][3])
        return (
            self.results[best_idx][0],
            self.results[best_idx][1],
            self.results[best_idx][2],
        )


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
    total_points = 100
    optimizer = JPAOptimizer(flx_sweep, fpt_sweep, pdr_sweep, total_points)

    print(f"Total points: {total_points}")
    print(f"Phase 1 points: {optimizer.phase1_total_points}")
    print(f"Phase 2 points: {optimizer.phase2_total_points}")
    print(
        f"Flux slices: {len(optimizer.flx_grid) if hasattr(optimizer, 'flx_grid') else 'N/A'}"
    )
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

        if i % 20 == 0:
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

    # Optimal point from optimizer
    try:
        opt_params = optimizer.get_optimal_params()
        opt_snr = synthetic_snr(*opt_params)
        print(
            f"\nOptimizer suggested point: flux={opt_params[0]:.4f}, "
            f"freq={opt_params[1]:.2f}, power={opt_params[2]:.2f}"
        )
        print(f"SNR at suggested point: {opt_snr:.2f}")
    except Exception as e:
        print(f"\nOptimizer lookup failed: {e}")
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

    # Create visualization - 3x3 layout
    fig = plt.figure(figsize=(18, 16))

    # Plot 1: SNR evolution over iterations
    ax1 = fig.add_subplot(3, 3, 1)
    colors = ["blue" if p == 1 else "red" for p in phases_arr]
    ax1.scatter(range(len(snrs_arr)), snrs_arr, c=colors, s=10, alpha=0.7)
    ax1.axhline(BASE_MAX_SNR, color="green", ls="--", label="Base max SNR")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("SNR")
    ax1.set_title("SNR Evolution (Blue=Phase1, Red=Phase2)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: SNR vs Flux
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.scatter(params_arr[:, 0], snrs_arr, c=colors, s=15, alpha=0.7)
    ax2.axvline(TRUE_OPT_FLX, color="green", ls="--", lw=2, label="True optimal")
    ax2.axvline(opt_params[0], color="purple", ls="-", lw=2, label="Found optimal")
    ax2.set_xlabel("Flux (a.u.)")
    ax2.set_ylabel("SNR")
    ax2.set_title("SNR vs Flux")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: SNR vs Frequency
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.scatter(params_arr[:, 1], snrs_arr, c=colors, s=15, alpha=0.7)
    ax3.axvline(TRUE_OPT_FPT, color="green", ls="--", lw=2, label="True optimal")
    ax3.axvline(opt_params[1], color="purple", ls="-", lw=2, label="Found optimal")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("SNR")
    ax3.set_title("SNR vs Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: SNR vs Power
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(params_arr[:, 2], snrs_arr, c=colors, s=15, alpha=0.7)
    ax4.axvline(TRUE_OPT_PDR, color="green", ls="--", lw=2, label="True optimal")
    ax4.axvline(opt_params[2], color="purple", ls="-", lw=2, label="Found optimal")
    ax4.set_xlabel("Power (dBm)")
    ax4.set_ylabel("SNR")
    ax4.set_title("SNR vs Power")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: True SNR surface
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

    # Plot 6: 3D scatter
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
    ax6.set_xlabel("Flux")
    ax6.set_ylabel("Frequency")
    ax6.set_zlabel("Power")
    ax6.set_title("3D Parameter Space")
    ax6.legend()
    fig.colorbar(scatter, ax=ax6, label="SNR", shrink=0.5)

    # Plot 7: Frequency vs Power Sampling
    ax7 = fig.add_subplot(3, 3, 7)
    phase1_mask = phases_arr == 1
    phase2_mask = phases_arr == 2
    ax7.scatter(
        params_arr[phase1_mask, 1],
        params_arr[phase1_mask, 2],
        c="blue",
        s=25,
        alpha=0.5,
        label=f"Phase 1 ({phase1_mask.sum()} pts)",
    )
    ax7.scatter(
        params_arr[phase2_mask, 1],
        params_arr[phase2_mask, 2],
        c="red",
        s=25,
        alpha=0.5,
        label=f"Phase 2 ({phase2_mask.sum()} pts)",
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

    # Plot 8: Flux Sampling Histogram
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(
        params_arr[:, 0],
        bins=30,
        alpha=0.7,
        color="gray",
        label="All points",
    )
    ax8.axvline(TRUE_OPT_FLX, color="green", ls="--", lw=2, label="True optimal")
    ax8.set_xlabel("Flux")
    ax8.set_ylabel("Count")
    ax8.set_title("Flux Sampling Distribution")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Convergence (Error)
    ax9 = fig.add_subplot(3, 3, 9)
    errors_flx = np.abs(params_arr[:, 0] - TRUE_OPT_FLX) / 2.0
    errors_fpt = np.abs(params_arr[:, 1] - TRUE_OPT_FPT) / 1000.0
    errors_pdr = np.abs(params_arr[:, 2] - TRUE_OPT_PDR) / 20.0
    total_errors = np.sqrt(errors_flx**2 + errors_fpt**2 + errors_pdr**2)

    running_min_error = np.minimum.accumulate(total_errors)

    ax9.plot(
        range(len(total_errors)), total_errors, ".", alpha=0.3, label="Point Error"
    )
    ax9.plot(
        range(len(running_min_error)),
        running_min_error,
        "r-",
        lw=2,
        label="Best Error Found",
    )
    ax9.axvline(
        optimizer.phase1_total_points,
        color="gray",
        ls="--",
        label="Phase Transition",
    )
    ax9.set_xlabel("Iteration")
    ax9.set_ylabel("Normalized Distance from Optimal")
    ax9.set_title("Convergence Analysis")
    ax9.set_yscale("log")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("jpa_optimizer_test.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'jpa_optimizer_test.png'")
    plt.show()
