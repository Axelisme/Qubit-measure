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
from typing_extensions import NotRequired

# Suppress skopt warnings about duplicate points
warnings.filterwarnings(
    "ignore",
    message="The objective has been evaluated at point .* before",
    category=UserWarning,
    module="skopt.optimizer.optimizer",
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    make_ge_sweep,
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
    set_power_in_dev_cfg,
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
    Multi-phase optimizer for JPA parameters using scikit-optimize:

    Phase 1 (50% of total budget):
        - Evenly distribute budget among all flux grid points
        - For each flux value: 100% LHS sampling (pure exploration)

    Phase 2+ (budget = total_points * (1/2)^n for phase n):
        - Select top 20% flux values by best SNR (ceil)
        - Generate 3 refinement points per selected flux at -0.5, 0, +0.5 of previous interval
        - For new flux points: 50% LHS + 50% Bayesian optimization
        - For existing flux points: 100% Bayesian optimization
        - Use nearest neighbor's best point as initial guess for BO
        - Terminates when next phase budget < 100, then use all remaining points
    """

    def __init__(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        total_points: int,
    ) -> None:
        from math import ceil

        from scipy.stats import qmc

        self.total_points = total_points

        # Extract bounds from sweeps
        self.flx_bounds = (flx_sweep["start"], flx_sweep["stop"])
        self.fpt_bounds = (fpt_sweep["start"], fpt_sweep["stop"])
        self.pdr_bounds = (pdr_sweep["start"], pdr_sweep["stop"])

        # Phase 1 budget = 50% of total
        self.phase1_budget = total_points // 2
        self.remaining_budget = total_points - self.phase1_budget

        # Determine flux grid points for phase 1 based on sweep expts ratios
        n_flx = max(1, flx_sweep["expts"])
        n_fpt = max(1, fpt_sweep["expts"])
        n_pdr = max(1, pdr_sweep["expts"])

        # Scale factor to map total product of expts to phase1_budget
        # n_flx_eff * n_fpt_eff * n_pdr_eff ~= phase1_budget
        # where n_i_eff = k * n_i
        total_expts_prod = n_flx * n_fpt * n_pdr
        k = (self.phase1_budget / total_expts_prod) ** (1 / 3)

        self.num_flx_points = max(2, int(round(k * n_flx)))
        self.flx_grid = np.linspace(
            self.flx_bounds[0], self.flx_bounds[1], self.num_flx_points
        )
        self.flx_interval = (
            (self.flx_bounds[1] - self.flx_bounds[0]) / (self.num_flx_points - 1)
            if self.num_flx_points > 1
            else (self.flx_bounds[1] - self.flx_bounds[0])
        )

        # Budget per flux slice in phase 1 (100% LHS)
        self.budget_per_flx = self.phase1_budget // self.num_flx_points

        # LHS sampler for 2D (freq, power) space
        self._lhs_sampler = qmc.LatinHypercube(d=2)
        self._ceil = ceil

        # State tracking
        self._phase = 1
        self._iter_count = 0
        self.current_flx_idx = 0
        self.current_slice_iter = 0
        self.last_flx: Optional[float] = None  # Track last measured flux value

        # Current slice LHS points and optimizer
        self._lhs_points: List[List[float]] = []
        self._lhs_idx = 0
        self.opt_2d: Optional[Optimizer] = None

        # Data storage per flux value
        # flux_data: flx -> [(fpt, pdr, snr), ...]
        self.flux_data: dict[float, List[Tuple[float, float, float]]] = {}
        # flux_best: flx -> (best_fpt, best_pdr, best_snr)
        self.flux_best: dict[float, Tuple[float, float, float]] = {}

        # Global history for compatibility
        self.history_X: List[List[float]] = []  # [flx, fpt, pdr]
        self.history_y: List[float] = []  # SNR values

        # Phase 2+ state
        self.phase_budget = self.phase1_budget  # Current phase budget
        self.prev_interval = self.flx_interval  # Previous phase interval
        self._refinement_flx_list: List[
            float
        ] = []  # Flux test points for current phase
        self._refinement_idx = 0
        self._current_refinement_flx: Optional[float] = None
        self._refinement_slice_iter = 0
        self._refinement_budget_per_flx = 0

        # Phase 2+ LHS and neighbor guess state
        self._refinement_lhs_budget = 0  # LHS budget for current refinement slice
        self._refinement_lhs_points: List[List[float]] = []
        self._refinement_lhs_idx = 0
        self._neighbor_guess: Optional[List[float]] = None  # [fpt, pdr] from neighbor
        self._neighbor_guess_used = False

        # Final phase flag: only enable optimizer in the final phase
        self._is_final_phase = False

        # Generate LHS points for first flux slice
        self._generate_lhs_samples()

    def _generate_lhs_samples(
        self,
        n_samples: Optional[int] = None,
        fpt_bounds: Optional[Tuple[float, float]] = None,
        pdr_bounds: Optional[Tuple[float, float]] = None,
    ) -> List[List[float]]:
        """
        Generate 2D LHS samples for current flux slice.

        Args:
            n_samples: Number of samples. If None, uses budget_per_flx.
            fpt_bounds: Frequency bounds. If None, uses self.fpt_bounds.
            pdr_bounds: Power bounds. If None, uses self.pdr_bounds.

        Returns:
            List of [fpt, pdr] samples.
        """
        if n_samples is None:
            n_samples = max(1, self.budget_per_flx)
        else:
            n_samples = max(1, n_samples)

        if fpt_bounds is None:
            fpt_bounds = self.fpt_bounds
        if pdr_bounds is None:
            pdr_bounds = self.pdr_bounds

        # Generate samples in [0, 1]^2
        samples = self._lhs_sampler.random(n=n_samples)
        # Scale to actual bounds
        fpt_range = fpt_bounds[1] - fpt_bounds[0]
        pdr_range = pdr_bounds[1] - pdr_bounds[0]
        lhs_points: List[List[float]] = []
        for s in samples:
            fpt = fpt_bounds[0] + s[0] * fpt_range
            pdr = pdr_bounds[0] + s[1] * pdr_range
            lhs_points.append([fpt, pdr])

        # For Phase 1 compatibility, also set instance variables
        self._lhs_points = lhs_points
        self._lhs_idx = 0

        return lhs_points

    def _get_flux_best_point(self, flx: float) -> Optional[Tuple[float, float, float]]:
        """Get the best (fpt, pdr, snr) for a given flux value."""
        return self.flux_best.get(flx)

    def _find_nearest_measured_flx(self, target_flx: float) -> Optional[float]:
        """Find the nearest flux value that has been measured."""
        if not self.flux_best:
            return None
        measured_flx_list = list(self.flux_best.keys())
        nearest = min(measured_flx_list, key=lambda x: abs(x - target_flx))
        return nearest

    def _get_restricted_bounds(
        self, center_fpt: float, center_pdr: float, phase: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get restricted 2D bounds centered on (center_fpt, center_pdr).
        Range shrinks by factor of 2^(phase-1).
        Returns intersection with original bounds.
        """
        shrink_factor = 2 ** (phase - 1)
        fpt_range = (self.fpt_bounds[1] - self.fpt_bounds[0]) / shrink_factor
        pdr_range = (self.pdr_bounds[1] - self.pdr_bounds[0]) / shrink_factor

        fpt_lo = max(self.fpt_bounds[0], center_fpt - fpt_range / 2)
        fpt_hi = min(self.fpt_bounds[1], center_fpt + fpt_range / 2)
        pdr_lo = max(self.pdr_bounds[0], center_pdr - pdr_range / 2)
        pdr_hi = min(self.pdr_bounds[1], center_pdr + pdr_range / 2)

        return ((fpt_lo, fpt_hi), (pdr_lo, pdr_hi))

    def _init_2d_optimizer_for_flux(
        self,
        target_flx: float,
        fpt_bounds: Tuple[float, float],
        pdr_bounds: Tuple[float, float],
    ) -> Optimizer:
        """
        Initialize 2D optimizer using ONLY data measured at the target flux.
        This avoids model pollution from neighbor flux values where the optimal
        (fpt, pdr) may be very different.

        Args:
            target_flx: The flux value to optimize for.
            fpt_bounds: Frequency bounds for optimization.
            pdr_bounds: Power bounds for optimization.

        Returns:
            Initialized Optimizer instance.
        """
        opt = Optimizer(
            dimensions=[fpt_bounds, pdr_bounds],
            base_estimator="ET",
            acq_func="EI",
            n_initial_points=0,
            n_jobs=-1,
            acq_optimizer="auto",
        )

        # Only use data from THIS flux value (avoid model pollution)
        if target_flx in self.flux_data:
            init_points: List[Tuple[List[float], float]] = []
            for fpt, pdr, snr in self.flux_data[target_flx]:
                # Check if point is within the current bounds
                if (
                    fpt_bounds[0] <= fpt <= fpt_bounds[1]
                    and pdr_bounds[0] <= pdr <= pdr_bounds[1]
                ):
                    init_points.append(([fpt, pdr], snr))

            # Tell optimizer about these points (fit=False for speed)
            if init_points:
                xs = [[x] for x, _ in init_points]
                ys = [-y for _, y in init_points]
                opt.tell(xs, ys, fit=True)

        return opt

    def _select_top_flux_values(self) -> List[float]:
        """Select top 20% flux values by best SNR."""
        if not self.flux_best:
            return []

        # Sort flux values by their best SNR (descending)
        sorted_flx = sorted(
            self.flux_best.keys(), key=lambda f: self.flux_best[f][2], reverse=True
        )

        # Take top 20% (ceil)
        n_top = self._ceil(len(sorted_flx) * 0.2)
        n_top = max(1, n_top)  # At least 1
        return sorted_flx[:n_top]

    def _generate_refinement_points(
        self, selected_flx_list: List[float]
    ) -> List[float]:
        """
        Generate refinement test points for selected flux values.
        Each selected flux generates up to 3 points: -0.5, 0, +0.5 of prev_interval.
        Points outside bounds are ignored.
        """
        new_points: set[float] = set()
        offsets = [
            -0.5 * self.prev_interval,
            0,
            0.5 * self.prev_interval,
        ]

        for flx in selected_flx_list:
            for offset in offsets:
                new_flx = flx + offset
                # Check bounds
                if self.flx_bounds[0] <= new_flx <= self.flx_bounds[1]:
                    # Avoid duplicates and already measured flux values
                    if new_flx not in self.flux_best:
                        new_points.add(new_flx)

        return list(new_points)

    def _sort_refinement_points(self, points: List[float]) -> List[float]:
        """
        Sort refinement points so the first point is closest to last_flx.
        Choose ascending or descending order accordingly.
        """
        if not points:
            return points

        if self.last_flx is None:
            # Default to ascending
            return sorted(points)

        sorted_asc = sorted(points)
        sorted_desc = sorted(points, reverse=True)

        # Check which order puts the first point closer to last_flx
        dist_asc = abs(sorted_asc[0] - self.last_flx)
        dist_desc = abs(sorted_desc[0] - self.last_flx)

        return sorted_asc if dist_asc <= dist_desc else sorted_desc

    def _init_next_phase(self) -> bool:
        """
        Initialize the next phase (phase 2+).
        Returns True if a new phase was started, False if optimization should end.
        """
        # Calculate next phase budget: total_points * (1/2)^n
        next_phase = self._phase + 1
        next_budget = int(self.total_points * (1 / 2) ** next_phase)

        # Check termination condition
        if next_budget < 100:
            # Last phase: use all remaining budget
            next_budget = self.remaining_budget
            if next_budget <= 0:
                return False
            # Mark this as the final phase to enable optimizer
            self._is_final_phase = True

        self._phase = next_phase
        self.phase_budget = next_budget
        self.remaining_budget -= next_budget

        # Select top flux values
        selected_flx = self._select_top_flux_values()
        if not selected_flx:
            return False

        # Generate refinement points
        new_points = self._generate_refinement_points(selected_flx)
        if not new_points:
            # No new points to test, continue with direct optimization on selected
            new_points = selected_flx.copy()

        # Sort refinement points
        self._refinement_flx_list = self._sort_refinement_points(new_points)
        self._refinement_idx = 0

        # Budget per refinement flux value
        if self._refinement_flx_list:
            self._refinement_budget_per_flx = max(
                1, self.phase_budget // len(self._refinement_flx_list)
            )
        else:
            self._refinement_budget_per_flx = 0

        # Update interval for next phase
        self.prev_interval = self.prev_interval / 2  # 1/2 of previous interval

        # Initialize first refinement flux
        if self._refinement_flx_list:
            self._init_refinement_slice(self._refinement_flx_list[0])

        return True

    def _init_refinement_slice(self, flx: float) -> None:
        """
        Initialize 2D optimization for a refinement flux value.

        Non-final phases:
            - 100% LHS sampling (no optimizer)
        Final phase:
            - For new flux points: 50% LHS + 50% Bayesian optimization
            - For existing flux points: 100% Bayesian optimization
        """
        self._current_refinement_flx = flx
        self._refinement_slice_iter = 0

        # Check if this flux has been measured before
        is_new_flux = flx not in self.flux_data

        # Find nearest measured flux and its best point
        nearest_flx = self._find_nearest_measured_flx(flx)
        if nearest_flx is not None and nearest_flx in self.flux_best:
            best_fpt, best_pdr, _ = self.flux_best[nearest_flx]
        else:
            # Fallback to center of bounds
            best_fpt = (self.fpt_bounds[0] + self.fpt_bounds[1]) / 2
            best_pdr = (self.pdr_bounds[0] + self.pdr_bounds[1]) / 2

        # Get restricted bounds
        fpt_bounds, pdr_bounds = self._get_restricted_bounds(
            best_fpt, best_pdr, self._phase
        )

        # Store bounds for later use
        self._current_fpt_bounds = fpt_bounds
        self._current_pdr_bounds = pdr_bounds

        # Store neighbor's best point as initial guess for BO phase
        self._neighbor_guess = [best_fpt, best_pdr]
        self._neighbor_guess_used = False

        # Determine LHS budget and optimizer usage based on phase
        if not self._is_final_phase:
            # Non-final phase: 100% LHS (no optimizer)
            self._refinement_lhs_budget = self._refinement_budget_per_flx
            self._refinement_lhs_points = self._generate_lhs_samples(
                n_samples=self._refinement_lhs_budget,
                fpt_bounds=fpt_bounds,
                pdr_bounds=pdr_bounds,
            )
            self._refinement_lhs_idx = 0
            self.opt_2d = None
        else:
            # Final phase: use optimizer
            if is_new_flux:
                # New flux: 50% LHS + 50% BO
                self._refinement_lhs_budget = self._refinement_budget_per_flx // 2
                self._refinement_lhs_points = self._generate_lhs_samples(
                    n_samples=self._refinement_lhs_budget,
                    fpt_bounds=fpt_bounds,
                    pdr_bounds=pdr_bounds,
                )
                self._refinement_lhs_idx = 0
            else:
                # Existing flux: 100% BO (no LHS)
                self._refinement_lhs_budget = 0
                self._refinement_lhs_points = []
                self._refinement_lhs_idx = 0

            # Initialize optimizer with only data from THIS flux (avoid model pollution)
            self.opt_2d = self._init_2d_optimizer_for_flux(flx, fpt_bounds, pdr_bounds)

    def _get_phase1_point(self) -> Optional[Tuple[float, float, float]]:
        """
        Get next point for phase 1 optimization.
        Phase 1 uses pure LHS sampling (no Bayesian optimization).
        """
        current_flx = self.flx_grid[self.current_flx_idx]

        # Check if we've used up budget for this slice
        if self.current_slice_iter >= self.budget_per_flx:
            return None  # Signal to move to next slice

        # Check if we're still in LHS phase
        if self._lhs_idx < len(self._lhs_points):
            fpt, pdr = self._lhs_points[self._lhs_idx]
            self._lhs_idx += 1
            return (current_flx, fpt, pdr)

        # All LHS points exhausted for this slice
        return None

    def _get_phaseN_point(self) -> Optional[Tuple[float, float, float]]:
        """
        Get next point for phase 2+ optimization.

        Execution order:
        1. LHS points (if any, for new flux points)
        2. Neighbor's best point as initial guess (if valid and not duplicate)
        3. Bayesian optimization via opt_2d.ask()
        """
        if self._current_refinement_flx is None:
            return None

        current_flx = self._current_refinement_flx

        # Check if we've used up budget for this refinement slice
        if self._refinement_slice_iter >= self._refinement_budget_per_flx:
            return None  # Signal to move to next refinement flux

        # Step 1: Execute LHS points first (for new flux points)
        if self._refinement_lhs_idx < len(self._refinement_lhs_points):
            fpt, pdr = self._refinement_lhs_points[self._refinement_lhs_idx]
            self._refinement_lhs_idx += 1
            return (current_flx, fpt, pdr)

        # Step 2: Use neighbor's best point as first BO point (if not used yet)
        if not self._neighbor_guess_used and self._neighbor_guess is not None:
            self._neighbor_guess_used = True
            fpt, pdr = self._neighbor_guess

            # Check if point is within current bounds
            if (
                self._current_fpt_bounds[0] <= fpt <= self._current_fpt_bounds[1]
                and self._current_pdr_bounds[0] <= pdr <= self._current_pdr_bounds[1]
            ):
                # Check if this point was already sampled at this flux
                if current_flx in self.flux_data:
                    already_sampled = any(
                        abs(existing_fpt - fpt) < 1e-6
                        and abs(existing_pdr - pdr) < 1e-6
                        for existing_fpt, existing_pdr, _ in self.flux_data[current_flx]
                    )
                    if not already_sampled:
                        return (current_flx, fpt, pdr)
                else:
                    return (current_flx, fpt, pdr)

        # Step 3: Get next point from 2D optimizer
        if self.opt_2d is not None:
            next_2d = self.opt_2d.ask()
            return (current_flx, next_2d[0], next_2d[1])

        return None

    def _advance_to_next_slice(self) -> None:
        """Move to the next flux slice in phase 1."""
        self.current_flx_idx += 1
        self.current_slice_iter = 0

        if self.current_flx_idx < self.num_flx_points:
            self._generate_lhs_samples()

    def _advance_to_next_refinement(self) -> bool:
        """
        Move to the next refinement flux in phase 2+.
        Returns True if there are more refinement points, False otherwise.
        """
        self._refinement_idx += 1
        if self._refinement_idx < len(self._refinement_flx_list):
            self._init_refinement_slice(self._refinement_flx_list[self._refinement_idx])
            return True
        return False

    def _record_measurement(
        self, flx: float, fpt: float, pdr: float, snr: float
    ) -> None:
        """Record a measurement result."""
        # Add to flux_data
        if flx not in self.flux_data:
            self.flux_data[flx] = []
        self.flux_data[flx].append((fpt, pdr, snr))

        # Update flux_best
        if flx not in self.flux_best or snr > self.flux_best[flx][2]:
            self.flux_best[flx] = (fpt, pdr, snr)

        # Update global history
        self.history_X.append([flx, fpt, pdr])
        self.history_y.append(snr)

        # Update last_flx
        self.last_flx = flx

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
        if last_snr is not None and i > 0 and len(self.history_X) > 0:
            last_x = self.history_X[-1]
            flx, fpt, pdr = last_x[0], last_x[1], last_x[2]
            # Update flux_data and flux_best (history_X was already added, just update flux structures)
            if flx not in self.flux_data:
                self.flux_data[flx] = []
            # Check if this point was already added
            if not self.flux_data[flx] or self.flux_data[flx][-1] != (
                fpt,
                pdr,
                last_snr,
            ):
                self.flux_data[flx].append((fpt, pdr, last_snr))
            # Update flux_best
            if flx not in self.flux_best or last_snr > self.flux_best[flx][2]:
                self.flux_best[flx] = (fpt, pdr, last_snr)
            # Update history_y
            self.history_y.append(last_snr)
            # Update last_flx
            self.last_flx = flx

            # Tell optimizer about the result (only in Phase 2+)
            # Phase 1 uses pure LHS, no optimizer feedback needed
            if self._phase >= 2 and self.opt_2d is not None:
                # Check if point is within current bounds
                if hasattr(self, "_current_fpt_bounds") and hasattr(
                    self, "_current_pdr_bounds"
                ):
                    if (
                        self._current_fpt_bounds[0]
                        <= fpt
                        <= self._current_fpt_bounds[1]
                        and self._current_pdr_bounds[0]
                        <= pdr
                        <= self._current_pdr_bounds[1]
                    ):
                        self.opt_2d.tell([fpt, pdr], -last_snr)

        # Get next point based on current phase
        if self._phase == 1:
            point = self._get_phase1_point()

            if point is None:
                # Move to next slice
                self._advance_to_next_slice()

                if self.current_flx_idx >= self.num_flx_points:
                    # All flux slices done, move to phase 2
                    if not self._init_next_phase():
                        return None  # Optimization complete
                    return self.next_params(i, None)
                else:
                    # Try again with new slice
                    return self.next_params(i, None)

            # Record point (will be updated with SNR in next call)
            self.history_X.append(list(point))
            self.current_slice_iter += 1
            self._iter_count += 1
            return point

        else:  # phase >= 2
            point = self._get_phaseN_point()

            if point is None:
                # Try to advance to next refinement flux
                if self._advance_to_next_refinement():
                    return self.next_params(i, None)
                else:
                    # All refinement points done, try next phase
                    if not self._init_next_phase():
                        return None  # Optimization complete
                    return self.next_params(i, None)

            # Record point
            self.history_X.append(list(point))
            self._refinement_slice_iter += 1
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
            # Track phase for each point
            phases = np.zeros(num_points, dtype=np.int32)

            def plot_fn(ctx: TaskContext) -> None:
                idx: int = ctx.env_dict["index"]
                snrs = np.abs(ctx.data)  # (num_points, )

                cur_flx, cur_fpt, cur_pdr = params[idx, :]

                # Record current phase for this point
                phases[idx] = optimizer.phase

                # Assign colors based on phase using matplotlib color cycle
                prop_cycle = plt.rcParams["axes.prop_cycle"]
                cycle_colors = prop_cycle.by_key()["color"]
                colors = np.array(
                    [
                        cycle_colors[(p - 1) % len(cycle_colors)]
                        if p > 0
                        else "lightgray"
                        for p in phases
                    ]
                )

                fig.suptitle(
                    f"Iteration {idx}, Phase {phases[idx]}, Flux: {1e3 * cur_flx:.2g} (mA), Freq: {1e-3 * cur_fpt:.4g} (GHz), Power: {cur_pdr:.2g} (dBm)"
                )

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("flux_scatter").update(
                    params[:, 0], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 1], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("power_scatter").update(
                    params[:, 2], snrs, colors=colors, refresh=False
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

        max_id = np.nanargmax(snrs)
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

    from tqdm.auto import trange

    from zcu_tools.notebook.utils import make_sweep

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
        noise = 0.1 * np.random.normal(0, noise_std * snr_clean)
        snr = max(0.1, snr_clean + noise)

        return snr

    # Define sweep ranges
    flx_sweep: SweepCfg = make_sweep(0.0, 1.0, 150)
    fpt_sweep: SweepCfg = make_sweep(6500.0, 7500.0, 50)
    pdr_sweep: SweepCfg = make_sweep(-20.0, 0.0, 20)

    total_points = 10000

    print("=" * 60)
    print("JPAOptimizer Test (Multi-phase Algorithm)")
    print("=" * 60)
    print(f"Total points: {total_points}")
    print(f"Flux range: [{flx_sweep['start']}, {flx_sweep['stop']}]")
    print(f"Frequency range: [{fpt_sweep['start']}, {fpt_sweep['stop']}] MHz")
    print(f"Power range: [{pdr_sweep['start']}, {pdr_sweep['stop']}] dBm")
    print("True optimal: flx=0.5, fpt=7000 MHz, pdr=-10 dBm")
    print("=" * 60)

    # Create optimizer
    optimizer = JPAOptimizer(flx_sweep, fpt_sweep, pdr_sweep, total_points)

    print(f"Phase 1 budget: {optimizer.phase1_budget}")
    print(f"Number of flux slices: {optimizer.num_flx_points}")
    print(f"Budget per flux slice: {optimizer.budget_per_flx} (100% LHS)")
    print(f"Remaining budget for phase 2+: {optimizer.remaining_budget}")
    print("=" * 60)

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("JPAOptimizer Test Results (Multi-phase)", fontsize=14)

    # Color map for phases
    phase_colors_map = {1: "blue", 2: "red", 3: "green", 4: "orange", 5: "purple"}

    # Setup axes
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # Set static titles/labels for non-LivePlotter managed axes or initial setup
    ax6.set_xlabel("Flux Index")
    ax6.set_ylabel("Best SNR")
    ax6.set_title("Best SNR per Flux")
    ax6.grid(True, alpha=0.3, axis="y")

    # Run optimization
    params_list: List[Tuple[float, float, float]] = []
    snrs_list: List[float] = []
    phases_list: List[int] = []

    last_snr: Optional[float] = None

    plt.ion()
    plt.show()

    for i in trange(total_points, smoothing=0):
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

        if i % 500 == 0:  # Update every 10 iterations
            params_arr = np.array(params_list)
            snrs_arr = np.array(snrs_list)
            phases_arr = np.array(phases_list)

            colors = [phase_colors_map.get(p, "gray") for p in phases_arr]

            # Update scatter plots
            # 1. SNR vs Iteration
            ax1.clear()
            ax1.scatter(np.arange(len(snrs_arr)), snrs_arr, c=colors, s=5, alpha=0.6)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("SNR")
            ax1.set_title("SNR vs Iteration")
            ax1.grid(True, alpha=0.3)

            # 2. SNR vs Flux
            ax2.clear()
            ax2.scatter(params_arr[:, 0], snrs_arr, c=colors, s=5, alpha=0.6)
            ax2.set_xlabel("Flux (a.u.)")
            ax2.set_ylabel("SNR")
            ax2.set_title("SNR vs Flux")
            ax2.grid(True, alpha=0.3)

            # 3. SNR vs Frequency
            ax3.clear()
            ax3.scatter(params_arr[:, 1], snrs_arr, c=colors, s=5, alpha=0.6)
            ax3.set_xlabel("Frequency (MHz)")
            ax3.set_ylabel("SNR")
            ax3.set_title("SNR vs Frequency")
            ax3.grid(True, alpha=0.3)

            # 4. SNR vs Power
            ax4.clear()
            ax4.scatter(params_arr[:, 2], snrs_arr, c=colors, s=5, alpha=0.6)
            ax4.set_xlabel("Power (dBm)")
            ax4.set_ylabel("SNR")
            ax4.set_title("SNR vs Power")
            ax4.grid(True, alpha=0.3)

            # 5. Sampled Points (Flux vs Freq)
            ax5.clear()
            ax5.scatter(params_arr[:, 0], params_arr[:, 1], c=colors, s=10, alpha=0.5)
            ax5.set_xlabel("Flux (a.u.)")
            ax5.set_ylabel("Frequency (MHz)")
            ax5.set_title("Sampled Points (Flux vs Freq)")
            ax5.grid(True, alpha=0.3)

            # 6. Bar chart (Best SNR per Flux)
            ax6.clear()
            flux_values = sorted(optimizer.flux_best.keys())
            if flux_values:
                best_snrs_per_flux = [optimizer.flux_best[f][2] for f in flux_values]
                ax6.bar(
                    range(len(flux_values)),
                    best_snrs_per_flux,
                    width=0.8,
                    alpha=0.7,
                )
            ax6.set_xlabel("Flux Index")
            ax6.set_ylabel("Best SNR")
            ax6.set_title(f"Best SNR per Flux ({len(flux_values)} values)")
            ax6.grid(True, alpha=0.3, axis="y")

            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.ioff()  # Turn off interactive mode at the end

    # Convert to arrays
    params_arr = np.array(params_list)
    snrs_arr = np.array(snrs_list)
    phases_arr = np.array(phases_list)

    # Find best result
    best_idx = np.argmax(snrs_arr)
    best_params = params_arr[best_idx]
    best_snr = snrs_arr[best_idx]

    # Count points per phase
    unique_phases = np.unique(phases_arr)
    phase_counts = {p: np.sum(phases_arr == p) for p in unique_phases}

    print("=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Total iterations: {len(params_list)}")
    for p, count in sorted(phase_counts.items()):
        print(f"  Phase {p} points: {count}")
    print(f"Number of unique flux values tested: {len(optimizer.flux_best)}")
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

    plt.savefig("jpa_optimizer_test.png", dpi=150)
    print("Figure saved to jpa_optimizer_test.png")
    plt.show()
