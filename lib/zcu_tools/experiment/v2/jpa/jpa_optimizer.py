"""
JPA Optimizer - Multi-phase optimization for JPA parameters.

This module provides the JPAOptimizer class for optimizing JPA (Josephson Parametric
Amplifier) parameters using a multi-phase approach with Latin Hypercube Sampling
and Bayesian optimization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc
from skopt import Optimizer

from zcu_tools.program import SweepCfg

# Suppress skopt warnings about duplicate points
warnings.filterwarnings(
    "ignore",
    message="The objective has been evaluated at point .* before",
    category=UserWarning,
    module="skopt.optimizer.optimizer",
)

# Type aliases for clarity
Point2D = Tuple[float, float]  # (freq, power)
Point3D = Tuple[float, float, float]  # (flux, freq, power)
Bounds = Tuple[float, float]  # (min, max)


@dataclass
class ParameterBounds:
    """Bounds configuration for the 3D parameter space."""

    flux: Bounds
    freq: Bounds
    power: Bounds


@dataclass
class BudgetConfig:
    """Budget allocation for the optimizer."""

    total: int
    phase1: int
    remaining: int
    per_flux_slice: int


@dataclass
class Phase1State:
    """State tracking for Phase 1 (LHS exploration)."""

    flux_grid: np.ndarray
    flux_interval: float
    current_flux_idx: int = 0
    current_slice_iter: int = 0
    lhs_points: List[List[float]] = field(default_factory=list)
    lhs_idx: int = 0


@dataclass
class RefinementState:
    """State tracking for Phase 2+ (refinement phases)."""

    flux_list: List[float] = field(default_factory=list)
    flux_idx: int = 0
    current_flux: Optional[float] = None
    slice_iter: int = 0
    budget_per_flux: int = 0
    lhs_budget: int = 0
    lhs_points: List[List[float]] = field(default_factory=list)
    lhs_idx: int = 0
    neighbor_guess: Optional[List[float]] = None
    neighbor_guess_used: bool = False
    current_freq_bounds: Optional[Bounds] = None
    current_power_bounds: Optional[Bounds] = None


@dataclass
class DataStorage:
    """Storage for optimization history and results."""

    # Per-flux data: flux -> [(freq, power, snr), ...]
    flux_data: Dict[float, List[Tuple[float, float, float]]] = field(
        default_factory=dict
    )
    # Best result per flux: flux -> (best_freq, best_power, best_snr)
    flux_best: Dict[float, Tuple[float, float, float]] = field(default_factory=dict)
    # Global history for compatibility
    history_X: List[List[float]] = field(default_factory=list)  # [flux, freq, power]
    history_y: List[float] = field(default_factory=list)  # SNR values
    # Last measured flux value
    last_flux: Optional[float] = None


class JPAOptimizer:
    """
    Multi-phase optimizer for JPA parameters using scikit-optimize.

    Algorithm Overview:
        Phase 1 (50% of total budget):
            - Evenly distribute budget among all flux grid points
            - For each flux value: 100% LHS sampling (pure exploration)

        Phase 2+ (budget = total_points * (1/2)^n for phase n):
            - Select top 20% flux values by best SNR (ceil)
            - Generate 3 refinement points per selected flux at -0.5, 0, +0.5
              of previous interval
            - For new flux points: 50% LHS + 50% Bayesian optimization
            - For existing flux points: 100% Bayesian optimization
            - Use nearest neighbor's best point as initial guess for BO

        Final Phase (when next budget < 200):
            - Select the best flux value from all measured points
            - Use 100% Bayesian optimization (2D) with all historical data
              (filtered by bounds)
            - Terminates when next phase budget < 200, then use all remaining points
    """

    def __init__(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        total_points: int,
    ) -> None:
        # Initialize bounds
        self.bounds = ParameterBounds(
            flux=(flx_sweep["start"], flx_sweep["stop"]),
            freq=(fpt_sweep["start"], fpt_sweep["stop"]),
            power=(pdr_sweep["start"], pdr_sweep["stop"]),
        )

        # Calculate budget allocation
        phase1_budget = total_points // 2
        num_flux_points = self._calculate_flux_grid_size(
            flx_sweep, fpt_sweep, pdr_sweep, phase1_budget
        )
        budget_per_flux = phase1_budget // num_flux_points

        self.budget = BudgetConfig(
            total=total_points,
            phase1=phase1_budget,
            remaining=total_points - phase1_budget,
            per_flux_slice=budget_per_flux,
        )

        # Initialize Phase 1 state
        flux_grid = np.linspace(
            self.bounds.flux[0], self.bounds.flux[1], num_flux_points
        )
        flux_interval = (
            (self.bounds.flux[1] - self.bounds.flux[0]) / (num_flux_points - 1)
            if num_flux_points > 1
            else (self.bounds.flux[1] - self.bounds.flux[0])
        )
        self.phase1 = Phase1State(flux_grid=flux_grid, flux_interval=flux_interval)

        # Initialize refinement state (for Phase 2+)
        self.refinement = RefinementState()

        # Initialize data storage
        self.data = DataStorage()

        # Phase tracking
        self._phase = 1
        self._iter_count = 0
        self._is_final_phase = False

        # Phase 2+ interval tracking (shrinks each phase)
        self._prev_interval = flux_interval
        self._phase_budget = phase1_budget

        # LHS sampler for 2D (freq, power) space
        self._lhs_sampler = qmc.LatinHypercube(d=2)

        # 2D Bayesian optimizer (initialized when needed)
        self._optimizer_2d: Optional[Optimizer] = None

        # Generate initial LHS points for first flux slice
        self._generate_lhs_samples()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _calculate_flux_grid_size(
        self,
        flx_sweep: SweepCfg,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        phase1_budget: int,
    ) -> int:
        """Calculate the number of flux grid points based on sweep ratios."""
        n_flx = max(1, flx_sweep["expts"])
        n_fpt = max(1, fpt_sweep["expts"])
        n_pdr = max(1, pdr_sweep["expts"])

        # Scale factor to map total product of expts to phase1_budget
        # n_flx_eff * n_fpt_eff * n_pdr_eff ~= phase1_budget
        # where n_i_eff = k * n_i
        total_expts_prod = n_flx * n_fpt * n_pdr
        k = (phase1_budget / total_expts_prod) ** (1 / 3)

        return max(2, int(round(k * n_flx)))

    # =========================================================================
    # LHS Sampling
    # =========================================================================

    def _generate_lhs_samples(
        self,
        n_samples: Optional[int] = None,
        freq_bounds: Optional[Bounds] = None,
        power_bounds: Optional[Bounds] = None,
    ) -> List[List[float]]:
        """
        Generate 2D LHS samples for the (freq, power) space.

        Args:
            n_samples: Number of samples. If None, uses budget_per_flux_slice.
            freq_bounds: Frequency bounds. If None, uses default bounds.
            power_bounds: Power bounds. If None, uses default bounds.

        Returns:
            List of [freq, power] samples.
        """
        if n_samples is None:
            n_samples = max(1, self.budget.per_flux_slice)
        else:
            n_samples = max(1, n_samples)

        if freq_bounds is None:
            freq_bounds = self.bounds.freq
        if power_bounds is None:
            power_bounds = self.bounds.power

        # Generate samples in [0, 1]^2 and scale to actual bounds
        samples = self._lhs_sampler.random(n=n_samples)
        freq_range = freq_bounds[1] - freq_bounds[0]
        power_range = power_bounds[1] - power_bounds[0]

        lhs_points: List[List[float]] = []
        for s in samples:
            freq = freq_bounds[0] + s[0] * freq_range
            power = power_bounds[0] + s[1] * power_range
            lhs_points.append([freq, power])

        # Update Phase 1 state for compatibility
        self.phase1.lhs_points = lhs_points
        self.phase1.lhs_idx = 0

        return lhs_points

    # =========================================================================
    # Flux Value Helpers
    # =========================================================================

    def _find_nearest_measured_flux(self, target_flux: float) -> Optional[float]:
        """Find the nearest flux value that has been measured."""
        if not self.data.flux_best:
            return None
        measured_flux_list = list(self.data.flux_best.keys())
        return min(measured_flux_list, key=lambda x: abs(x - target_flux))

    def _get_best_flux(self) -> Optional[float]:
        """Get the flux value with the highest SNR from all measured points."""
        if not self.data.flux_best:
            return None

        # Filter flux values within bounds
        valid_flux = [
            f
            for f in self.data.flux_best.keys()
            if self.bounds.flux[0] <= f <= self.bounds.flux[1]
        ]
        if not valid_flux:
            return None

        return max(valid_flux, key=lambda f: self.data.flux_best[f][2])

    def _select_top_flux_values(self) -> List[float]:
        """Select top 20% flux values by best SNR."""
        if not self.data.flux_best:
            return []

        # Sort flux values by their best SNR (descending)
        sorted_flux = sorted(
            self.data.flux_best.keys(),
            key=lambda f: self.data.flux_best[f][2],
            reverse=True,
        )

        # Take top 20% (ceil), at least 1
        n_top = max(1, ceil(len(sorted_flux) * 0.2))
        return sorted_flux[:n_top]

    # =========================================================================
    # Bounds Calculation
    # =========================================================================

    def _get_restricted_bounds(
        self, center_freq: float, center_power: float, phase: int
    ) -> Tuple[Bounds, Bounds]:
        """
        Get restricted 2D bounds centered on (center_freq, center_power).

        Range shrinks by factor of 2^(phase-1).
        Returns intersection with original bounds.
        """
        shrink_factor = 2 ** (phase - 1)
        freq_range = (self.bounds.freq[1] - self.bounds.freq[0]) / shrink_factor
        power_range = (self.bounds.power[1] - self.bounds.power[0]) / shrink_factor

        freq_lo = max(self.bounds.freq[0], center_freq - freq_range / 2)
        freq_hi = min(self.bounds.freq[1], center_freq + freq_range / 2)
        power_lo = max(self.bounds.power[0], center_power - power_range / 2)
        power_hi = min(self.bounds.power[1], center_power + power_range / 2)

        return ((freq_lo, freq_hi), (power_lo, power_hi))

    # =========================================================================
    # Bayesian Optimizer Initialization
    # =========================================================================

    def _init_2d_optimizer_for_flux(
        self,
        target_flux: float,
        freq_bounds: Bounds,
        power_bounds: Bounds,
    ) -> Optimizer:
        """
        Initialize 2D optimizer using ONLY data measured at the target flux.

        This avoids model pollution from neighbor flux values where the optimal
        (freq, power) may be very different.
        """
        opt = Optimizer(
            dimensions=[freq_bounds, power_bounds],
            base_estimator="ET",
            acq_func="EI",
            n_initial_points=0,
            n_jobs=-1,
            acq_optimizer="auto",
        )

        # Only use data from THIS flux value (avoid model pollution)
        if target_flux in self.data.flux_data:
            init_points: List[Tuple[List[float], float]] = []
            for freq, power, snr in self.data.flux_data[target_flux]:
                # Check if point is within the current bounds
                if (
                    freq_bounds[0] <= freq <= freq_bounds[1]
                    and power_bounds[0] <= power <= power_bounds[1]
                ):
                    init_points.append(([freq, power], snr))

            # Tell optimizer about these points
            xs = [x for x, _ in init_points]
            ys = [-y for _, y in init_points]  # Minimize negative SNR
            opt.tell(xs, ys, fit=True)

        return opt

    # =========================================================================
    # Refinement Point Generation
    # =========================================================================

    def _generate_refinement_points(
        self, selected_flux_list: List[float]
    ) -> List[float]:
        """
        Generate refinement test points for selected flux values.

        Each selected flux generates up to 3 points at offsets of
        -0.5, 0, +0.5 of prev_interval from the original point.
        Points outside bounds are ignored.

        Note: We include ALL points (even previously measured ones) because
        in each refinement phase, we want to re-measure with finer (freq, power)
        grids centered around the best known values.
        """
        new_points: set[float] = set()
        offsets = [
            -0.5 * self._prev_interval,
            0,
            0.5 * self._prev_interval,
        ]

        for flux in selected_flux_list:
            for offset in offsets:
                new_flux = flux + offset
                # Check bounds only - include previously measured flux values
                # because we want to refine with smaller (freq, power) bounds
                if self.bounds.flux[0] <= new_flux <= self.bounds.flux[1]:
                    new_points.add(new_flux)

        return list(new_points)

    def _sort_refinement_points(self, points: List[float]) -> List[float]:
        """
        Sort refinement points so the first point is closest to last_flux.

        Choose ascending or descending order accordingly.
        """
        if not points:
            return points

        if self.data.last_flux is None:
            return sorted(points)

        sorted_asc = sorted(points)
        sorted_desc = sorted(points, reverse=True)

        # Check which order puts the first point closer to last_flux
        dist_asc = abs(sorted_asc[0] - self.data.last_flux)
        dist_desc = abs(sorted_desc[0] - self.data.last_flux)

        return sorted_asc if dist_asc <= dist_desc else sorted_desc

    # =========================================================================
    # Phase Transitions
    # =========================================================================

    def _init_next_phase(self) -> bool:
        """
        Initialize the next phase (phase 2+).

        Returns:
            True if a new phase was started, False if optimization should end.
        """
        next_phase = self._phase + 1
        next_budget = int(self.budget.total * (1 / 2) ** next_phase)

        # Check termination condition: next_budget < 200 triggers final phase
        if next_budget < 200:
            return self._init_final_phase(next_phase)

        # Non-final phase: proceed with normal refinement logic
        return self._init_refinement_phase(next_phase, next_budget)

    def _init_final_phase(self, next_phase: int) -> bool:
        """Initialize the final optimization phase."""
        next_budget = self.budget.remaining
        if next_budget <= 0:
            return False

        self._is_final_phase = True
        self._phase = next_phase
        self._phase_budget = next_budget
        self.budget.remaining -= next_budget

        # Select the single best flux from all measured points
        best_flux = self._get_best_flux()
        if best_flux is None:
            return False

        # Use only the best flux (no new points generated)
        self.refinement.flux_list = [best_flux]
        self.refinement.flux_idx = 0
        self.refinement.budget_per_flux = next_budget

        # Update interval for next phase
        self._prev_interval = self._prev_interval / 2

        # Initialize refinement slice for the best flux
        self._init_refinement_slice(best_flux)
        return True

    def _init_refinement_phase(self, next_phase: int, next_budget: int) -> bool:
        """Initialize a non-final refinement phase."""
        self._phase = next_phase
        self._phase_budget = next_budget
        self.budget.remaining -= next_budget

        # Select top flux values
        selected_flux = self._select_top_flux_values()
        if not selected_flux:
            return False

        # Generate refinement points (includes previously measured flux values
        # for refinement with finer freq/power grids)
        new_points = self._generate_refinement_points(selected_flux)
        if not new_points:
            return False

        # Sort refinement points
        self.refinement.flux_list = self._sort_refinement_points(new_points)
        self.refinement.flux_idx = 0

        # Budget per refinement flux value
        if self.refinement.flux_list:
            self.refinement.budget_per_flux = max(
                1, self._phase_budget // len(self.refinement.flux_list)
            )
        else:
            self.refinement.budget_per_flux = 0

        # Update interval for next phase
        self._prev_interval = self._prev_interval / 2

        # Initialize first refinement flux
        if self.refinement.flux_list:
            self._init_refinement_slice(self.refinement.flux_list[0])

        return True

    def _init_refinement_slice(self, flux: float) -> None:
        """
        Initialize 2D optimization for a refinement flux value.

        Non-final phases: 100% LHS sampling (no optimizer)
        Final phase: Uses 100% Bayesian optimization (2D) on the best flux
        """
        self.refinement.current_flux = flux
        self.refinement.slice_iter = 0

        # Check if this flux has been measured before
        is_new_flux = flux not in self.data.flux_data

        # Find nearest measured flux and its best point
        nearest_flux = self._find_nearest_measured_flux(flux)
        if nearest_flux is not None and nearest_flux in self.data.flux_best:
            best_freq, best_power, _ = self.data.flux_best[nearest_flux]
        else:
            # Fallback to center of bounds
            best_freq = (self.bounds.freq[0] + self.bounds.freq[1]) / 2
            best_power = (self.bounds.power[0] + self.bounds.power[1]) / 2

        # Get restricted bounds
        freq_bounds, power_bounds = self._get_restricted_bounds(
            best_freq, best_power, self._phase
        )

        # Store bounds in refinement state
        self.refinement.current_freq_bounds = freq_bounds
        self.refinement.current_power_bounds = power_bounds

        # Store neighbor's best point as initial guess for BO phase
        self.refinement.neighbor_guess = [best_freq, best_power]
        self.refinement.neighbor_guess_used = False

        # Determine LHS budget and optimizer usage based on phase
        if not self._is_final_phase:
            # Non-final phase: 100% LHS (no optimizer)
            self.refinement.lhs_budget = self.refinement.budget_per_flux
            self.refinement.lhs_points = self._generate_lhs_samples(
                n_samples=self.refinement.lhs_budget,
                freq_bounds=freq_bounds,
                power_bounds=power_bounds,
            )
            self.refinement.lhs_idx = 0
            self._optimizer_2d = None
        else:
            # Final phase: use optimizer
            if is_new_flux:
                # New flux: 50% LHS + 50% BO
                self.refinement.lhs_budget = self.refinement.budget_per_flux // 2
                self.refinement.lhs_points = self._generate_lhs_samples(
                    n_samples=self.refinement.lhs_budget,
                    freq_bounds=freq_bounds,
                    power_bounds=power_bounds,
                )
                self.refinement.lhs_idx = 0
            else:
                # Existing flux: 100% BO (no LHS)
                self.refinement.lhs_budget = 0
                self.refinement.lhs_points = []
                self.refinement.lhs_idx = 0

            # Initialize optimizer with only data from THIS flux
            self._optimizer_2d = self._init_2d_optimizer_for_flux(
                flux, freq_bounds, power_bounds
            )

    # =========================================================================
    # Point Generation
    # =========================================================================

    def _get_phase1_point(self) -> Optional[Point3D]:
        """
        Get next point for phase 1 optimization.

        Phase 1 uses pure LHS sampling (no Bayesian optimization).
        """
        current_flux = self.phase1.flux_grid[self.phase1.current_flux_idx]

        # Check if we've used up budget for this slice
        if self.phase1.current_slice_iter >= self.budget.per_flux_slice:
            return None  # Signal to move to next slice

        # Check if we're still in LHS phase
        if self.phase1.lhs_idx < len(self.phase1.lhs_points):
            freq, power = self.phase1.lhs_points[self.phase1.lhs_idx]
            self.phase1.lhs_idx += 1
            return (current_flux, freq, power)

        # All LHS points exhausted for this slice
        return None

    def _get_phaseN_point(self) -> Optional[Point3D]:
        """
        Get next point for phase 2+ optimization.

        Execution order:
            1. LHS points (if any, for new flux points)
            2. Neighbor's best point as initial guess (if valid and not duplicate)
            3. Bayesian optimization via optimizer.ask()
        """
        if self.refinement.current_flux is None:
            return None

        current_flux = self.refinement.current_flux

        # Check if we've used up budget for this refinement slice
        if self.refinement.slice_iter >= self.refinement.budget_per_flux:
            return None  # Signal to move to next refinement flux

        # Step 1: Execute LHS points first (for new flux points)
        if self.refinement.lhs_idx < len(self.refinement.lhs_points):
            freq, power = self.refinement.lhs_points[self.refinement.lhs_idx]
            self.refinement.lhs_idx += 1
            return (current_flux, freq, power)

        # Step 2: Use neighbor's best point as first BO point (if not used yet)
        if (
            not self.refinement.neighbor_guess_used
            and self.refinement.neighbor_guess is not None
        ):
            self.refinement.neighbor_guess_used = True
            freq, power = self.refinement.neighbor_guess

            # Check if point is within current bounds
            freq_bounds = self.refinement.current_freq_bounds
            power_bounds = self.refinement.current_power_bounds
            if freq_bounds and power_bounds:
                if (
                    freq_bounds[0] <= freq <= freq_bounds[1]
                    and power_bounds[0] <= power <= power_bounds[1]
                ):
                    # Check if this point was already sampled at this flux
                    if current_flux in self.data.flux_data:
                        already_sampled = any(
                            abs(existing_freq - freq) < 1e-6
                            and abs(existing_power - power) < 1e-6
                            for existing_freq, existing_power, _ in self.data.flux_data[
                                current_flux
                            ]
                        )
                        if not already_sampled:
                            return (current_flux, freq, power)
                    else:
                        return (current_flux, freq, power)

        # Step 3: Get next point from 2D optimizer
        if self._optimizer_2d is not None:
            next_2d = self._optimizer_2d.ask()
            assert next_2d is not None
            return (current_flux, next_2d[0], next_2d[1])

        return None

    # =========================================================================
    # Slice/Phase Advancement
    # =========================================================================

    def _advance_to_next_slice(self) -> None:
        """Move to the next flux slice in phase 1."""
        self.phase1.current_flux_idx += 1
        self.phase1.current_slice_iter = 0

        if self.phase1.current_flux_idx < len(self.phase1.flux_grid):
            self._generate_lhs_samples()

    def _advance_to_next_refinement(self) -> bool:
        """
        Move to the next refinement flux in phase 2+.

        Returns:
            True if there are more refinement points, False otherwise.
        """
        self.refinement.flux_idx += 1
        if self.refinement.flux_idx < len(self.refinement.flux_list):
            self._init_refinement_slice(
                self.refinement.flux_list[self.refinement.flux_idx]
            )
            return True
        return False

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def phase(self) -> int:
        """Current optimization phase."""
        return self._phase

    @phase.setter
    def phase(self, value: int) -> None:
        self._phase = value

    # Compatibility properties for external access
    @property
    def phase1_budget(self) -> int:
        return self.budget.phase1

    @property
    def remaining_budget(self) -> int:
        return self.budget.remaining

    @property
    def num_flx_points(self) -> int:
        return len(self.phase1.flux_grid)

    @property
    def budget_per_flx(self) -> int:
        return self.budget.per_flux_slice

    @property
    def flux_best(self) -> Dict[float, Tuple[float, float, float]]:
        return self.data.flux_best

    @property
    def history_X(self) -> List[List[float]]:
        return self.data.history_X

    @property
    def history_y(self) -> List[float]:
        return self.data.history_y

    def next_params(self, i: int, last_snr: Optional[float]) -> Optional[Point3D]:
        """
        Get the next parameter set to evaluate.

        Args:
            i: Current iteration index.
            last_snr: SNR value from the previous iteration (None for first iteration).

        Returns:
            Tuple of (flux, freq, power) or None if optimization is complete.
        """
        if i >= self.budget.total:
            return None

        # Record last result
        if last_snr is not None and i > 0 and len(self.data.history_X) > 0:
            self._record_last_result(last_snr)

        # Get next point based on current phase
        if self._phase == 1:
            return self._get_next_phase1_point(i)
        else:
            return self._get_next_phaseN_point(i)

    def _record_last_result(self, last_snr: float) -> None:
        """Record the result from the last measurement."""
        last_x = self.data.history_X[-1]
        flux, freq, power = last_x[0], last_x[1], last_x[2]

        # Update flux_data
        if flux not in self.data.flux_data:
            self.data.flux_data[flux] = []
        # Check if this point was already added
        if not self.data.flux_data[flux] or self.data.flux_data[flux][-1] != (
            freq,
            power,
            last_snr,
        ):
            self.data.flux_data[flux].append((freq, power, last_snr))

        # Update flux_best
        if flux not in self.data.flux_best or last_snr > self.data.flux_best[flux][2]:
            self.data.flux_best[flux] = (freq, power, last_snr)

        # Update history_y
        self.data.history_y.append(last_snr)

        # Update last_flux
        self.data.last_flux = flux

        # Tell optimizer about the result (only in Phase 2+)
        if self._phase >= 2 and self._optimizer_2d is not None:
            freq_bounds = self.refinement.current_freq_bounds
            power_bounds = self.refinement.current_power_bounds
            if freq_bounds and power_bounds:
                if (
                    freq_bounds[0] <= freq <= freq_bounds[1]
                    and power_bounds[0] <= power <= power_bounds[1]
                ):
                    self._optimizer_2d.tell([freq, power], -last_snr)

    def _get_next_phase1_point(self, i: int) -> Optional[Point3D]:
        """Get the next point during Phase 1."""
        point = self._get_phase1_point()

        if point is None:
            # Move to next slice
            self._advance_to_next_slice()

            if self.phase1.current_flux_idx >= len(self.phase1.flux_grid):
                # All flux slices done, move to phase 2
                if not self._init_next_phase():
                    return None  # Optimization complete
                # Don't use recursion with None - directly get next point
                return self._get_next_phaseN_point(i)
            else:
                # Try again with new slice - get point directly without recursion
                point = self._get_phase1_point()
                if point is None:
                    return None  # Should not happen, but handle gracefully

        # Record point (will be updated with SNR in next call)
        self.data.history_X.append(list(point))
        self.phase1.current_slice_iter += 1
        self._iter_count += 1
        return point

    def _get_next_phaseN_point(self, i: int) -> Optional[Point3D]:
        """Get the next point during Phase 2+."""
        point = self._get_phaseN_point()

        if point is None:
            # Try to advance to next refinement flux
            if self._advance_to_next_refinement():
                # Get point directly without recursion
                point = self._get_phaseN_point()
                if point is None:
                    return None  # Should not happen, but handle gracefully
            else:
                # All refinement points done, try next phase
                if not self._init_next_phase():
                    return None  # Optimization complete
                # Get point directly without recursion
                point = self._get_phaseN_point()
                if point is None:
                    return None  # Should not happen, but handle gracefully

        # Record point
        self.data.history_X.append(list(point))
        self.refinement.slice_iter += 1
        self._iter_count += 1
        return point


if __name__ == "__main__":
    # Test JPAOptimizer with a simulated SNR function

    import time
    from typing import List as ListType
    from typing import Optional as OptionalType
    from typing import Tuple as TupleType

    import matplotlib.pyplot as plt
    from tqdm.auto import trange

    from zcu_tools.notebook.utils import make_sweep

    def simulate_snr(
        flux: float, freq: float, power: float, noise_std: float = 0.1
    ) -> float:
        """
        Simulate a more complex SNR landscape as a combination of multiple Gaussians,
        higher-frequency sinusoidal, and polynomial modulations, plus noise.

        - Multiple Gaussian peaks at different locations and widths
        - Sinusoidal modulation in all three axes with distinct frequencies/phases
        - Cross-terms between variables for richer features
        - Measurement noise
        """
        # Main Gaussian peak
        flux_0, freq_0, power_0 = 0.5, 7000.0, -10.0
        sigma_flux, sigma_freq, sigma_power = 0.15, 300.0, 3.0
        gauss_main = np.exp(
            -((flux - flux_0) ** 2) / (2 * sigma_flux**2)
            - ((freq - freq_0) ** 2) / (2 * sigma_freq**2)
            - ((power - power_0) ** 2) / (2 * sigma_power**2)
        )
        # Secondary, offset Gaussian peak
        gauss_side = 0.5 * np.exp(
            -((flux - 0.3) ** 2) / (2 * 0.08**2)
            - ((freq - 7200.0) ** 2) / (2 * 120.0**2)
            - ((power + 5.0) ** 2) / (2 * 1.2**2)
        )
        # Tertiary, broad Gaussian hill
        gauss_broad = 0.3 * np.exp(
            -((flux - 0.8) ** 2) / (2 * 0.25**2)
            - ((freq - 6900.0) ** 2) / (2 * 350.0**2)
            - ((power + 15.0) ** 2) / (2 * 4.0**2)
        )

        # Sinusoidal modulations on each axis
        sin_flux = 0.4 * np.sin(2 * np.pi * (flux - 0.25) / 0.18 + 0.5)
        sin_freq = 0.18 * np.sin(2 * np.pi * (freq - 6700) / 80 + np.pi / 4)
        sin_power = 0.25 * np.sin(2 * np.pi * (power + 8) / 2.8 + 1)

        # High-frequency ripples (simulate interference/complexities)
        hf_ripple = 0.12 * np.sin(12 * np.pi * flux + 8 * np.cos(freq / 570))
        hf_ripple += 0.09 * np.cos(8 * np.pi * power + flux * 7)

        # Cross-terms for correlation between axes
        cross_term = 0.15 * np.sin(3 * np.pi * flux * freq / 7800)
        cross_term += 0.11 * np.cos((freq - 7000) * (power + 10) / 600)

        # Polynomial baseline
        poly = 1.2 - 0.8 * (flux - 0.6) * (freq / 8000) * (power / 15)

        # Combine all features
        snr_clean = (
            10.0 * gauss_main * (1 + sin_flux + sin_freq + sin_power + cross_term)
            + 2.5 * gauss_side
            + 1.2 * gauss_broad
            + hf_ripple
            + poly
        )

        # Add measurement noise
        noise = 0.1 * np.random.normal(0, noise_std * abs(snr_clean))
        snr = max(0.1, snr_clean + noise)

        return snr

    # Define sweep ranges
    flux_sweep: SweepCfg = make_sweep(0.0, 1.0, 150)
    freq_sweep: SweepCfg = make_sweep(6500.0, 7500.0, 50)
    power_sweep: SweepCfg = make_sweep(-20.0, 0.0, 20)

    total_points = 1000

    print("=" * 60)
    print("JPAOptimizer Test (Multi-phase Algorithm)")
    print("=" * 60)
    print(f"Total points: {total_points}")
    print(f"Flux range: [{flux_sweep['start']}, {flux_sweep['stop']}]")
    print(f"Frequency range: [{freq_sweep['start']}, {freq_sweep['stop']}] MHz")
    print(f"Power range: [{power_sweep['start']}, {power_sweep['stop']}] dBm")
    print("True optimal: flux=0.5, freq=7000 MHz, power=-10 dBm")
    print("=" * 60)

    # Create optimizer
    optimizer = JPAOptimizer(flux_sweep, freq_sweep, power_sweep, total_points)

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
    params_list: ListType[TupleType[float, float, float]] = []
    snrs_list: ListType[float] = []
    phases_list: ListType[int] = []

    last_snr: OptionalType[float] = None

    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)  # Give time for window to appear

    last_plot_time = time.time() - 2.0
    for i in trange(total_points, smoothing=0):
        params = optimizer.next_params(i, last_snr)
        if params is None:
            print(f"Optimization stopped at iteration {i}")
            break

        flux, freq, power = params
        snr = simulate_snr(flux, freq, power, noise_std=0.05)

        params_list.append(params)
        snrs_list.append(snr)
        phases_list.append(optimizer.phase)

        last_snr = snr

        # Update plot every 2 seconds
        if time.time() - last_plot_time > 1.0:
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

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)  # Force GUI event processing

            last_plot_time = time.time()

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
        f"Best params: flux={best_params[0]:.4f}, "
        f"freq={best_params[1]:.1f} MHz, power={best_params[2]:.2f} dBm"
    )
    print("True optimal: flux=0.5, freq=7000 MHz, power=-10 dBm")
    print(
        f"Error: flux={abs(best_params[0] - 0.5):.4f}, "
        f"freq={abs(best_params[1] - 7000):.1f} MHz, power={abs(best_params[2] + 10):.2f} dBm"
    )
    print("=" * 60)

    plt.savefig("jpa_optimizer_test.png", dpi=150)
    print("Figure saved to jpa_optimizer_test.png")
    plt.show()
