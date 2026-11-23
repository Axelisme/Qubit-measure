"""Shared RWA admission, retention and model/WorkPlan coefficient adapters."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ..secular import SecularModel
from ..workplan import FactKey, WorkPlan
from . import _rwa_fourier
from ._dop853 import tolerances
from ._dop853_data import pack_models
from ._workload_facts import envelope_bounds, fixed_flux, packed_model, quadrature_norms

_REVISION = "joint-rwa-v3"
STEP_NS = 4.0
CONSTANT_STEP_NS = 16.0
MIN_RWA_BUDGET = 1e-3
_MAX_RATIO = 0.25
_MAX_SLOPE = 0.025
_MAX_CURVATURE = 0.003
# Integer 256 ns gain sweep exceeded 1e-2 at gain 0.4184, proxy 0.063 rad.
_MAX_PHASE_RAD = 0.06
# Distinct-port dual constants: strict-lab calibration, half the 0.338 rad boundary.
_MAX_CONSTANT_PHASE_RAD = 0.16
CONSTANT_RWA_POLICY = (
    "constant-omission-v1",
    _rwa_fourier.MAX_OMISSION_TRANSFER,
    _MAX_CONSTANT_PHASE_RAD,
)
RWA_POLICY = (
    _REVISION,
    _rwa_fourier.EDGE_GHZ,
    STEP_NS,
    CONSTANT_STEP_NS,
    _rwa_fourier.MAX_OMISSION_TRANSFER,
    MIN_RWA_BUDGET,
    _MAX_RATIO,
    _MAX_SLOPE,
    _MAX_CURVATURE,
    _MAX_PHASE_RAD,
)


COMPLEX_ENVELOPE_RWA_POLICY = ("complex-envelope-modulus-v1", RWA_POLICY)
ADAPTIVE_RWA_POLICY = ("omission-strongest-parent-v2", CONSTANT_RWA_POLICY, RWA_POLICY)
# Full-density calibration supports 0.08 with margin on the fixed candidate
# acceptance set, not a maximum threshold or arbitrary-input accuracy bound.
_MAX_KAPPA2 = 0.08
SINGLE_CONSTANT_RWA_POLICY = (
    "single-constant-kappa2-v1",
    ADAPTIVE_RWA_POLICY,
    _MAX_KAPPA2,
)
_MAX_OMISSION_PHASE_RAD = 0.0045
_MAX_RETAINED_DIMENSION = 256
MULTITONE_RETENTION_POLICY = (
    "multitone-omission-retention-v1",
    ADAPTIVE_RWA_POLICY,
    _MAX_OMISSION_PHASE_RAD,
    _MAX_RETAINED_DIMENSION,
)
# The endpoint kick removes first-order omitted-edge error, not its secular
# interaction with retained couplings. The calibrated constant-pair cap is
# epsilon_dom**2 * Omega_ret * T <=1.476e-4 cycle; retention has its own policy.
_MAX_MULTITONE_SECULAR_CYCLES = 1.476e-4
MULTITONE_GUARD_POLICY = (
    "constant-pair-secular-v1",
    MULTITONE_RETENTION_POLICY,
    MIN_RWA_BUDGET,
    _rwa_fourier.EDGE_GHZ,
    _MAX_MULTITONE_SECULAR_CYCLES,
)


@dataclass(frozen=True)
class RWARetention:
    """Resolved additions to the initial window and remaining omission estimate.

    edges has one ordered tuple of added (i,j) coordinates per positive carrier;
    the final set is the 50 MHz window union these edges. Negative Fourier masks
    are the transposes. omission_transfer and omission_phase are maximum
    remaining first/second-order transfer and phase proxies, not rigorous bounds.
    dimension is sum(component_size**2), the real population/coherence dimension
    of the full Fourier union graph, including isolated populations. Phase and
    dimension are None for single constant rows, whose phase-graph owner keeps
    its original phase/pruned-dimension policy. omission_computed is true only
    when omission analysis completed for the final mask. A dimension stop can
    leave infinite or prior-mask proxies, which must not be reported as final.
    """

    edges: tuple[tuple[tuple[int, int], ...], ...]
    omission_transfer: float
    omission_phase: float | None = None
    dimension: int | None = None
    omission_computed: bool = False


@dataclass(frozen=True)
class RWAGuardMetric:
    """One immutable numeric admission comparison, not a density-error bound.

    value is the measured proxy; threshold uses the same units, documented on
    the containing diagnostic field. comparison is the inclusive admission
    direction, either <= or >=. passed evaluates that comparison.
    """

    value: float
    threshold: float
    comparison: Literal["<=", ">="] = "<="

    @property
    def passed(self) -> bool:
        """Whether value satisfies the inclusive threshold comparison."""
        return (
            self.value <= self.threshold
            if self.comparison == "<="
            else self.value >= self.threshold
        )


@dataclass(frozen=True)
class RWAUncomputed:
    """An admission metric whose numeric value was not computed.

    This frozen marker has neither a value nor a passed status. Earlier guard
    rejection skipped the numeric work; it is distinct from a failed metric.
    """


@dataclass(frozen=True)
class ConstantRWAAdmission:
    """Shared constant-row retention and numeric admission comparisons.

    retention owns the final Fourier edge set. carrier_ghz is the absolute
    carrier in GHz, required >=0.05. kappa2 is the dimensionless spectral norm
    of the entrywise absolute omitted c/d coefficients summed over both signs,
    after edge restoration. Its 0.08 cap has full-density calibration with
    margin on the fixed candidate acceptance set. This does not establish a
    maximum usable threshold or arbitrary-input accuracy.
    strength_ratio is the full-quadrature strength divided by absolute carrier.
    phase_rad is the retained-norm phase proxy in radians, or RWAUncomputed
    when the strength-ratio comparison already rejected the row.
    """

    retention: RWARetention
    carrier_ghz: RWAGuardMetric
    kappa2: RWAGuardMetric
    strength_ratio: RWAGuardMetric
    phase_rad: RWAGuardMetric | RWAUncomputed

    @property
    def accepted(self) -> bool:
        """Whether the carrier and shared numeric comparisons pass."""
        return (
            self.carrier_ghz.passed
            and self.strength_ratio.passed
            and isinstance(self.phase_rad, RWAGuardMetric)
            and self.phase_rad.passed
            and self.kappa2.passed
        )


@dataclass(frozen=True)
class MultitoneRWAAdmission:
    """Constant-pair domain, final retention and secular admission fact.

    budget compares the dimensionless allowance to the shared minimum.
    carrier_ghz compares the smaller absolute carrier to 0.05 GHz, or is
    uncomputed after budget or same-port rejection. retention is None when
    domain/carrier rejection skips resolution; otherwise it is the final
    retained set, including dimension rejection. omission_transfer is the final
    mask's dimensionless transfer proxy, required <=MAX_OMISSION_TRANSFER;
    omission_phase_rad is its phase in rad, required <=_MAX_OMISSION_PHASE_RAD.
    Both are uncomputed when the final mask's omission analysis was skipped.
    real_dimension compares the final graph's coordinate count to
    _MAX_RETAINED_DIMENSION, even when that limit rejects the row.
    secular_accumulation is
    epsilon_dom**2 * Omega_ret * duration in cycles, or uncomputed when no
    dimension-legal final mask was resolved. accepted includes every domain
    and numeric comparison, not just the secular metric.
    """

    budget: RWAGuardMetric
    carrier_ghz: RWAGuardMetric | RWAUncomputed
    retention: RWARetention | None
    secular_accumulation: RWAGuardMetric | RWAUncomputed
    accepted: bool
    omission_transfer: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    omission_phase_rad: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
    real_dimension: RWAGuardMetric | RWAUncomputed = RWAUncomputed()


@dataclass(frozen=True)
class ConstantRetentions:
    """Single-constant retained sets aligned with absolute SegmentTable rows.

    rows contains None outside the fixed-flux single constant carrier domain.
    The complete immutable inventory avoids repeated per-key fact lookups.
    """

    rows: tuple[RWARetention | None, ...]


def _smooth(
    table: SegmentTable, envelope: int, context: KernelContext, *, allow_complex: bool
) -> bool:
    spec = table.envelopes[envelope]
    bounds = envelope_bounds(table, context.work_plan)
    real = bounds.real[envelope]
    policy = RWA_POLICY if real or not allow_complex else COMPLEX_ENVELOPE_RWA_POLICY

    def build() -> bool:
        if spec.kind == "constant":
            return True
        if len(spec.samples) < 3 or (not real and not allow_complex):
            return False
        peak = float(bounds.peaks[envelope])
        if peak == 0:
            return True
        samples = spec.samples.real if real else spec.samples
        spacing = np.diff(spec.sample_ns)
        slopes = np.diff(samples) / spacing
        curvature = np.diff(slopes) / ((spacing[1:] + spacing[:-1]) / 2)
        return bool(
            np.max(np.abs(slopes)) <= _MAX_SLOPE * peak
            and np.max(np.abs(curvature)) <= _MAX_CURVATURE * peak
        )

    return context.work_plan.fact(
        FactKey(("multitone-rwa-smooth", policy, spec.identity), bool), build
    )


def _retained_norm(
    model: SecularModel,
    port: int,
    frequency: float,
    work_plan: WorkPlan,
    retention: RWARetention | None = None,
) -> float:
    """Cache the near-resonant positive Fourier operator spectral norm."""
    key = FactKey(
        (
            "multitone-rwa-retained-norm",
            RWA_POLICY,
            ADAPTIVE_RWA_POLICY,
            model.identity,
            model.level,
            port,
            retained_mask(model, (frequency,), retention).tobytes(),
        ),
        float,
    )

    def build() -> float:
        return float(
            np.linalg.norm(retained_edges(model, port, frequency, retention), 2)
        )

    return work_plan.fact(key, build)


def _resolve_retention(
    model: SecularModel,
    ports: tuple[int, ...],
    carriers: tuple[float, ...],
    amplitudes: NDArray[np.float64],
    duration_ns: float,
    *,
    limits: tuple[float | None, int | None] = (None, None),
) -> RWARetention:
    """Resolve model-only retained edges for single or simultaneous carriers.

    Start at 50 MHz. Restore over-threshold one-photon edges and the strongest
    missing path per over-threshold two-photon channel/time order, with adjoints.
    Recompute after each round. Monotone addition terminates on a finite graph.
    A finite phase_limit also restores channels with excessive residual phase;
    dimension_limit stops expansion when the real graph dimension is too large.
    limits is (phase_limit in rad or None, dimension_limit or None).
    None skips the corresponding analysis, including its compiled helpers.
    Single constant callers keep their original phase/strength/pruning policy.
    """
    phase_limit, dimension_limit = limits
    quadratures = (
        np.array([model.drives[p].in_phase_ghz for p in ports]),
        np.array([model.drives[p].quadrature_ghz for p in ports]),
    )
    # Match immutable model quadratures used by coefficient construction, so
    # admission does not compile an otherwise identical writeable specialization.
    for quadrature in quadratures:
        quadrature.setflags(write=False)
    frequencies = np.array(carriers)
    initial = _rwa_fourier.window_mask(model.energies_ghz, frequencies)
    retained = initial
    probability = phase = np.inf
    dimension: int | None = None
    omission_computed = False
    while True:
        if dimension_limit is not None:
            dimension = int(_rwa_fourier.retained_dimension(retained, quadratures))
            if dimension > dimension_limit:
                break
        probability, phase, additions = _rwa_fourier.omission_analysis(
            model.energies_ghz,
            quadratures,
            frequencies,
            (amplitudes, duration_ns),
            retained,
            phase_limit,
        )
        omission_computed = True
        if not np.any(additions & ~retained):
            break
        retained = retained | additions
        omission_computed = False
    edges = tuple(
        tuple((int(i), int(j)) for i, j in np.argwhere(retained[a, 0] & ~initial[a, 0]))
        for a in range(len(ports))
    )
    return RWARetention(
        edges,
        float(probability),
        None if phase_limit is None else float(phase),
        dimension,
        omission_computed,
    )


def constant_retention(
    model: SecularModel,
    port: int,
    frequency: float,
    amplitude: float,
    duration_ns: float,
    work_plan: WorkPlan,
) -> RWARetention:
    """Resolve and cache one constant row's Fourier set before strength checks.

    Inputs are a valid model drive index, signed carrier with magnitude >=50 MHz,
    finite gain and nonnegative ns duration. The table inventory and standalone
    diagnostic share this fact without computing strength or phase for other rows.
    """
    key = FactKey(
        (
            "constant-rwa-retention",
            ADAPTIVE_RWA_POLICY,
            model.identity,
            model.level,
            port,
            frequency,
            amplitude,
            duration_ns,
        ),
        RWARetention,
    )
    return work_plan.fact(
        key,
        lambda: _resolve_retention(
            model, (port,), (frequency,), np.array([abs(amplitude)]), duration_ns
        ),
    )


def constant_admission(
    model: SecularModel,
    port: int,
    frequency: float,
    amplitude: float,
    duration_ns: float,
    work_plan: WorkPlan,
) -> ConstantRWAAdmission:
    """Cache the shared single-constant metrics and resolved Fourier set.

    Caller supplies a valid drive index, signed carrier with magnitude >=50 MHz,
    finite gain and nonnegative ns duration. No budget or density enters these
    comparisons. Guard and public diagnostics consume this same product. Kappa2
    uses the same resolved omitted edges as the endpoint kick and is computed
    before strength/phase checks. A failed strength comparison skips the
    phase-radian value, matching guard short-circuiting.
    """
    key = FactKey(
        (
            "constant-rwa-admission",
            SINGLE_CONSTANT_RWA_POLICY,
            model.identity,
            model.level,
            port,
            frequency,
            amplitude,
            duration_ns,
        ),
        ConstantRWAAdmission,
    )

    def build() -> ConstantRWAAdmission:
        retention = constant_retention(
            model, port, frequency, amplitude, duration_ns, work_plan
        )
        coefficients = rwa_kick_coefficients(
            model, port, frequency, amplitude, retention
        )
        kappa2 = float(np.linalg.norm(np.abs(coefficients).sum(axis=0), ord=2))
        norms = quadrature_norms(model, port, work_plan)
        retained = _retained_norm(model, port, frequency, work_plan, retention)
        # Preserve the table guard's NumPy scalar arithmetic, including overflow.
        gain = np.float64(amplitude)
        carrier = np.float64(abs(frequency))
        ratio = abs(gain) * (norms.in_phase + norms.quadrature) / carrier
        phase_rate = (gain * retained) ** 2 / carrier
        strength = RWAGuardMetric(float(ratio), _MAX_RATIO)
        phase: RWAGuardMetric | RWAUncomputed = RWAUncomputed()
        if strength.passed:
            phase = RWAGuardMetric(
                float(2 * np.pi * duration_ns * phase_rate), _MAX_PHASE_RAD
            )
        return ConstantRWAAdmission(
            retention,
            RWAGuardMetric(abs(frequency), _rwa_fourier.EDGE_GHZ, ">="),
            RWAGuardMetric(kappa2, _MAX_KAPPA2),
            strength,
            phase,
        )

    return work_plan.fact(key, build)


def constant_retentions(
    table: SegmentTable, context: KernelContext
) -> ConstantRetentions:
    """Publish one retained-set inventory for a complete constant-row table.

    Each eligible row uses its own model, signed carrier, abs(gain) and duration;
    no edges are shared across rows. Ineligible rows have None. The table/source
    scoped fact is built once, then guard, keys and per-key builders reuse it.
    Graph facts and prepare keys additionally identify the resolved edge set.
    This avoids one resource lookup per row at every subsequent prepare step.
    """
    key = FactKey(
        ("constant-rwa-retentions", ADAPTIVE_RWA_POLICY, id(table), id(context.source)),
        ConstantRetentions,
    )

    def build() -> ConstantRetentions:
        results: list[RWARetention | None] = []
        for row in range(len(table.timeline)):
            if (
                table.tones[row] != 1
                or table.envelopes[table.tone_envelope[row, 0]].kind != "constant"
                or abs(table.tone_frequency_ghz[row, 0]) < _rwa_fourier.EDGE_GHZ
                or not fixed_flux(table, row)
            ):
                results.append(None)
                continue
            model = table.model(context.source, row)
            results.append(
                constant_retention(
                    model,
                    int(table.tone_port[row, 0]),
                    float(table.tone_frequency_ghz[row, 0]),
                    float(table.tone_amplitude[row, 0]),
                    float(table.duration_ns[row]),
                    context.work_plan,
                )
            )
        return ConstantRetentions(tuple(results))

    return context.work_plan.fact(key, build)


def multitone_retention(
    table: SegmentTable, row: int, context: KernelContext
) -> RWARetention:
    """Resolve a row's omitted channels using peak-weighted envelope bounds.

    Caller establishes the shaped-single or multitone fixed-flux/smooth domain. The
    row-local fact includes signed carriers, gains, peaks and duration; it never
    depends on other scan rows or input densities. Keys and preparation reuse
    the final mask. Phase and dimension limits are part of its policy identity.
    """
    count = int(table.tones[row])
    model = table.model(context.source, row)
    ports = tuple(int(v) for v in table.tone_port[row, :count])
    frequencies = tuple(float(v) for v in table.tone_frequency_ghz[row, :count])
    bounds = envelope_bounds(table, context.work_plan)
    amplitudes = (
        np.abs(table.tone_amplitude[row, :count])
        * bounds.peaks[table.tone_envelope[row, :count]]
    )
    return _multitone_retention(
        model,
        ports,
        frequencies,
        amplitudes,
        float(table.duration_ns[row]),
        context.work_plan,
    )


def _multitone_retention(
    model: SecularModel,
    ports: tuple[int, ...],
    frequencies: tuple[float, ...],
    amplitudes: NDArray[np.float64],
    duration_ns: float,
    work_plan: WorkPlan,
) -> RWARetention:
    """Share peak-weighted resolved masks between admission and preparation.

    Inputs are aligned valid drive indices, signed GHz, nonnegative peak gains
    and row duration in ns. Preserve the table retention identity and limits.
    Numerical builder failures propagate and WorkPlan shares them without retry.
    """
    key = FactKey(
        (
            "multitone-retention",
            MULTITONE_RETENTION_POLICY,
            model.identity,
            model.level,
            ports,
            frequencies,
            tuple(amplitudes),
            duration_ns,
        ),
        RWARetention,
    )
    return work_plan.fact(
        key,
        lambda: _resolve_retention(
            model,
            ports,
            frequencies,
            amplitudes,
            duration_ns,
            limits=(_MAX_OMISSION_PHASE_RAD, _MAX_RETAINED_DIMENSION),
        ),
    )


def multitone_constant_admission(
    model: SecularModel,
    ports: tuple[int, int],
    frequencies: tuple[float, float],
    amplitudes: tuple[float, float],
    controls: tuple[float, float],
    work_plan: WorkPlan,
) -> MultitoneRWAAdmission:
    """Publish constant-pair admission once for complete model and row inputs.

    Caller establishes fixed flux and constant envelopes. ports are valid drive
    indices in carrier order; frequencies are finite signed GHz and amplitudes
    finite signed gains. controls is (nonnegative row duration in ns, positive
    budget). Same-port, low-budget and low-carrier pairs return rejected facts.
    Invalid budgets raise ValueError; numerical construction failures propagate.
    No phase, absolute start, density, history or cross-call cache enters this fact.
    """
    duration_ns, budget = controls
    tolerances(budget)
    key = FactKey(
        (
            "multitone-constant-admission",
            MULTITONE_GUARD_POLICY,
            model.identity,
            model.level,
            ports,
            frequencies,
            amplitudes,
            duration_ns,
            budget,
        ),
        MultitoneRWAAdmission,
    )

    def build() -> MultitoneRWAAdmission:
        allowance = RWAGuardMetric(budget, MIN_RWA_BUDGET, ">=")
        uncomputed = RWAUncomputed()
        if not allowance.passed or ports[0] == ports[1]:
            return MultitoneRWAAdmission(allowance, uncomputed, None, uncomputed, False)
        carrier = RWAGuardMetric(
            min(abs(f) for f in frequencies), _rwa_fourier.EDGE_GHZ, ">="
        )
        if not carrier.passed:
            return MultitoneRWAAdmission(allowance, carrier, None, uncomputed, False)
        gains = np.array(amplitudes)
        resolved = _multitone_retention(
            model, ports, frequencies, np.abs(gains), duration_ns, work_plan
        )
        secular: RWAGuardMetric | RWAUncomputed = uncomputed
        if (
            resolved.dimension is not None
            and resolved.dimension <= _MAX_RETAINED_DIMENSION
        ):
            quadratures = (
                np.array([model.drives[p].in_phase_ghz for p in ports]),
                np.array([model.drives[p].quadrature_ghz for p in ports]),
            )
            secular = RWAGuardMetric(
                float(
                    _rwa_fourier.secular_accumulation(
                        model.energies_ghz,
                        quadratures,
                        np.array(frequencies),
                        gains,
                        retained_mask(model, frequencies, resolved),
                        duration_ns,
                    )
                ),
                _MAX_MULTITONE_SECULAR_CYCLES,
            )
        accepted = (
            _resolved_retention_admissible(resolved)
            and isinstance(secular, RWAGuardMetric)
            and secular.passed
        )
        return MultitoneRWAAdmission(
            allowance,
            carrier,
            resolved,
            secular,
            accepted,
            omission_transfer=(
                RWAGuardMetric(
                    resolved.omission_transfer, _rwa_fourier.MAX_OMISSION_TRANSFER
                )
                if resolved.omission_computed
                else uncomputed
            ),
            omission_phase_rad=(
                RWAGuardMetric(resolved.omission_phase, _MAX_OMISSION_PHASE_RAD)
                if resolved.omission_computed and resolved.omission_phase is not None
                else uncomputed
            ),
            real_dimension=(
                RWAGuardMetric(resolved.dimension, _MAX_RETAINED_DIMENSION)
                if resolved.dimension is not None
                else uncomputed
            ),
        )

    return work_plan.fact(key, build)


def admit_multitone_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit ordinary multitone RWA with adaptive omission and dimension limits.

    Requires two tones at fixed flux and the shared minimum budget. Constant
    pairs require distinct ports. Sampled rows use these proxies before a
    separate measured guard; smoothness does not reject them. Uses row-local retained
    transfer, phase and dimension bounds, not strength or carrier-spacing caps.
    Constant pairs also require epsilon_dom**2 * Omega_ret * row duration
    <=1.476e-4 cycle after retention. Invalid budgets raise ValueError.
    """
    pairs = table.tones[rows] == 2
    accepted = np.zeros(len(rows), dtype=np.bool_)
    accepted[pairs] = _admit_rows(
        table, rows[pairs], context, single_tone=False, adaptive_multitone=True
    )
    return accepted


def admit_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit simultaneous rows under the shared calibrated joint RWA policy.

    Requires at least two tones. The unified omission rule applies only to
    distinct-port dual constants; sampled/gate policies remain unchanged.
    Reads model-only facts, never density;
    splitting rows preserves decisions. Invalid budgets raise ValueError.
    """
    return _admit_rows(table, rows, context, single_tone=False)


def admit_single_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit exactly one shaped tone for covariant gates, not row RWA routing.

    Uses shared omission-driven retention with calibrated transfer, residual
    phase and block limits. Smoothness, fixed-flux and budget still apply.
    """
    return _admit_rows(table, rows, context, single_tone=True)


def admit_constant_rows(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    """Admit single constant rows under the same kappa2/strength/phase policy.

    Fixed flux, |f|>=50 MHz and budget>=1e-3 are required. Graph pruning is a
    separate consumer admission; this function never reads input density.
    """
    return _admit_rows(table, rows, context, single_tone=True, constant_single=True)


def _constant_domain(
    table: SegmentTable, row: int, *, constant: bool, single: bool
) -> bool:
    """Keep single constants separate from distinct-port dual constants.

    Same-port Stark/probe spectra retain their Floquet/direct route: small
    population errors can still reorder a narrow spectrum's local maxima.
    Shaped rows use the existing simultaneous or single-gate domain.
    """
    if single:
        return constant
    return not constant or (
        table.tones[row] == 2 and table.tone_port[row, 0] != table.tone_port[row, 1]
    )


def _row_admissible(
    table: SegmentTable,
    row: int,
    count: np.int64,
    context: KernelContext,
    *,
    single_tone: bool,
) -> bool:
    """Require the existing budget, tone-count and fixed-flux domain."""
    return not (
        context.budget < MIN_RWA_BUDGET
        or (count != 1 if single_tone else count < 2)
        or not fixed_flux(table, int(row))
    )


def _admissible_frequencies(
    table: SegmentTable,
    row: int,
    count: np.int64,
    envs: NDArray[np.int64],
    context: KernelContext,
    *,
    adaptive_multitone: bool,
) -> NDArray[np.float64] | None:
    """Check carriers; only single-tone gates admit smooth complex samples."""
    if not adaptive_multitone and not all(
        _smooth(table, int(e), context, allow_complex=count == 1) for e in envs
    ):
        return None
    frequencies = np.abs(table.tone_frequency_ghz[row, :count])
    if np.min(frequencies) < _rwa_fourier.EDGE_GHZ or (
        not adaptive_multitone
        and np.any(np.diff(np.sort(frequencies)) < 2 * _rwa_fourier.EDGE_GHZ)
    ):
        return None
    return frequencies


def _resolved_retention_admissible(resolved: RWARetention) -> bool:
    """Check the requested transfer, phase and retained-block limits."""
    # Shaped single gates and ordinary multitone request both limits.
    assert resolved.dimension is not None and resolved.omission_phase is not None
    return (
        resolved.omission_transfer <= _rwa_fourier.MAX_OMISSION_TRANSFER
        and resolved.omission_phase <= _MAX_OMISSION_PHASE_RAD
        and resolved.dimension <= _MAX_RETAINED_DIMENSION
    )


def _retained_limits_admissible(
    table: SegmentTable,
    row: int,
    ratio: float | np.float64,
    phase: float | np.float64,
    retention: RWARetention | None,
    *,
    dual_constant: bool | np.bool_,
) -> bool:
    """Check retained strength before reading duration for the phase cap."""
    phase_limit = _MAX_CONSTANT_PHASE_RAD if dual_constant else _MAX_PHASE_RAD
    return bool(
        ratio <= _MAX_RATIO
        and 2 * np.pi * table.duration_ns[row] * phase <= phase_limit
        and (
            retention is None
            or retention.omission_transfer <= _rwa_fourier.MAX_OMISSION_TRANSFER
        )
    )


def _admit_rows(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    *,
    single_tone: bool,
    constant_single: bool = False,
    adaptive_multitone: bool = False,
) -> NDArray[np.bool_]:
    tolerances(context.budget)
    bounds = envelope_bounds(table, context.work_plan)
    accepted = np.zeros(len(rows), dtype=np.bool_)
    for index, raw_row in enumerate(rows):
        row = int(raw_row)
        count = table.tones[row]
        envs = table.tone_envelope[row, :count]
        constant = all(table.envelopes[e].kind == "constant" for e in envs)
        if adaptive_multitone and constant:
            if count != 2 or not fixed_flux(table, row):
                continue
            accepted[index] = multitone_constant_admission(
                table.model(context.source, row),
                (int(table.tone_port[row, 0]), int(table.tone_port[row, 1])),
                (
                    float(table.tone_frequency_ghz[row, 0]),
                    float(table.tone_frequency_ghz[row, 1]),
                ),
                (
                    float(table.tone_amplitude[row, 0]),
                    float(table.tone_amplitude[row, 1]),
                ),
                (float(table.duration_ns[row]), context.budget),
                context.work_plan,
            ).accepted
            continue
        if not _row_admissible(table, row, count, context, single_tone=single_tone):
            continue
        if not _constant_domain(
            table, int(row), constant=constant, single=constant_single
        ):
            continue
        frequencies = _admissible_frequencies(
            table, row, count, envs, context, adaptive_multitone=adaptive_multitone
        )
        if frequencies is None:
            continue
        model = table.model(context.source, row)
        if adaptive_multitone or (single_tone and not constant_single):
            resolved = multitone_retention(table, int(row), context)
            accepted[index] = _resolved_retention_admissible(resolved)
            continue
        retention = None
        if constant_single:
            constant_retentions(table, context)
            accepted[index] = constant_admission(
                model,
                int(table.tone_port[row, 0]),
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_amplitude[row, 0]),
                float(table.duration_ns[row]),
                context.work_plan,
            ).accepted
            continue
        if constant:
            ports = table.tone_port[row, :count]
            probability = _rwa_fourier.omission_probability(
                model.energies_ghz,
                np.array([model.drives[p].in_phase_ghz for p in ports]),
                np.array([model.drives[p].quadrature_ghz for p in ports]),
                table.tone_frequency_ghz[row, :count],
                np.abs(table.tone_amplitude[row, :count]) * bounds.peaks[envs],
                float(table.duration_ns[row]),
            )
            if probability > _rwa_fourier.MAX_OMISSION_TRANSFER:
                continue
        ratio = 0.0
        phase = 0.0
        for tone, env in enumerate(envs):
            port = int(table.tone_port[row, tone])
            norms = quadrature_norms(model, port, context.work_plan)
            retained = _retained_norm(
                model,
                port,
                float(table.tone_frequency_ghz[row, tone]),
                context.work_plan,
                retention,
            )
            phase += (
                table.tone_amplitude[row, tone] * bounds.peaks[env] * retained
            ) ** 2 / frequencies[tone]
            ratio += (
                abs(table.tone_amplitude[row, tone])
                * bounds.peaks[env]
                * (norms.in_phase + norms.quadrature)
                / frequencies[tone]
            )
        accepted[index] = _retained_limits_admissible(
            table, row, ratio, phase, retention, dual_constant=constant and count == 2
        )
    return accepted


def retained_mask(
    model: SecularModel,
    frequencies: tuple[float, ...],
    retention: RWARetention | None = None,
) -> NDArray[np.bool_]:
    """Return the exact final (tones,signs,M,M) Fourier-edge mask.

    Start with each signed 50 MHz window and union all adaptively restored
    positive edges and their adjoints. Equal masks permit graph sharing, but
    never sharing of frequency-dependent virtual diagonals or row admission.
    Misaligned retained sets raise ValueError. The caller owns the result.
    """
    mask = _rwa_fourier.window_mask(model.energies_ghz, np.array(frequencies))
    if retention is not None:
        if len(retention.edges) != len(frequencies):
            raise ValueError("RWA retention must align with carriers")
        for tone, edges in enumerate(retention.edges):
            for i, j in edges:
                mask[tone, 0, i, j] = True
                mask[tone, 1, j, i] = True
    return mask


def retained_edges(
    model: SecularModel,
    port: int,
    frequency: float,
    retention: RWARetention | None = None,
    tone: int = 0,
) -> NDArray[np.complex128]:
    """Return the positive-carrier joint operator in GHz/gain.

    The negative-carrier operator is its adjoint. Both RWA integrators and
    covariant gate construction default to the same 50 MHz retained-edge policy.
    retention overrides it with the resolved edges for the ordered tone index.
    The caller owns the returned mutable matrix.
    """
    drive = model.drives[port]
    gaps = model.energies_ghz[:, None] - model.energies_ghz[None, :]
    coupling = _rwa_fourier.fourier_coefficient(
        drive.in_phase_ghz, drive.quadrature_ghz, 1
    )
    mask = np.abs(gaps + frequency) < _rwa_fourier.EDGE_GHZ
    if retention is not None:
        for i, j in retention.edges[tone]:
            mask[i, j] = True
    return coupling * mask


def rwa_kick_coefficients(
    model: SecularModel,
    port: int,
    frequency: float,
    amplitude: float,
    retention: RWARetention,
) -> NDArray[np.complex128]:
    """Prepare caller-owned (2,M,M) omitted c/d for a constant-tone kick.

    port indexes the model drive; frequency is signed GHz, amplitude is signed
    gain. retention is this row's resolved window plus restored edges, not the
    pruned phase graph. Signs are ordered (+1,-1). Includes diagonal +/-f
    terms, excludes every retained edge and its adjoint. Outputs are
    dimensionless; phase and duration enter only during endpoint playback.
    """
    drive = model.drives[port]
    return _rwa_fourier.kick_coefficients(
        model.energies_ghz,
        drive.in_phase_ghz,
        drive.quadrature_ghz,
        frequency,
        amplitude,
        retained_mask(model, (frequency,), retention)[0],
    )


def rwa_coefficients(
    model: SecularModel,
    ports: tuple[int, ...],
    frequencies: tuple[float, ...],
    retention: RWARetention | None = None,
    *,
    work_plan: WorkPlan | None = None,
) -> _rwa_fourier.RWACoefficients:
    """Build all joint drive coefficients for ordered signed carriers.

    This is the shared coefficient owner for ordinary RWA and covariant gates.
    Each tone contributes sum_j,s |C_s[i,j]|²/(Ei-Ej+s*f) to the diagonal
    in GHz/gain², summed over the final omitted Fourier mask. Without adaptive
    retention this is abs(Ei-Ej+s*f)>=0.05 GHz. Retained edges evolve explicitly,
    with no zero division or double counting. The common
    compiled builder also serves batched ordinary RWA preparation.
    retention overrides the initial window in both Fourier terms and virtual
    diagonal, so restored edges are never counted twice. Graph pruning may
    replace the Fourier operators but retains the diagonal.
    work_plan shares only model packing, never frequency-dependent diagonals.
    Returned arrays are owned and mutable. Misaligned/empty carriers raise
    ValueError; ports must index the supplied model's validated drives.
    """
    if not ports or len(ports) != len(frequencies):
        raise ValueError("RWA coefficients require aligned nonempty carriers")
    retained = None
    if retention is not None and any(retention.edges):
        retained = retained_mask(model, frequencies, retention)
    positive, negative, diagonal = _rwa_fourier.coefficients(
        pack_models([model]) if work_plan is None else packed_model(model, work_plan),
        np.array([ports], dtype=np.int64),
        np.array([frequencies], dtype=np.float64),
        np.array([len(ports)], dtype=np.int64),
        retained,
    )
    return _rwa_fourier.RWACoefficients(positive[0], negative[0], diagonal[0])
