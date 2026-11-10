"""Stationary R-pi chain analysis from one setting's saved sums."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RPiSums:
    """One setting's completed stationary segments, pooled before normalization.

    read_count: Positive integer total number of reads, not reps or blocks.
    first_moment_sums: Float64 shape (2,), sum I and sum Q in normalized ADC units.
    same_read_square_sum: Sum (I²+Q²), in ADC units squared.
    lag_product_sums: Float64 shape (15,), sum x[t] dot x[t+j], j=1..15,
        in ADC units squared. No pairs may cross a segment boundary.
    pair_counts: Integer shape (15,), actual pair counts for those sums.
        C_j = sum(segment_counts * maximum(segment_lengths - j, 0)).
        A zero count requires a zero sum and makes the estimate invalid.
    segment_lengths: Integer shape (H,), positive segment lengths in reads.
    segment_counts: Integer shape (H,), positive multiplicities of those lengths.
        Repeated lengths are allowed. Each rep cycle is one segment; rep,
        round and setting boundaries are cuts. Transport chunks do not change
        cycle boundaries.

    There are 18 moment sums (2 first, 1 square, 15 lag), plus counts/metadata.
    Sum completed rounds before calling analysis; never average round estimates.
    No raw shots, program, hardware or file format is required. Construction
    only stores fields. rpi_analysis validates every call, including array edits.
    """

    read_count: int
    first_moment_sums: NDArray[np.float64]
    same_read_square_sum: float
    lag_product_sums: NDArray[np.float64]
    pair_counts: NDArray[np.int64]
    segment_lengths: NDArray[np.int64]
    segment_counts: NDArray[np.int64]

    @classmethod
    def from_processor(cls, record: NDArray[np.float64]) -> "RPiSums":
        """Decode one setting's pooled float64 shape (35,) R-pi record.

        Columns are 0: completed segment count S; 1: read count N; 2:4: ΣI,ΣQ;
        4: Σ(I²+Q²); 5:20: lag dot products j=1..15; 20:35: pair counts.
        First moments use normalized ADC units, square/products ADC units².
        Counts are exact integers encoded as float64, no larger than 2**53.
        All completed rounds must have the same segment length L=N/S. This
        H=1 encoding recovers segment_lengths=[L], segment_counts=[S]; it
        cannot encode mixed-length segments. Only add records of the same L.

        Return detached arrays without changing record. Raise ValueError for
        wrong shape/dtype, nonfinite values, noninteger or inconsistent counts,
        or zero segments/reads. No missing lags are synthesized. This validates
        the data contract, not the physical continuity or π calibration.
        """
        if record.shape != (35,) or record.dtype != np.float64:
            raise ValueError("R-pi record must have shape (35,) and dtype float64")
        if not np.isfinite(record).all():
            raise ValueError("R-pi record must contain finite sums and counts")
        counts = np.concatenate((record[:2], record[20:35]))
        if (
            np.any(counts < 0)
            or np.any(counts > 2**53)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("R-pi counts must be nonnegative exact integers <= 2**53")
        segments, reads = int(record[0]), int(record[1])
        if segments <= 0 or reads <= 0 or reads % segments:
            raise ValueError("R-pi N/S must be a positive integer segment length")
        result = cls(
            read_count=reads,
            first_moment_sums=record[2:4].copy(),
            same_read_square_sum=float(record[4]),
            lag_product_sums=record[5:20].copy(),
            pair_counts=record[20:35].astype(np.int64),
            segment_lengths=np.array([reads // segments], dtype=np.int64),
            segment_counts=np.array([segments], dtype=np.int64),
        )
        _validate_sums(result)
        return result


@dataclass(frozen=True)
class RPiAnalysis:
    """Pointwise estimates, contrast score and white-noise gate diagnostics.

    score: Dimensionless alternating-contrast score (T - 3.719 sqrt(V0)) / r0,
        only for valid points; None otherwise. Maximize over valid points only.
        This heuristic is not Fisher information, QND fidelity or MIST.
        It uses a finite lag window, not a Q_eff-dependent infinite tail.
        At high SNR it can prefer stronger readout than the information optimum.
    q: Dimensionless alternating amplitude / per-quadrature noise variance.
    q_eff: Dimensionless effective retention including pi errors, not corrected QND.
    amplitude: Alternating-mode amplitude a in ADC units squared.
    noise_variance: Per-quadrature sigma² in ADC units squared, NOT a standard error.
    contrast: Fixed lag contrast T in ADC units squared.
    null_variance: White-noise V0(T) in ADC units to the fourth power.
    same_read_variance: r0 = centered same-read variance / 2, in ADC units squared.
    gate_passed: Whether T - 3.719 sqrt(V0) > 0.0224 r0 for this point alone.
    invalid_reasons: Empty iff valid. Possible entries are gate_failed,
        q_eff_out_of_range, nonpositive_amplitude, nonpositive_noise_variance,
        zero_denominator, nonpositive_same_read_variance. Reasons can coexist.
        Computable invalid values remain unclipped. Uncomputable estimates and
        diagnostics are None, not successful fallback values. Estimation stops
        at a zero denominator or out-of-range Q_eff, so reasons are not exhaustive.
    """

    q: float | None
    q_eff: float | None
    amplitude: float | None
    noise_variance: float | None
    contrast: float | None
    null_variance: float | None
    same_read_variance: float
    gate_passed: bool
    invalid_reasons: tuple[str, ...]
    score: float | None = None

    @property
    def valid(self) -> bool:
        """True only when the gate and all physical estimate checks pass."""
        return not self.invalid_reasons


def rpi_analysis(sums: RPiSums) -> RPiAnalysis:
    """Analyze fixed lags 1..15 without changing the supplied saved arrays.

    Normalize first/same-read sums by N and each lag sum by its actual C_j.
    For j=2..14 form h_j=(-1)^j (2 M_j - M_{j-1} - M_{j+1}). Then
    T=sum(h_2..h_14), Q_eff=sum(h_3..h_14)/sum(h_2..h_13), and
    a=Q_eff*T / ((1+Q_eff)^2 * sum(Q_eff^j, j=2..14)). Subtract the
    stationary mean square from M_1 and M_2, then remove the alternating
    contribution to get b_1 and b_2. Set B=2*b_1-b_2 and sigma²=(V-a-B)/2.
    B approximates a slow mode; it is not a leakage probability or correction.
    No fit, adaptive cutoff, iteration or pi-error correction is performed.

    Write T=sum(w_j*M_j). For independent isotropic Gaussian IQ with mean mu
    and per-quadrature variance r0, each undirected within-segment pair has
    coefficient c_j=w_j/C_j. The null variance is
    V0=2*r0²*sum(w_j²/C_j) + r0*|mu|²*sum_t degree(t)²,
    where degree(t) sums c_j over all incident pairs at read t, segment by
    segment. This is exact for known model parameters, including short segment
    ends and unequal counts. The implementation visits at most 30 end reads per
    segment length and includes the interior term even when sum(w_j)=0.
    The same point's mu and r0=V/2 are plug-in estimates, NOT fitted sigma².
    The gate uses the one-sided nominal alpha=1e-4 normal scale 3.719 and fixed
    minimum effect 0.0224*r0. This is not exact finite-sample alpha or a CI.
    Correlated electronics can invalidate the white-noise scale. Passing the
    gate does not establish qubit identity, exclude leakage or remove systematic
    offsets larger than the minimum effect.

    Exact symmetric two-state stationary moments recover Q_eff=Q*(1-2*epsilon)
    for fixed pi-flip error epsilon. Unknown epsilon, nonstationary starts,
    third states and large IQ offsets can bias estimates. There is no warmup
    removal, general third-state correction, standard error or hardware check.

    Raise ValueError for malformed shapes/dtypes, nonfinite sums, nonpositive
    read/segment counts, or counts inconsistent with segment metadata. Valid
    summaries with missing pairs, zero denominators or nonphysical estimates
    return invalid results, as defined by RPiAnalysis. Inputs stay unchanged.
    """
    _validate_sums(sums)
    mean = sums.first_moment_sums / sums.read_count
    mean_square = float(mean @ mean)
    variance = sums.same_read_square_sum / sums.read_count - mean_square
    if np.any(sums.pair_counts == 0) or variance <= 0:
        reasons = ["gate_failed"]
        if np.any(sums.pair_counts == 0) or variance == 0:
            reasons.append("zero_denominator")
        if variance <= 0:
            reasons.append("nonpositive_same_read_variance")
        return RPiAnalysis(
            q=None,
            q_eff=None,
            amplitude=None,
            noise_variance=None,
            contrast=None,
            null_variance=None,
            same_read_variance=variance / 2,
            gate_passed=False,
            invalid_reasons=tuple(reasons),
        )
    moments = sums.lag_product_sums / sums.pair_counts
    signs = (-1.0) ** np.arange(2, 15)
    differences = signs * (2 * moments[1:14] - moments[:13] - moments[2:15])
    contrast = float(differences.sum())
    weights = np.zeros(15)
    weights[:13] -= signs
    weights[1:14] += 2 * signs
    weights[2:15] -= signs
    r0 = variance / 2
    null_variance = _null_variance(sums, weights, mean_square, r0)
    passed = contrast - 3.719 * np.sqrt(null_variance) > 0.0224 * r0
    result = RPiAnalysis(
        q=None,
        q_eff=None,
        amplitude=None,
        noise_variance=None,
        contrast=contrast,
        null_variance=null_variance,
        same_read_variance=r0,
        gate_passed=bool(passed),
        invalid_reasons=() if passed else ("gate_failed",),
    )
    result = _estimate(result, moments, differences, mean_square)
    if result.valid:
        result = replace(
            result, score=float((contrast - 3.719 * np.sqrt(null_variance)) / r0)
        )
    return result


def _validate_sums(sums: RPiSums) -> None:
    _validate_read_count(sums.read_count)
    for name, array, shape in (
        ("first_moment_sums", sums.first_moment_sums, (2,)),
        ("lag_product_sums", sums.lag_product_sums, (15,)),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if array.dtype != np.float64:
            raise ValueError(f"{name} must use float64")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite sums")
    if not np.isfinite(sums.same_read_square_sum):
        raise ValueError("same_read_square_sum must be finite")
    _validate_counts(sums)


def _validate_read_count(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError("read_count must be a positive integer")


def _validate_counts(sums: RPiSums) -> None:
    lengths = sums.segment_lengths
    if lengths.ndim != 1 or not lengths.size:
        raise ValueError("segment_lengths must have nonempty shape (H,)")
    for name, array, shape in (
        ("segment_lengths", lengths, lengths.shape),
        ("segment_counts", sums.segment_counts, lengths.shape),
        ("pair_counts", sums.pair_counts, (15,)),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must use an integer dtype")
        if np.any(array < 0) or (name != "pair_counts" and np.any(array == 0)):
            raise ValueError(
                f"{name} must contain positive segment values/nonnegative pairs"
            )
    segments = [
        (int(length), int(count))
        for length, count in zip(lengths, sums.segment_counts, strict=True)
    ]
    if sum(length * count for length, count in segments) != sums.read_count:
        raise ValueError("read_count disagrees with segment metadata")
    expected = [
        sum(max(length - lag, 0) * count for length, count in segments)
        for lag in range(1, 16)
    ]
    if not np.array_equal(sums.pair_counts, expected):
        raise ValueError("pair_counts disagree with segment metadata")
    if np.any(sums.lag_product_sums[sums.pair_counts == 0] != 0):
        raise ValueError("lag_product_sums must be zero when no pairs exist")


def _null_variance(
    sums: RPiSums,
    weights: NDArray[np.float64],
    mean_square: float,
    r0: float,
) -> float:
    coefficients = weights / sums.pair_counts
    degree_square = 0.0
    lags = np.arange(1, 16)
    for length, copies in zip(sums.segment_lengths, sums.segment_counts, strict=True):
        length = int(length)
        # Enumerate each boundary vertex once, including overlapping short ends.
        positions = (
            np.arange(length)
            if length <= 30
            else np.concatenate((np.arange(15), np.arange(length - 15, length)))
        )
        incidence = (positions[:, None] >= lags) + (
            positions[:, None] + lags < length
        ).astype(int)
        degrees = incidence @ coefficients
        interior = max(length - 30, 0) * (2 * coefficients.sum()) ** 2
        degree_square += int(copies) * (float(degrees @ degrees) + interior)
    return float(
        2 * r0**2 * np.sum(weights**2 / sums.pair_counts)
        + r0 * mean_square * degree_square
    )


def _estimate(
    result: RPiAnalysis,
    moments: NDArray[np.float64],
    differences: NDArray[np.float64],
    mean_square: float,
) -> RPiAnalysis:
    denominator = float(differences[:-1].sum())
    if denominator == 0:
        return replace(
            result, invalid_reasons=(*result.invalid_reasons, "zero_denominator")
        )
    retention = float(differences[1:].sum() / denominator)
    result = replace(result, q_eff=retention)
    if not 0 <= retention <= 1:
        return replace(
            result, invalid_reasons=(*result.invalid_reasons, "q_eff_out_of_range")
        )
    denominator = float((1 + retention) ** 2 * np.sum(retention ** np.arange(2, 15)))
    if denominator == 0:
        return replace(
            result, invalid_reasons=(*result.invalid_reasons, "zero_denominator")
        )
    amplitude = float(retention * differences.sum() / denominator)
    b1 = moments[0] - mean_square + amplitude * retention
    b2 = moments[1] - mean_square - amplitude * retention**2
    noise = float((2 * result.same_read_variance - amplitude - (2 * b1 - b2)) / 2)
    reasons = list(result.invalid_reasons)
    if amplitude <= 0:
        reasons.append("nonpositive_amplitude")
    if noise <= 0:
        reasons.append("nonpositive_noise_variance")
    if noise == 0:
        reasons.append("zero_denominator")
    return replace(
        result,
        amplitude=amplitude,
        noise_variance=noise,
        q=amplitude / noise if noise != 0 else None,
        invalid_reasons=tuple(reasons),
    )
