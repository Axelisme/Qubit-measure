"""Public stationary R-pi saved-summary analysis contracts."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.analysis.readout_chain import RPiSums, rpi_analysis


def model_sums(
    retention: float = 0.75,
    amplitude: float = 0.0225,
    noise: float = 1.0,
    mean: tuple[float, float] = (0.0, 0.0),
    lengths: tuple[int, ...] = (200000,),
    copies: tuple[int, ...] = (1,),
) -> RPiSums:
    """Exact symmetric two-state moments; a = d²/4 and noise = sigma²."""
    count = sum(length * copies[i] for i, length in enumerate(lengths))
    pairs = np.array(
        [
            sum(max(length - lag, 0) * copies[i] for i, length in enumerate(lengths))
            for lag in range(1, 16)
        ],
        dtype=np.int64,
    )
    center = np.array(mean, dtype=np.float64)
    offset = float(center @ center)
    return RPiSums(
        read_count=count,
        first_moment_sums=count * center,
        same_read_square_sum=count * (offset + amplitude + 2 * noise),
        lag_product_sums=pairs
        * (offset + amplitude * (-retention) ** np.arange(1, 16)),
        pair_counts=pairs,
        segment_lengths=np.array(lengths, dtype=np.int64),
        segment_counts=np.array(copies, dtype=np.int64),
    )


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0),
        (0, 1.5),
        (1, 97),
        (1, 2**54),
        (7, np.nan),
        (20, -1),
        (20, 94.5),
        (20, 95),
    ],
)
def test_processor_decoder_rejects_malformed_counts_and_sums(index: int, value: float):
    record = np.zeros(35, dtype=np.float64)
    record[:5] = [2, 96, 1, 2, 100]
    record[20:35] = np.arange(94, 65, -2)
    record[index] = value
    with pytest.raises(ValueError, match="R-pi|pair_counts"):
        RPiSums.from_processor(record)


@pytest.mark.parametrize(
    "record", [np.zeros(34), np.zeros((1, 35)), np.zeros(35, dtype=np.float32)]
)
def test_processor_decoder_requires_one_float64_setting_record(record):
    with pytest.raises(ValueError, match="shape.*float64"):
        RPiSums.from_processor(record)


@pytest.mark.parametrize("pi_error", [0.0, 0.04])
def test_exact_two_state_moments_recover_effective_retention_and_noise(
    pi_error: float,
) -> None:
    sums = model_sums(
        retention=0.75 * (1 - 2 * pi_error), mean=(1.2, -0.7), lengths=(1000000,)
    )
    saved = sums.lag_product_sums.copy()

    result = rpi_analysis(sums)

    assert result.valid, result.invalid_reasons
    assert result.gate_passed
    assert result.q_eff == pytest.approx(0.75 * (1 - 2 * pi_error), rel=1e-11)
    assert result.amplitude == pytest.approx(0.0225, rel=1e-11)
    assert result.noise_variance == pytest.approx(1.0, rel=1e-11)
    assert result.q == pytest.approx(0.0225, rel=1e-11)
    assert_allclose(sums.lag_product_sums, saved, rtol=0, atol=0)


def test_white_noise_variance_includes_offset_and_each_finite_segment_edge() -> None:
    sums = model_sums(mean=(4.0, -3.0), lengths=(3, 17, 31, 64), copies=(7, 2, 1, 3))
    # Independent dense quadratic-form oracle: x.T A x, off-diagonal c_j/2.
    weights = np.array([-1, 3, -4, 4, -4, 4, -4, 4, -4, 4, -4, 4, -4, 3, -1])
    trace_square = 0.0
    mean_term = 0.0
    for length, copies in zip(sums.segment_lengths, sums.segment_counts, strict=True):
        matrix = np.zeros((length, length))
        for t in range(length):
            for u in range(t + 1, min(t + 16, length)):
                matrix[t, u] = weights[u - t - 1] / sums.pair_counts[u - t - 1] / 2
                matrix[u, t] = matrix[t, u]
        trace_square += copies * np.trace(matrix @ matrix)
        mean_term += copies * np.sum((matrix @ np.ones(length)) ** 2)
    # Known r0=(a+2 sigma²)/2 from exact supplied same-read moments.
    expected = 4 * 1.01125**2 * trace_square + 4 * 1.01125 * 25 * mean_term

    result = rpi_analysis(sums)

    assert result.null_variance == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("retention", [0.75, 0.999, 0.9999999, 1.0])
def test_contrast_score_does_not_extrapolate_a_near_unit_retention_tail(
    retention: float,
) -> None:
    # All inputs have T=0.2 and r0=1. Q_eff may change diagnoses, not this score.
    coefficient = (
        sum(retention**j for j in range(2, 15)) * (1 + retention) ** 2 / retention
    )
    amplitude = 0.2 / coefficient
    sums = model_sums(retention=retention, amplitude=amplitude, noise=1 - amplitude / 2)
    # Finite C_j are not approximated by N.
    weights_squared = np.array([1, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 9, 1])
    expected = 0.2 - 3.719 * np.sqrt(2 * np.sum(weights_squared / sums.pair_counts))

    result = rpi_analysis(sums)

    assert result.valid, result.invalid_reasons
    assert result.q_eff == pytest.approx(retention)
    assert result.score == pytest.approx(expected)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"read_count": 200001}, "read_count"),
        ({"read_count": True}, "read_count"),
        ({"read_count": 200000.0}, "read_count"),
        ({"first_moment_sums": np.zeros(3)}, "shape"),
        ({"first_moment_sums": np.zeros(2, dtype=np.float32)}, "float64"),
        ({"lag_product_sums": np.zeros(14)}, "shape"),
        ({"lag_product_sums": np.full(15, np.nan)}, "finite"),
        ({"same_read_square_sum": np.inf}, "finite"),
        ({"pair_counts": np.full(15, 199999)}, "pair_counts"),
        ({"pair_counts": np.full(15, 199999.0)}, "integer"),
        ({"segment_lengths": np.array([0])}, "positive"),
        ({"segment_counts": np.array([-1])}, "positive"),
        ({"segment_counts": np.array([1, 2])}, "shape"),
    ],
)
def test_malformed_saved_summaries_raise_at_the_analysis_entry(
    changes: dict[str, object],
    message: str,
) -> None:
    sums = replace(model_sums(), **changes)
    with pytest.raises(ValueError, match=message):
        rpi_analysis(sums)


def test_missing_pairs_are_an_invalid_estimate_not_fabricated_lags() -> None:
    result = rpi_analysis(model_sums(lengths=(8,), copies=(10,)))

    assert not result.valid
    assert "zero_denominator" in result.invalid_reasons
    assert not result.gate_passed
    assert result.contrast is None
    assert result.score is None


@pytest.mark.parametrize(
    "retention, amplitude, noise, reason",
    [
        (0.75, 0.0, 1.0, "zero_denominator"),
        (-0.2, 1.0, 1.0, "q_eff_out_of_range"),
        (1.2, 1.0, 1.0, "q_eff_out_of_range"),
        (0.75, -0.1, 1.0, "nonpositive_amplitude"),
        (1.0, 1.0, 0.0, "nonpositive_noise_variance"),
        (1.0, 1.0, -0.2, "nonpositive_noise_variance"),
        (0.75, 0.0, 0.0, "zero_denominator"),
    ],
)
def test_invalid_estimates_keep_named_reasons_and_never_offer_a_score(
    retention: float,
    amplitude: float,
    noise: float,
    reason: str,
) -> None:
    result = rpi_analysis(model_sums(retention, amplitude, noise))

    assert not result.valid
    assert reason in result.invalid_reasons
    assert result.score is None


def measured(value: float | None) -> float:
    """Require a computable public diagnostic before statistical comparisons."""
    assert value is not None
    return value


@pytest.mark.parametrize("mean", [(0.0, 0.0), (4.0, -3.0)])
def test_white_noise_contrast_has_unit_standardized_scale(
    mean: tuple[float, float],
) -> None:
    # 800 independent trials, 24 disconnected 48-read segments each; 921600 reads.
    # Relative SD sampling SE is about 1/sqrt(2*799)=2.5%; 10% is four SE.
    rng = np.random.default_rng(49183)
    samples = rng.normal(size=(800, 24, 48, 2)) + np.array(mean)
    first = samples.sum(axis=(1, 2))
    squares = np.sum(samples**2, axis=(1, 2, 3))
    products = np.stack(
        [
            np.sum(samples[:, :, :-lag] * samples[:, :, lag:], axis=(1, 2, 3))
            for lag in range(1, 16)
        ],
        axis=1,
    )
    standardized = []
    for trial in range(800):
        result = rpi_analysis(
            RPiSums(
                read_count=1152,
                first_moment_sums=first[trial],
                same_read_square_sum=float(squares[trial]),
                lag_product_sums=products[trial],
                pair_counts=24 * np.arange(47, 32, -1, dtype=np.int64),
                segment_lengths=np.array([48], dtype=np.int64),
                segment_counts=np.array([24], dtype=np.int64),
            )
        )
        standardized.append(
            measured(result.contrast) / np.sqrt(measured(result.null_variance))
        )

    assert np.std(standardized, ddof=1) == pytest.approx(1.0, abs=0.10)
    assert abs(np.mean(standardized)) < 0.15  # >4 mean-standard-errors.


@pytest.mark.parametrize("reads", [200000, 1000000000, 1000000000000])
def test_fixed_subthreshold_contrast_stays_closed_as_read_count_grows(
    reads: int,
) -> None:
    # Analytic moments scale counts only; no billion-read arrays or raw-shot simulation.
    # a=0.001 gives T/r0 about 0.00896, below the 0.0224 minimum effect.
    result = rpi_analysis(model_sums(amplitude=0.001, lengths=(reads,)))

    assert not result.gate_passed
    assert "gate_failed" in result.invalid_reasons
    assert measured(result.contrast) < 0.0224 * result.same_read_variance
    assert result.score is None


def test_d_over_sigma_point_three_qnd_point_seven_five_passes_at_200k_reads() -> None:
    result = rpi_analysis(model_sums())

    assert result.valid, result.invalid_reasons
    assert result.gate_passed
    assert result.same_read_variance == pytest.approx(1.01125)
    assert (
        measured(result.contrast) - 3.719 * np.sqrt(measured(result.null_variance))
        > 0.0224 * result.same_read_variance
    )


def test_three_state_moments_show_residual_slow_mode_bias_not_a_leakage_correction() -> (
    None
):
    # A reversible three-state oracle: stationary g/e/f=(0.4,0.4,0.2),
    # alternating eigenvalue -0.75, slow eigenvalue 0.95. No fitted oracle.
    transition = np.array([[0.12, 0.87, 0.01], [0.87, 0.12, 0.01], [0.02, 0.02, 0.96]])
    population = np.array([0.4, 0.4, 0.2])
    centers = np.array([[-0.3, 0.0], [0.3, 0.0], [0.0, 0.8]])
    sums = model_sums(lengths=(1000000,))
    lag_moments = [
        np.sum(
            (population[:, None] * np.linalg.matrix_power(transition, lag))
            * (centers @ centers.T)
        )
        for lag in range(1, 16)
    ]
    sums = replace(
        sums,
        first_moment_sums=sums.read_count * (population @ centers),
        same_read_square_sum=float(
            sums.read_count * (population @ np.sum(centers**2, axis=1) + 2)
        ),
        lag_product_sums=sums.pair_counts * np.array(lag_moments),
    )

    result = rpi_analysis(sums)

    assert result.valid, result.invalid_reasons
    # This particular slow mode is only approximately removed. These bounds are
    # a counterexample to exact general three-state recovery, not a universal bound.
    assert 0.7501 < measured(result.q_eff) < 0.7503
    assert 0.071 < measured(result.amplitude) < 0.073
    assert 1.0 < measured(result.noise_variance) < 1.001
