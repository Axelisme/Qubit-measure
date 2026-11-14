"""Read-only constant RWA diagnostics agree with public Engine admission."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from .accuracy_cases import coherent_source, drive_track, linear_drive_model, timeline


def expect_metric(
    metric: evolution.RWAGuardMetric | evolution.RWAUncomputed,
) -> evolution.RWAGuardMetric:
    """Require an actually computed numeric value, not a skipped metric."""
    assert isinstance(metric, evolution.RWAGuardMetric)
    return metric


@pytest.mark.sim
@pytest.mark.parametrize("kappa,accepted", [(0.079, True), (0.081, False)])
def test_kappa2_edge_diagnostic_matches_pinned_routing(
    kappa: float, accepted: bool
) -> None:
    model = linear_drive_model()
    x = np.diag([0.6, -0.3]).astype(complex)
    model = replace(
        model, drives=(evolution.SecularDrive("drive", x, np.zeros_like(x)),)
    )
    batch = timeline(1, 1.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(
                batch,
                evolution.EnvelopeSpec("constant"),
                frequency=0.6,
                gain=kappa,
            ),
        ),
    )
    diagnostic = evolution.single_tone_rwa_diagnostic(model, "drive", 0.6, kappa, 1.0)
    metric = expect_metric(diagnostic.kappa2)
    # Each +/-f term contributes |gain*X|/(2*f), so A=diag(kappa,kappa/2).
    # Its spectral norm is kappa, not zero (signed cancellation) or sqrt(1.25)*kappa.
    assert metric.value == pytest.approx(kappa)
    assert metric.comparison == "<="
    assert metric.passed is accepted
    assert diagnostic.accepted is accepted
    assert expect_metric(diagnostic.strength_ratio).passed
    assert expect_metric(diagnostic.phase_rad).value == pytest.approx(0.0)
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("single_tone_rwa",))) as engine,
    ):
        if accepted:
            result = engine.evolve(source, (batch,))
            assert dict(result.paths.segments_by_kernel) == {"single_tone_rwa": 1}
        else:
            with pytest.raises(evolution.NoValidKernel, match="domain plateau"):
                engine.evolve(source, (batch,))


@pytest.mark.sim
@pytest.mark.parametrize(
    "scale,expected_phase,accepted",
    [(1.0, 0.05999994, True), (np.sqrt(1.000001 / 0.999999), 0.06000006, False)],
)
def test_phase_edge_diagnostic_matches_routing(
    scale: float, expected_phase: float, accepted: bool
) -> None:
    # Keep the original phase boundary independent of the kappa2 edge fixture.
    amplitude = np.sqrt(0.06 * 0.999999 * 0.6 / (2 * np.pi * 8)) / 0.001 * scale
    batch = timeline(2, 8.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(
                batch, evolution.EnvelopeSpec("constant"), frequency=0.6, gain=amplitude
            ),
        ),
    )
    assert hasattr(evolution, "single_tone_rwa_diagnostic"), (
        "constant RWA guard must expose its numeric admission diagnostics"
    )
    diagnostic = evolution.single_tone_rwa_diagnostic(
        linear_drive_model(), "drive", 0.6, amplitude, 8.0
    )
    assert expect_metric(diagnostic.phase_rad).value == pytest.approx(expected_phase)
    assert expect_metric(diagnostic.phase_rad).passed is accepted
    assert diagnostic.accepted is accepted
    assert expect_metric(diagnostic.kappa2).passed
    assert expect_metric(diagnostic.strength_ratio).passed
    if accepted:
        assert expect_metric(diagnostic.pruning).passed
        assert expect_metric(diagnostic.real_dimension).passed
    else:
        assert diagnostic.pruning == evolution.RWAUncomputed()
        assert diagnostic.real_dimension == evolution.RWAUncomputed()
    with (
        closing(coherent_source(linear_drive_model())) as source,
        closing(evolution.Engine(1)) as engine,
    ):
        result = engine.evolve(source, (batch,))
        assert dict(result.paths.segments_by_kernel) == {
            "single_tone_rwa" if accepted else "floquet_period": 2
        }
        # Reading diagnostics cannot alter the source's later pinned admission.
        with closing(evolution.Engine(1, families=("single_tone_rwa", "t4"))) as pinned:
            if accepted:
                result = pinned.evolve(source, (batch,))
                assert dict(result.paths.segments_by_kernel) == {"single_tone_rwa": 2}
            else:
                with pytest.raises(RuntimeError, match="domain plateau"):
                    pinned.evolve(source, (batch,))


@pytest.mark.sim
@pytest.mark.parametrize("frequency", [0.049999, -0.049999, 0.0])
def test_low_carrier_reports_rejection_without_later_metrics(frequency: float) -> None:
    model = linear_drive_model()
    diagnostic = evolution.single_tone_rwa_diagnostic(
        model, "drive", frequency, 0.08, 8.0
    )
    assert not diagnostic.accepted
    assert expect_metric(diagnostic.carrier_ghz).value == abs(frequency)
    assert not expect_metric(diagnostic.carrier_ghz).passed
    assert diagnostic.kappa2 == evolution.RWAUncomputed()
    assert diagnostic.strength_ratio == evolution.RWAUncomputed()
    assert diagnostic.phase_rad == evolution.RWAUncomputed()
    assert diagnostic.pruning == evolution.RWAUncomputed()
    assert diagnostic.real_dimension == evolution.RWAUncomputed()
    batch = timeline(1, 8.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(
                batch,
                evolution.EnvelopeSpec("constant"),
                frequency=frequency,
                gain=0.08,
            ),
        ),
    )
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("single_tone_rwa", "t4"))) as engine,
        pytest.raises(RuntimeError, match="domain (slow|plateau)"),
    ):
        engine.evolve(source, (batch,))
