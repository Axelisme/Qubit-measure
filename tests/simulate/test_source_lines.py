"""Physical source filtering precedes loop mixing and keeps passive tails."""

import numpy as np
import pytest
from scipy.integrate import quad
from zcu_tools.simulate import lines
from zcu_tools.simulate.evolution import ControlPulse, ControlSequence, Envelope


def test_source_line_delay_tail_and_independent_shot() -> None:
    from zcu_tools.simulate.lines import FilteredFlux, FluxLine

    controls = ControlSequence(
        (
            ControlPulse("a", 10.0, 30.0, Envelope.flat(-2.0)),
            ControlPulse("b", 0.0, 20.0, Envelope.flat(1.0)),
        )
    )
    lines = (
        FluxLine("a", (1.0, 2.0, 0.0), delay_ns=4.0, tau_ns=10.0),
        FluxLine("b", (0.0, -1.0, 3.0), delay_ns=0.0, tau_ns=0.0),
    )
    flux = FilteredFlux(np.array([0.1, 0.2, 0.3]), controls, lines)
    np.testing.assert_allclose(flux.at(10.0), [0.1, -0.8, 3.3], atol=1e-12, rtol=0)
    a = -2 * (1 - np.exp(-2)) * np.exp(-0.6)
    np.testing.assert_allclose(
        flux.at(40.0), [0.1 + a, 0.2 + 2 * a, 0.3], atol=1e-12, rtol=0
    )
    assert flux.at(40.0)[0] != pytest.approx(0.1)
    repeat = FilteredFlux(np.array([0.1, 0.2, 0.3]), controls, lines)
    np.testing.assert_array_equal(repeat.at(10.0), flux.at(10.0))


@pytest.mark.parametrize("dimensions", [1, 2, 4])
def test_source_line_maps_any_positive_loop_dimension(dimensions: int) -> None:
    coefficients = tuple(float(i + 1) for i in range(dimensions))
    try:
        line = lines.FluxLine("bias", coefficients, delay_ns=3.0)
        flux = lines.FilteredFlux(
            np.zeros(dimensions),
            ControlSequence((ControlPulse("bias", 2, 10, Envelope.flat(-2)),)),
            (line,),
        )
    except ValueError as error:
        pytest.fail(f"positive loop dimension {dimensions} must be supported: {error}")
    np.testing.assert_array_equal(flux.at(4), np.zeros(dimensions))
    np.testing.assert_array_equal(flux.at(5), -2 * np.array(coefficients))
    np.testing.assert_array_equal(flux.at(13), np.zeros(dimensions))
    assert flux.event_times_ns == (5.0, 13.0)


def test_source_line_rejects_empty_or_mismatched_loop_dimensions() -> None:
    try:
        line = lines.FluxLine("bias", (1.0,))
    except ValueError as error:
        pytest.fail(f"single-loop calibration must be accepted: {error}")
    with pytest.raises(ValueError, match="coefficient"):
        lines.FluxLine("bias", ())
    for dc in (np.array([]), np.zeros((1, 1)), np.array([np.nan])):
        with pytest.raises(ValueError, match="DC flux"):
            lines.FilteredFlux(dc, ControlSequence(), ())
    with pytest.raises(ValueError, match="dimension|shape"):
        lines.FilteredFlux(np.zeros(2), ControlSequence(), (line,))


@pytest.mark.parametrize("tau0", [0.0, 4.0])
def test_multiexponential_step_and_switch_off_keep_signed_causal_tails(
    tau0: float,
) -> None:
    assert hasattr(lines, "DistortionTerm"), "source lines need long-tail terms"
    terms = (lines.DistortionTerm(0.3, 12), lines.DistortionTerm(-0.1, 40))
    line = lines.FluxLine("bias", (1.0,), 3, tau0, distortion_terms=terms)
    flux = lines.FilteredFlux(
        np.array([0.25]),
        ControlSequence((ControlPulse("bias", 2, 10, Envelope.flat(-2)),)),
        (line,),
    )

    def step(t: float) -> float:
        if t < 0:
            return 0.0
        fast = 0.8 * np.exp(-t / tau0) if tau0 else 0.0
        return float(1 - fast - 0.3 * np.exp(-t / 12) + 0.1 * np.exp(-t / 40))

    for time in (4.0, 5.0, 7.0, 12.0, 13.0, 17.0, 90.0):
        expected = 0.25 - 2 * (step(time - 5) - step(time - 13))
        np.testing.assert_allclose(flux.at(time), [expected], atol=2e-13, rtol=0)
    assert flux.at(90)[0] != pytest.approx(0.25, abs=1e-6)
    # Reconstruction is independent of evaluation order and retains no shot state.
    np.testing.assert_array_equal(flux.at(4), [0.25])


@pytest.mark.parametrize("tau0", [0.0, 4.0])
def test_multiexponential_filter_convolves_real_shaped_carrier(tau0: float) -> None:
    assert hasattr(lines, "DistortionTerm"), "source lines need long-tail terms"
    envelope = Envelope.sampled(
        np.array([0.0, 8.0]), np.array([0.2 + 0.3j, 1.0 - 0.2j])
    )
    flux = lines.FilteredFlux(
        np.zeros(1),
        ControlSequence((ControlPulse("bias", 2, 10, envelope, 0.031, 0.4),)),
        (lines.FluxLine("bias", (1.0,), 3, tau0, (lines.DistortionTerm(0.25, 20),)),),
    )

    def source(t: float) -> float:
        value = (0.2 + 0.3j) + (0.1 - 0.0625j) * t
        carrier = np.exp(1j * (2 * np.pi * 0.031 * t + 0.4))
        return float((value * carrier).real)

    for time in (5.0, 7.0, 13.0, 25.0):
        local = time - 5
        direct = 0.75 * source(local) if tau0 == 0 and local < 8 else 0.0

        def integrand(t: float, local: float = local) -> float:
            kernel = 0.25 / 20 * np.exp(-(local - t) / 20)
            if tau0:
                kernel += 0.75 / tau0 * np.exp(-(local - t) / tau0)
            return float(kernel * source(t))

        expected = direct + quad(integrand, 0, min(local, 8), epsabs=1e-13)[0]
        np.testing.assert_allclose(flux.at(time), [expected], atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    ("amplitude", "tau"),
    [
        (np.nan, 1.0),
        (np.inf, 1.0),
        (0.1, 0.0),
        (0.1, -1.0),
        (0.1, np.inf),
        (0.1, np.nan),
    ],
)
def test_distortion_terms_reject_nonfinite_or_nonpositive_parameters(
    amplitude: float, tau: float
) -> None:
    assert hasattr(lines, "DistortionTerm"), "long-tail validation needs DistortionTerm"
    with pytest.raises(ValueError, match="finite|positive"):
        lines.DistortionTerm(amplitude, tau)
