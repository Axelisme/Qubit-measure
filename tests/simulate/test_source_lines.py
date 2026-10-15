"""Physical source filtering precedes loop mixing and keeps passive tails."""

import numpy as np
import pytest
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
