"""Physical source filtering precedes loop mixing and keeps passive tails."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import ControlPulse, ControlSequence, Envelope


def test_source_line_delay_tail_and_independent_shot() -> None:
    from zcu_tools.simulate.lines import FilteredFlux, FluxLine

    controls = ControlSequence((ControlPulse("a", 10., 30., Envelope.flat(-2.)), ControlPulse("b", 0., 20., Envelope.flat(1.))))
    lines = (FluxLine("a", (1., 2., 0.), delay_ns=4., tau_ns=10.), FluxLine("b", (0., -1., 3.), delay_ns=0., tau_ns=0.))
    flux = FilteredFlux(np.array([.1, .2, .3]), controls, lines)
    np.testing.assert_allclose(flux.at(10.), [.1, -.8, 3.3], atol=1e-12)
    a = -2 * (1 - np.exp(-2)) * np.exp(-.6)
    np.testing.assert_allclose(flux.at(40.), [.1 + a, .2 + 2*a, .3], atol=1e-12)
    assert flux.at(40.)[0] != pytest.approx(.1)
    repeat = FilteredFlux(np.array([.1, .2, .3]), controls, lines)
    np.testing.assert_array_equal(repeat.at(10.), flux.at(10.))
