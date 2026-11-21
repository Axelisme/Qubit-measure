"""Six-cavity observation routing through the detached request seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    DDSCongruence,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment.routing import route_request

from tests.core._simulation import TIME_BASE, single_point_template, tick_tuple, ticks
from tests.simulators.six_fluxonium.config._data import six_document


def raw_adc6_request() -> AcquireRequest:
    """One Q4 tone with an ADC word compiled to its folded DDS image."""
    images = (DDSCongruence(period_ghz=2.4576),)
    window = ReadoutWindow(
        6,
        13,
        4,
        TIME_BASE.period_ticks(1000),
        tick_tuple((0.018,)),
        frequency_mhz=(-197.80000019073486,),
        phase_deg=(0,),
        generator_channels=(13,),
        frequency_images=images,
    )
    timeline = PointTimeline(
        (),
        (PulseEvent(13, ticks(0.016), ticks(0.008), 7175, 0, 0.1),),
        (window,),
        ticks(0.024),
        time_base=TIME_BASE,
    )
    template = single_point_template(timeline)
    slot = replace(template.readouts[0], frequency_images=(images,))
    return AcquireRequest(
        "integrated",
        (1,),
        0,
        (window,),
        123,
        0,
        "test",
        templates=(replace(template, readouts=(slot,)),),
        time_base=TIME_BASE,
    )


def test_raw_adc6_observes_q4_by_dds_congruence():
    settings = parse_six_fluxonium(six_document())
    (group,) = route_request(settings, raw_adc6_request())
    assert group.cavity == 3  # Q4
    assert group.channels == (6,)
    np.testing.assert_array_equal(group.points, [0])


def test_raw_adc6_congruence_tie_names_the_window_and_both_cavities():
    document = six_document()
    document["qubits"]["Q5"]["readout"]["resonator"]["frequency_ghz"] = 9.6326
    settings = parse_six_fluxonium(document)
    with pytest.raises(ValueError, match=r"ADC 6 window 0.*Q4.*Q5"):
        route_request(settings, raw_adc6_request())
