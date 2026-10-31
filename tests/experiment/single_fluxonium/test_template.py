"""Public compiled-template planning, without evolving or constructing an ADC."""

from tests.core._simulation import TIME_BASE, tick_tuple
import importlib
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    ChargeDriveRequirement,
    PulseEnvelope,
    ReadoutWindow,
)
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.single_fluxonium import SingleFluxoniumSource
from zcu_tools.resources.single_fluxonium import parse_single_fluxonium

from ._inputs import settings, snapshot, template


@pytest.fixture
def charge_preflight():
    values = settings().model_dump(mode="json")
    values["drive_lines"]["second"] = {
        "name": "second_charge",
        "gen_ch": 2,
        "charge_drive_ghz_per_gain": {"real": 0.1, "imag": 0},
    }
    values["flux_line"]["gen_ch"] = 3
    source = SingleFluxoniumSource(parse_single_fluxonium(values), DeviceManager())
    original = template()
    times = np.linspace(0, 0.008, 17)
    gaussian = PulseEnvelope(
        672,  # 0.5 ns on the synthetic clock.
        tuple(complex(x) for x in np.exp(-(((times - 0.004) / 0.002) ** 2))),
    )
    flux_event = replace(original.events[1], gen_ch=3, envelopes=(gaussian,))
    compiled = replace(
        original,
        points=np.array([0, 1], dtype=np.int64),
        events=(original.events[0], flux_event, original.events[2]),
    )
    request = AcquireRequest(
        "integrated",
        (1, 2),
        0,
        (ReadoutWindow(0, 0, 4, TIME_BASE.period_ticks(1000), tick_tuple((0.018,))),),
        123,
        0,
        "test",
        point_dims=(2,),
        point_names=("scan",),
        templates=(compiled,),
        charge_requirements=(ChargeDriveRequirement("Stokes", 3),),
     time_base=TIME_BASE)
    try:
        yield source, request
    finally:
        source.close()


@pytest.mark.parametrize("channel", [3, 0, 4])
def test_preflight_rejects_charge_request_on_noncharge_generator(
    charge_preflight, channel
):
    source, request = charge_preflight
    request = replace(
        request, charge_requirements=(ChargeDriveRequirement("Stokes", channel),)
    )
    with pytest.raises(ValueError, match=rf"Stokes.*{channel}.*charge drive line"):
        source.preflight(request, snapshot())


def test_preflight_accepts_named_charge_requirements_with_legal_fast_flux(
    charge_preflight,
):
    source, request = charge_preflight
    request = replace(
        request,
        charge_requirements=(
            ChargeDriveRequirement("Pump", 1),
            ChargeDriveRequirement("arbitrary label", 2),
        ),
    )
    batch = source.preflight(request, snapshot()).plans[0][0].batch
    assert [track.port for track in batch.tracks] == ["mw"]
    assert batch.flux_waveform is not None
    assert np.max(batch.flux_waveform.flux) > 0.5


def test_plan_preserves_complete_timeline_and_filtered_flux() -> None:
    planner = importlib.import_module("zcu_tools.experiment.single_fluxonium.template")
    plan = planner.plan_template(settings(), snapshot(), template(), time_base=TIME_BASE)
    batch = plan.batch
    np.testing.assert_array_equal(batch.points, [0, 3])
    np.testing.assert_array_equal(batch.end_ns, [16, 16])
    assert len(batch.tracks) == 1
    track = batch.tracks[0]
    assert track.port == "mw"
    np.testing.assert_array_equal(track.start_ns, [2, 2])
    np.testing.assert_array_equal(track.duration_ns, [6, 6])
    np.testing.assert_array_equal(track.frequency_ghz, [0.25, 0.25])
    np.testing.assert_allclose(track.phase_rad, [np.pi / 2, np.pi / 2])
    np.testing.assert_array_equal(track.amplitude, [-0.2, -0.2])
    assert batch.flux.level.shape[1] == 1
    np.testing.assert_array_equal(batch.flux.level[batch.flux.offsets[:-1]], [[0], [0]])
    waveform = batch.flux_waveform
    assert waveform is not None
    times = waveform.time_ns[: waveform.offsets[1]]
    values = waveform.flux[: waveform.offsets[1], 0]
    # Analytic first-order step: delayed turn-on at 2 ns, switch-off at 10 ns.
    assert np.interp(4, times, values) == pytest.approx(0.5 + 0.1 * (1 - np.exp(-1)))
    assert np.interp(16, times, values) == pytest.approx(
        0.5 + 0.1 * (1 - np.exp(-4)) * np.exp(-3)
    )
    assert plan.readout.pulses[0][0].start_ns == 16
    assert plan.readout.pulses[0][0].amplitude == 1
    np.testing.assert_array_equal(plan.readout.grids[0][0].times_ns, [18, 19, 20, 21])
    # Detached plan arrays do not change when borrowed template storage changes.
    original = template()
    detached = planner.plan_template(settings(), snapshot(), original, time_base=TIME_BASE)
    original.events[0].gain[:] = 0.9
    np.testing.assert_array_equal(detached.batch.tracks[0].amplitude, [-0.2, -0.2])


def test_ideal_flux_platforms_use_levels_without_artificial_ramps() -> None:
    planner = importlib.import_module("zcu_tools.experiment.single_fluxonium.template")
    config = settings()
    if config.flux_line is None:
        raise AssertionError("pulsed fixture requires a bound fast-flux line")
    config = config.model_copy(
        update={"flux_line": config.flux_line.model_copy(update={"tau_ns": 0})}
    )
    original = template()
    flux_event = replace(original.events[1], active=np.array([True, False]))
    original = replace(
        original, events=(original.events[0], flux_event, original.events[2])
    )
    plan = planner.plan_template(config, snapshot(), original, time_base=TIME_BASE)
    assert plan.batch.flux_waveform is None
    flux = plan.batch.flux
    np.testing.assert_array_equal(flux.offsets, [0, 3, 4])
    np.testing.assert_array_equal(flux.start_ns, [0, 2, 10, 0])
    np.testing.assert_array_equal(flux.level[:, 0], [0, 100, 0, 0])
