"""Detached private-input storage, not normal data.h5 or waveform assets."""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    ChargeDriveRequirement,
    EnvironmentSnapshot,
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
    TimeBase,
)
from zcu_tools.simulators.ftf.config import parse_ftf

from tests.core._simulation import (
    TIME_BASE,
    single_point_template,
    tick_array,
    tick_tuple,
    ticks,
)
from tests.simulators.ftf.config._data import ftf_document


@pytest.mark.parametrize("field", ["reps", "relax_delay_ticks"])
def test_replay_reader_requires_runtime_inputs(field: str) -> None:
    from pydantic import ValidationError
    from zcu_tools.simulators.replay import ReplayAcquire

    request = AcquireRequest(
        "integrated", (1,), 0, (), 123, 0, "test-rng-v1", time_base=TIME_BASE
    )
    fixed = ReplayAcquire(
        settings=parse_ftf(ftf_document()), request=request, versions=()
    )
    document = json.loads(fixed.model_dump_json())
    del document["request"][field]
    with pytest.raises(ValidationError, match=rf"request\.{field}"):
        ReplayAcquire.model_validate_json(json.dumps(document))


@pytest.mark.parametrize("requirements", [(), (ChargeDriveRequirement("drive", 2),)])
@pytest.mark.parametrize("time_base", [TIME_BASE, TimeBase(Fraction(5, 3))])
def test_replay_retains_complex_samples_and_deduplicated_acquire(
    tmp_path: Path,
    requirements: tuple[ChargeDriveRequirement, ...],
    time_base: TimeBase,
) -> None:
    from zcu_tools.simulators.replay import ReplayAcquire, read_replay, save_replay

    window = ReadoutWindow(0, 0, 3, TIME_BASE.period_ticks(100.0), tick_tuple((0.02,)))
    event = PulseEvent(
        2,
        ticks(0.0),
        ticks(0.016),
        42.0,
        90.0,
        -0.3,
        PulseEnvelope(ticks(0.008), (1 + 2j, -3 - 4j)),
    )
    timeline = PointTimeline((), (event,), (window,), ticks(0.05), time_base=TIME_BASE)
    request = AcquireRequest(
        "integrated",
        (1,),
        0,
        (window,),
        12345,
        17,
        "test-rng-v1",
        templates=(single_point_template(timeline),),
        charge_requirements=requirements,
        time_base=TIME_BASE,
    )
    request = replace(request, time_base=time_base)
    fixed = ReplayAcquire(
        settings=parse_ftf(ftf_document()),
        request=request,
        versions=(("backend", "test-v1"),),
    )
    a = EnvironmentSnapshot("ftf-v2", "fixed-model", (("bias", 0.2),))
    b = EnvironmentSnapshot("ftf-v2", "fixed-model", (("bias", 0.4),))
    first = save_replay(tmp_path, fixed, a, 0)
    second = save_replay(tmp_path, fixed, b, 1)
    restored = read_replay(first)
    assert restored.acquire.request.materialize_point(0) == request.materialize_point(0)
    assert restored.acquire.request.charge_requirements == requirements
    assert restored.round.snapshot == a
    assert read_replay(second).round.snapshot == b
    assert restored.round.acquire_identity == read_replay(second).round.acquire_identity
    assert restored.acquire.settings == fixed.settings
    # A missing shared input is an error, not current configuration substitution.
    acquire_path = tmp_path / "acquires" / f"{restored.round.acquire_identity}.json"
    acquire_path.unlink()
    with pytest.raises(FileNotFoundError):
        read_replay(second)


def test_single_fluxonium_replay_roundtrip_keeps_its_own_settings(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError
    from zcu_tools.simulators.replay import ReplayAcquire, read_replay, save_replay
    from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium

    settings = parse_single_fluxonium(
        {
            "format": "zcu.single-fluxonium",
            "version": 1,
            "chip": {
                "qubit": {"ej_ghz": 4, "ec_ghz": 1, "el_ghz": 1},
                "temperature_k": 0,
            },
            "dc_bias": {
                "offset_phi0": 0.5,
                "source": {
                    "name": "bias",
                    "unit": "A",
                    "limits": [-1, 1],
                    "initial": 0,
                    "ramp_step": 0.01,
                    "flux_per_unit": 0.1,
                },
            },
            "drive_line": {
                "name": "mw",
                "gen_ch": 1,
                "charge_drive_ghz_per_gain": {"real": 0.1, "imag": 0},
            },
            "flux_line": {"name": "ff", "gen_ch": 2, "flux_phi0_per_gain": 0.2},
            "readout": {
                "drive": {"gen_ch": 0, "input_per_gain": 10},
                "resonator": {"thermal_occupation": None, 
                    "frequency_ghz": 7,
                    "charge_coefficient_ghz": 0,
                    "kappa_external_mhz": 1,
                    "kappa_internal_mhz": 0,
                },
                "adc": {"background": {"ripple_period_mhz": 1000.0, "pivot_ghz": 0.0}, "ro_ch": 0, "gain": 1, "noise_psd_adc2_ns": [[0, 0], [0, 0]]},
            },
            "numerics": {"projection_prune_floor": 1e-9, "slow_envelope_segments_per_period": 20, 
                "truncation": {"native_cutoff": 12, "levels": 4},
                "solver": {
                    "rtol": 1e-8,
                    "atol": 1e-10,
                    "max_step_ns": 0.1,
                    "bath_segment_ns": 2,
                },
                "seed": 123,
            },
        }
    )
    request = AcquireRequest(
        "integrated", (1,), 0, (), 123, 0, "test-rng-v1", time_base=TIME_BASE
    )
    try:
        fixed = ReplayAcquire(
            settings=settings, request=request, versions=(("backend", "single"),)
        )
    except ValidationError as error:
        pytest.fail(f"private replay must accept single-fluxonium settings: {error}")
    snapshot = EnvironmentSnapshot(
        "single-fluxonium-v1", "single-model", (("bias", 0.2),)
    )
    loaded = read_replay(save_replay(tmp_path, fixed, snapshot, 2))
    assert loaded.acquire.settings == settings
    assert loaded.acquire.settings.format == "zcu.single-fluxonium"
    assert loaded.round.snapshot == snapshot
    assert loaded.round.index == 2


def test_replay_rejects_float_time_format_version() -> None:
    from zcu_tools.simulators.replay import ReplayRound

    with pytest.raises(ValueError, match="version"):
        ReplayRound.model_validate(
            {
                "version": 1,
                "acquire_identity": "0" * 64,
                "snapshot": {"version": "test", "model_identity": "test", "values": []},
                "index": 0,
            }
        )


def test_replay_rejects_fractional_values_in_integer_time_arrays() -> None:
    from zcu_tools.simulators.replay import ReplayAcquire

    timeline = PointTimeline(
        (),
        (PulseEvent(0, 0, 1344, 100.0, 0.0, 0.1),),
        (),
        1344,
        time_base=TIME_BASE,
    )
    request = AcquireRequest(
        "integrated",
        (1,),
        0,
        (),
        123,
        0,
        "test",
        time_base=TIME_BASE,
        templates=(single_point_template(timeline),),
    )
    fixed = ReplayAcquire(
        settings=parse_ftf(ftf_document()), request=request, versions=()
    )
    payload = fixed.model_dump(mode="json")
    payload["request"]["templates"][0]["events"][0]["start_ticks"]["values"][0] = 0.5
    with pytest.raises(ValueError, match="int64"):
        ReplayAcquire.model_validate(payload)


def test_replay_rejects_changed_saved_content(tmp_path: Path) -> None:
    from zcu_tools.simulators.replay import read_replay

    damaged = tmp_path / ("0" * 64 + ".json")
    damaged.write_bytes(b"{}")
    with pytest.raises(ValueError, match="integrity"):
        read_replay(damaged)


def test_replay_round_trips_every_template_array_with_dtype_and_shape(tmp_path: Path):
    from dataclasses import fields

    import numpy as np
    from zcu_tools.core.simulation import EventSlot, ReadoutSlot, TimelineTemplate
    from zcu_tools.simulators.replay import ReplayAcquire, read_replay, save_replay

    event = EventSlot(
        gen_ch=2,
        source="selected",
        iterations=(("inner", 1),),
        start_ticks=tick_array(np.array([0.125, 0.25])),
        duration_ticks=tick_array(np.array([0.016, 0.032])),
        frequency_mhz=np.array([42.0, 43.125]),
        phase_deg=np.array([90.0, -30.0]),
        gain=np.array([-0.3, 0.25]),
        envelopes=(None, PulseEnvelope(ticks(0.008), (1 + 2j, -3 - 4j))),
        envelope_index=np.array([1, 0], dtype=np.int64),
        active=np.array([True, False], dtype=np.bool_),
        phase_reference_ticks=ticks(0.125),
    )
    readout = ReadoutSlot(
        channel=0,
        samples=3,
        sample_period_ticks=TIME_BASE.period_ticks(100.0),
        trigger_ticks=tick_array(np.array([[0.02, 0.04], [0.03, 0.05]])),
        frequency_mhz=np.array([[42.0, 43.0], [44.0, 45.0]]),
        phase_deg=np.array([[0.0, 90.0], [180.0, -90.0]]),
        generator_channels=np.array([[2, 0], [0, 2]], dtype=np.int64),
    )
    template = TimelineTemplate(
        points=np.array([0, 1], dtype=np.int64),
        events=(event,),
        readouts=(readout,),
        duration_ticks=tick_array(np.array([0.4, 0.5])),
    )
    request = AcquireRequest(
        mode="integrated",
        loop_dims=(3, 2),
        avg_level=0,
        readouts=(
            ReadoutWindow(
                0, 0, 3, TIME_BASE.period_ticks(100.0), tick_tuple((0.02, 0.04))
            ),
        ),
        root_seed=12345,
        sequence=17,
        rng_version="test-rng-v1",
        point_dims=(2,),
        point_names=("axis",),
        reps=3,
        templates=(template,),
        time_base=TIME_BASE,
    )
    acquire = ReplayAcquire(
        settings=parse_ftf(ftf_document()),
        request=request,
        versions=(("backend", "test-v1"),),
    )
    snapshot = EnvironmentSnapshot("test-v1", "detached")
    restored = read_replay(save_replay(tmp_path, acquire, snapshot, 2))
    actual = restored.acquire.request
    for field in fields(request):
        if field.name != "templates":
            assert getattr(actual, field.name) == getattr(request, field.name)
    assert len(actual.templates) == 1
    observed = actual.templates[0]
    assert len(observed.events) == len(template.events)
    assert len(observed.readouts) == len(template.readouts)
    pairs = [
        (observed, template),
        (observed.events[0], event),
        (observed.readouts[0], readout),
    ]
    for loaded, original in pairs:
        for field in fields(original):
            value, expected = getattr(loaded, field.name), getattr(original, field.name)
            if isinstance(expected, np.ndarray):
                assert value.dtype == expected.dtype
                assert value.shape == expected.shape
                np.testing.assert_array_equal(value, expected)
            elif field.name not in ("events", "readouts"):
                assert value == expected
    assert restored.round.snapshot == snapshot
    assert restored.round.index == 2
