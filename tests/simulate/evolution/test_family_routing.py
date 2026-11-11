"""Acquire-wide family allowlists at the public Engine boundary."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    FluxExpansion,
    FluxWaveform,
    ModelSource,
)
from zcu_tools.simulate.evolution.kernel import GuardResult

from ._engine import RecordingKernel, TimelineKernel, batch, run, source, track

pytestmark = pytest.mark.sim


def shaped_envelope(duration=3.0):
    return EnvelopeSpec("sampled", np.array([0.5, 1.0]), np.array([0.0, duration]))


def test_unknown_family_fails_at_construction():
    with pytest.raises(ValueError, match="unknown.*not_a_family"):
        Engine(1, families=("not_a_family",))


def test_allowlist_excludes_other_admitted_families():
    envelope = shaped_envelope()
    timeline = replace(batch(1), tracks=(track(1, envelope),))
    kernels = tuple(
        RecordingKernel(name)
        for name in (
            "single_tone_gate",
            "shaped_sambe",
            "gate_cache",
            "period_table",
            "dop853",
        )
    )
    result = run(
        Engine(1, kernels=kernels, families=("dop853",)), source(), (timeline,)
    )
    assert result.paths.segments_by_kernel == {"dop853": 1}


@pytest.mark.parametrize("family", ["single_tone_gate", "common_frame"])
def test_allowed_expensive_family_is_tried_without_its_competitors(family):
    envelope = shaped_envelope()
    pulse = track(
        1, envelope if family == "single_tone_gate" else EnvelopeSpec("constant")
    )
    tracks = (
        (pulse,)
        if family == "single_tone_gate"
        else (pulse, replace(pulse, frequency_ghz=np.array([0.13])))
    )
    timeline = replace(batch(1), tracks=tracks)
    result = run(
        Engine(1, kernels=(RecordingKernel(family),), families=(family,)),
        source(),
        (timeline,),
    )
    assert result.paths.segments_by_kernel == {family: 1}


@pytest.mark.parametrize(
    "domain",
    ["idle", "adiabatic", "flux", "slow", "plateau", "shaped", "two", "multiple"],
)
def test_empty_domain_reports_affected_points_and_segments(domain):
    timeline = replace(batch(1), points=np.array([42]))
    if domain not in ("idle", "adiabatic"):
        envelope = shaped_envelope() if domain == "shaped" else EnvelopeSpec("constant")
        count = {"two": 2, "multiple": 3}.get(domain, 1)
        timeline = replace(
            timeline,
            tracks=tuple(
                replace(
                    track(1, envelope),
                    frequency_ghz=np.array(
                        [0.02 if domain == "slow" else 0.1 + index * 0.03]
                    ),
                )
                for index in range(count)
            ),
        )
    src = source()
    if domain in ("adiabatic", "flux"):
        timeline = replace(
            timeline,
            flux_waveform=FluxWaveform(
                np.array([0, 2]), np.array([0.0, 3.0]), np.array([[0.0], [0.0001]])
            ),
        )
        src = ModelSource(
            src.initial_density,
            src.model,
            FluxExpansion(np.diag([0.0, 0.1, 0.4]).astype(complex), ()),
        )
    with pytest.raises(
        RuntimeError, match=rf"domain {domain}.*points.*42.*segments.*0"
    ):
        run(Engine(1, families=()), src, (timeline,))


def test_pinned_family_rejected_by_numerical_guard_does_not_reroute():
    timeline = replace(batch(1), tracks=(track(1, EnvelopeSpec("constant")),))
    with pytest.raises(RuntimeError, match=r"domain plateau.*points.*0.*segments.*0"):
        run(Engine(1, families=("dyson",)), source(), (timeline,))


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("accepts", [False, True])
def test_prefix_and_deferred_workers_keep_the_acquire_allowlist(prefix, accepts):
    class LocalPeriod(TimelineKernel):
        guard_builds = True

        def guard_prepared(self, table, rows, context):
            return GuardResult(np.full(len(rows), accepts))

    period = LocalPeriod("period_table")
    pulse = replace(track(2, shaped_envelope()), amplitude=np.array([0.1, 0.2]))
    tracks = (pulse,)
    if prefix:
        tracks = (
            track(2, shaped_envelope(1.0), duration=1),
            replace(
                pulse,
                start_ns=np.ones(2),
                duration_ns=np.full(2, 2.0),
                envelopes=(shaped_envelope(2.0),),
            ),
        )
    timeline = replace(batch(2), tracks=tracks)
    engine = Engine(
        2, kernels=(RecordingKernel("shaped_sambe"), period), families=("period_table",)
    )
    if accepts:
        result = run(engine, source(), (timeline,))
        assert result.paths.segments_by_kernel == {"period_table": 4 if prefix else 2}
        assert result.paths.points_by_class == {"period_table": 2}
    else:
        with pytest.raises(RuntimeError, match=r"domain shaped.*points.*segments"):
            run(engine, source(), (timeline,))


def test_allowlist_does_not_extend_common_frame_to_three_tones():
    pulse = track(1, EnvelopeSpec("constant"))
    timeline = replace(
        batch(1),
        tracks=tuple(
            replace(pulse, frequency_ghz=np.array([frequency]))
            for frequency in (0.1, 0.13, 0.16)
        ),
    )
    with pytest.raises(RuntimeError, match=r"domain multiple.*points.*segments"):
        run(
            Engine(
                1,
                kernels=(RecordingKernel("common_frame"),),
                families=("common_frame",),
            ),
            source(),
            (timeline,),
        )


@pytest.mark.parametrize("offset", [0.0, 1e-11])
def test_common_frame_zero_mean_carriers_report_domain_and_points(offset):
    pulse = track(1, EnvelopeSpec("constant"))
    timeline = replace(
        batch(1),
        points=np.array([42]),
        tracks=(pulse, replace(pulse, frequency_ghz=np.array([-0.1 + offset]))),
    )
    with pytest.raises(RuntimeError, match=r"domain two.*points.*42.*segments.*0"):
        run(Engine(1, families=("common_frame",)), source(), (timeline,))
    default = run(Engine(1), source(), (timeline,))
    unrestricted = run(Engine(1, families=None), source(), (timeline,))
    assert "common_frame" not in default.paths.segments_by_kernel
    assert unrestricted.paths == default.paths
    np.testing.assert_array_equal(unrestricted.densities, default.densities)


def test_none_preserves_default_routing_and_density():
    timeline = replace(
        batch(2),
        tracks=(
            replace(track(2, EnvelopeSpec("constant")), amplitude=np.full(2, 1e-5)),
        ),
    )
    default = run(Engine(1), source(), (timeline,))
    unrestricted = run(Engine(1, families=None), source(), (timeline,))
    assert unrestricted.paths == default.paths
    np.testing.assert_array_equal(unrestricted.densities, default.densities)
