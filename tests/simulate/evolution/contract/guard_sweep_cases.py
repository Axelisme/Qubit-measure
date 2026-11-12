"""Importable finite sweeps of the four inaccurate guard-edge inputs.

These builders vary physical inputs only, never production guards. Gain scale
is a public admission-axis coordinate for common_frame and shaped_sambe, not
an observation of their private coherent/convergence indicators. No linear or
globally monotone relation between that coordinate and density error is assumed.
"""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec

from .accuracy_cases import AccuracyCase, coherent_source, drive_track, timeline
from .guard_edge_cases import guard_edge_case


def scaled_edge_case(family: str, scale: float) -> AccuracyCase:
    """Multiply all gains of an existing edge input by positive scale <= 1.

    Supported families are single_tone_rwa, common_frame and shaped_sambe.
    Duration, bath, phases, envelope and initial complex density stay unchanged.
    Each call owns a fresh ModelSource/history. Names encode every varied input.
    For single_tone_rwa the phase indicator is .05999994 * scale**2 rad.
    For the other two families only the public gain/admission bracket is known.
    """
    if family not in ("single_tone_rwa", "common_frame", "shaped_sambe"):
        raise ValueError(f"no gain sweep for {family}")
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    case = guard_edge_case(family)
    tracks = tuple(
        replace(track, amplitude=track.amplitude * scale) for track in case.batch.tracks
    )
    return replace(
        case,
        name=f"guard_edge/sweep/{family}/scale={scale:.17g}",
        batch=replace(case.batch, tracks=tracks),
    )


def gate_split_case(*, gamma: float, v: float, split: float) -> AccuracyCase:
    """Build a gate_cache input with d=(16*split/(gamma*v**2))**(1/3) ns.

    gamma is positive directional relaxation /ns; pure dephasing is gamma/2.
    v is positive gain*peak*norm(X) in GHz, without 2*pi; X is .002 GHz.
    split is the positive target d**3*gamma*v**2/16, at most .01. Use .00999999
    for an inside-edge sample rather than relying on rounding of exactly .01.
    All other physical inputs match guard_edge_case("gate_cache"): two points,
    coherent initial density, .6 GHz carrier and phases .17/1.37 rad. The
    envelope is always [.8, 1, .8] at 0,d/2,d (peak=1), including both gamma
    curves. This avoids the plateau extraction observed for the unity-sampled
    long inputs; it is not the original unity guard-edge case. Source/history
    are fresh on every call.
    """
    if gamma <= 0 or v <= 0 or not 0 < split <= 0.01:
        raise ValueError("gamma/v must be positive and split in (0, .01]")
    base = guard_edge_case("gate_cache")
    model = replace(
        base.source.model((0,)),
        rates_per_ns=np.array([[0.0, gamma], [0.0, 0.0]]),
        dephasing_per_ns=np.array([[0.0, gamma / 2], [gamma / 2, 0.0]]),
    )
    base.source.close()
    duration = float(np.cbrt(16 * split / (gamma * v**2)))
    batch = timeline(2, duration)
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.8, 1.0, 0.8], complex),
        np.array([0.0, duration / 2, duration]),
    )
    return AccuracyCase(
        f"guard_edge/sweep/gate_cache/gamma={gamma:.17g}/v={v:.17g}/split={split:.17g}",
        coherent_source(model),
        replace(
            batch, tracks=(drive_track(batch, envelope, frequency=0.6, gain=v / 0.002),)
        ),
        ("gate_cache",),
    )


def gate_shape_bridge_case(*, shaped: bool) -> AccuracyCase:
    """Compare unity and [.8,1,.8] envelopes at gamma=1/ns and d=8 ns.

    Both use the same peak V, one float below sqrt(.01*16/8**3), so the actual
    split is just inside .01 rather than rounded outside. Only envelope samples
    differ. This is a new comparison, not a replacement of the existing edge.
    """
    case = guard_edge_case("gate_cache")
    v = float(np.nextafter(np.sqrt(0.01 * 16 / 8**3), 0.0))
    track = case.batch.tracks[0]
    envelope = replace(
        track.envelopes[0],
        samples=np.array([0.8, 1.0, 0.8] if shaped else [1.0, 1.0, 1.0], complex),
    )
    track = replace(track, amplitude=np.full(2, v / 0.002), envelopes=(envelope,))
    return replace(
        case,
        name=f"guard_edge/sweep/gate_cache/bridge/shaped={shaped}",
        batch=replace(case.batch, tracks=(track,)),
    )
