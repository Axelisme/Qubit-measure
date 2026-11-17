"""Importable finite sweeps of the four inaccurate guard-edge inputs.

These builders vary physical inputs only, never production guards. Gain scale
is a public admission-axis coordinate for common_frame and shaped_sambe, not
an observation of their private coherent/convergence indicators. No linear or
globally monotone relation between that coordinate and density error is assumed.
"""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec

from .accuracy_cases import (
    AccuracyCase,
    coherent_source,
    drive_track,
    linear_drive_model,
    timeline,
)
from .guard_edge_cases import guard_edge_case

# Scale=1 is covered by the guard-edge matrix for common_frame/shaped_sambe.
# Single-tone sweeps retain the original phase-edge input, not the kappa2 edge.
SCALED_SAMPLES = (
    ("single_tone_rwa", 0.05, None),
    ("single_tone_rwa", 0.1, None),
    ("single_tone_rwa", 0.2, None),
    ("single_tone_rwa", 0.3, None),
    ("single_tone_rwa", 0.4, None),
    ("single_tone_rwa", 0.5, None),
    ("single_tone_rwa", 0.6, None),
    ("single_tone_rwa", 0.7, None),
    ("single_tone_rwa", 0.8, None),
    ("single_tone_rwa", 0.9, None),
    ("common_frame", 0.05, None),
    ("common_frame", 0.1, None),
    ("common_frame", 0.2, None),
    ("common_frame", 0.3, None),
    ("common_frame", 0.4, None),
    ("common_frame", 0.5, None),
    ("common_frame", 0.6, None),
    ("common_frame", 0.7, None),
    ("common_frame", 0.8, None),
    ("common_frame", 0.9, None),
    ("shaped_sambe", 0.05, 0.006976190290283084),
    ("shaped_sambe", 0.1, 0.013939786732875038),
    ("shaped_sambe", 0.2, 0.009994325894870235),
    ("shaped_sambe", 0.3, 0.003890045479847463),
    ("shaped_sambe", 0.4, 0.005218876551386592),
    ("shaped_sambe", 0.5, 0.00522067773534268),
    ("shaped_sambe", 0.6, 0.0043120889572121545),
    ("shaped_sambe", 0.7, 0.005127072522687026),
    ("shaped_sambe", 0.8, 0.0030436859867629283),
    ("shaped_sambe", 0.9, 0.0027439229863831147),
    ("single_tone_rwa", 0.048593750000000005, None),
    ("single_tone_rwa", 0.04859765625, None),
    ("common_frame", 0.5242578124999999, None),
    ("common_frame", 0.524296875, None),
    ("shaped_sambe", 0.01431640625, None),
    ("shaped_sambe", 0.0143359375, 0.002000606697780786),
)
GATE_SAMPLES = (
    (1, 0.018, 0.0001, None),
    (1, 0.018, 0.00025, 0.002172180191409219),
    (1, 0.018, 0.0005, 0.005515447677048631),
    (1, 0.018, 0.001, 0.005880947950789922),
    (1, 0.018, 0.002, 0.0020873329106250948),
    (1, 0.018, 0.003, 0.002917304387132671),
    (1, 0.018, 0.004, 0.007307560557721029),
    (1, 0.018, 0.005, 0.0080745542318555),
    (1, 0.018, 0.006, 0.009301157648956159),
    (1, 0.018, 0.007, 0.008645463048205817),
    (1, 0.018, 0.008, 0.009450772257367157),
    (1, 0.018, 0.009, 0.008587737685302781),
    (1, 0.018, 0.00999999, 0.006692074814160416),
    (0.01, 0.018, 0.0001, None),
    (0.01, 0.018, 0.00025, None),
    (0.01, 0.018, 0.0005, None),
    (0.01, 0.018, 0.001, None),
    (0.01, 0.018, 0.002, None),
    (0.01, 0.018, 0.003, 0.0022881885442749953),
    (0.01, 0.018, 0.004, 0.003015382710179194),
    (0.01, 0.018, 0.005, 0.0039805202839394436),
    (0.01, 0.018, 0.006, 0.004371803365788659),
    (0.01, 0.018, 0.007, 0.005033371893005003),
    (0.01, 0.018, 0.008, 0.005335892195141911),
    (0.01, 0.018, 0.009, 0.004983074897931769),
    (0.01, 0.018, 0.00999999, 0.005381528914108578),
    (0.01, 0.05, 0.00999999, None),
    (0.01, 0.1, 0.00999999, None),
    (1, 0.018, 0.00018984375000000002, None),
    (1, 0.018, 0.00019023437500000003, 0.002002034486524994),
    (0.01, 0.018, 0.002271875, None),
    (0.01, 0.018, 0.0022720703125, 0.002000054029053949),
    (0.01, 0.018, 0.002356640625, 0.0020010038115975412),
    (0.01, 0.018, 0.0023568359374999998, None),
    (0.01, 0.018, 0.0024378906250000004, None),
    (0.01, 0.018, 0.0024380859375000004, 0.002000723587033869),
    (0.01, 0.018, 0.0027375, 0.002000106363492022),
    (0.01, 0.018, 0.0027376953125, None),
    (0.01, 0.018, 0.0027603515624999996, None),
    (0.01, 0.018, 0.0027605468749999995, 0.0020001496452692),
)


def scaled_edge_case(family: str, scale: float) -> AccuracyCase:
    """Multiply all gains of a fixed edge input by positive scale <= 1.

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
    if family == "single_tone_rwa":
        # Keep the original 8 ns non-diagonal phase-edge regression independent
        # of the 1 ns diagonal-drive kappa2 edge (ADR-0078).
        batch = timeline(2, 8.0)
        frequency = 0.6
        # Retained positive-frequency X/2 has norm .001 GHz.
        # phase = 2*pi*d*(gain*.001)^2/f = .05999994 rad < .06.
        gain = np.sqrt(0.06 * (1 - 1e-6) * frequency / (2 * np.pi * 8)) / 0.001
        case = AccuracyCase(
            "guard_edge/" + family,
            coherent_source(linear_drive_model()),
            replace(
                batch,
                tracks=(
                    drive_track(
                        batch, EnvelopeSpec("constant"), frequency=frequency, gain=gain
                    ),
                ),
            ),
            (family,),
        )
    else:
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
