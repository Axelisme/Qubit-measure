"""Parameterized synchronous two-tone research segments, without lab I/O."""

from contextlib import ExitStack, closing
from dataclasses import asdict
from decimal import Decimal
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.program import build_soccfg
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_scene_builders import numeric_builder
from tools.family_pareto_settings import (
    BOARD_PROFILE,
    INTEGER_MODEL,
    SINGLE_MODEL,
    SOURCE_COMMIT,
)


class Dop853Coordinates(BaseModel):
    """Coordinates for simultaneous sampled tones, independent of Engine workers.

    envelope selects Gaussian or deterministic non-Gaussian samples. duration_ns
    is each segment's support in ns; knots includes both endpoints per envelope.
    levels is the retained model dimension (3..native_cutoff). repeats joins
    identical segments on one timeline, not independent rows. points creates
    independent timelines with Pump gains from .8 to 1.2 times .043 (one point
    uses .043). Change one coordinate at a time for reduced-axis comparisons.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    envelope: Literal["gaussian", "sampled"] = "gaussian"
    duration_ns: float = Field(default=20.0, gt=0, allow_inf_nan=False)
    knots: int = Field(default=9, ge=3, strict=True)
    levels: int = Field(default=8, ge=3, strict=True)
    repeats: Literal[1, 2, 20, 100] = 1
    points: int = Field(default=1, ge=1, strict=True)


_PROFILE_COORDINATES = (
    Dop853Coordinates(envelope="gaussian"),
    Dop853Coordinates(envelope="gaussian", duration_ns=40.0),
    Dop853Coordinates(envelope="gaussian", levels=20),
    Dop853Coordinates(envelope="gaussian", knots=17),
    Dop853Coordinates(envelope="gaussian", repeats=2),
    Dop853Coordinates(envelope="gaussian", repeats=20),
    Dop853Coordinates(envelope="gaussian", repeats=100),
    Dop853Coordinates(envelope="sampled"),
    Dop853Coordinates(envelope="sampled", duration_ns=40.0),
    Dop853Coordinates(envelope="sampled", levels=20),
    Dop853Coordinates(envelope="sampled", knots=17),
    Dop853Coordinates(envelope="sampled", repeats=2),
    Dop853Coordinates(envelope="sampled", repeats=20),
    Dop853Coordinates(envelope="sampled", repeats=100),
)
# This workers candidate is not wall-time calibrated or reference-prepared.
_WORKERS_CANDIDATE = Dop853Coordinates(envelope="gaussian", points=64)


def profile_inputs(scenario: str | None = None) -> list[Input]:
    """Build the declared reduced-axis segments and one uncalibrated workers input.

    Return Gaussian then sampled inputs, each with the baseline d20/l8/k9/r1/p1
    and single-axis duration 40 ns, levels 20, knots 17, or repeats 2/20/100.
    Append a Gaussian baseline with 64 independent points as a workers candidate.
    scenario=None builds the whole catalogue. A coordinate-derived scenario name
    selects only that Input before any model construction. An unknown name raises
    ValueError listing legal profile names without constructing models.
    This does not time, evolve, solve references, freeze scenes or register them
    in the default family-table inventory. Model construction errors propagate.
    Each returned Input.build() creates a source that its caller must close.
    """
    coordinates: tuple[Dop853Coordinates, ...] = (
        *_PROFILE_COORDINATES,
        _WORKERS_CANDIDATE,
    )
    if scenario is not None:
        selected = tuple(
            item for item in coordinates if _segment_name(item) == scenario
        )
        if not selected:
            raise ValueError(
                f"Unknown scenario: {scenario}. Legal profile names: "
                f"{', '.join(_segment_name(item) for item in coordinates)}"
            )
        coordinates = selected
    return [dop853_segment_input(item) for item in coordinates]


def _segment_name(coordinates: Dop853Coordinates) -> str:
    """Derive the Input/Scene name without constructing a model."""
    duration = format(Decimal(str(coordinates.duration_ns)).normalize(), "f")
    return (
        f"dop853_segment-{coordinates.envelope}-d{duration.replace('.', 'p')}"
        f"-l{coordinates.levels}-k{coordinates.knots}"
        f"-r{coordinates.repeats}-p{coordinates.points}"
    )


def dop853_segment_input(coordinates: Dop853Coordinates) -> Input:
    """Build a tool-owned two-tone segment without readout or idle companions.

    Coordinates select sampled envelopes, support, dimension and repetitions.
    Settings are validated before the public single-fluxonium owner rebuilds
    the model and Gibbs initial state. levels above native_cutoff raise ValueError.
    Input and Scene share a coordinate-derived name containing envelope, duration
    in ns, levels, knots, repeats and points. Integer durations omit the decimal
    part; fractional durations replace the decimal point with "p".
    Every Input.build() returns a fresh owned source, which callers must close.
    Pin Engine(workers=N, families=("dop853",)); guards remain enabled. Same-dimension
    reference agreement does not establish retained-level convergence.
    """
    settings = parse_single_fluxonium(
        {
            **{
                key: value for key, value in SINGLE_MODEL.items() if key != "drive_line"
            },
            "drive_lines": INTEGER_MODEL["drive_lines"],
            "flux_line": None,
        }
    )
    settings = settings.model_copy(
        update={
            "dc_bias": settings.dc_bias.model_copy(update={"offset_phi0": 0.45}),
            "drift": None,
            "numerics": settings.numerics.model_copy(
                update={
                    "time_scale": 0.0,
                    "truncation": settings.numerics.truncation.model_copy(
                        update={"levels": coordinates.levels}
                    ),
                }
            ),
        }
    )
    # model_copy does not validate nested updates; validate before any modeling.
    settings = type(settings).model_validate(settings.model_dump())
    env = single_fluxonium(settings, soccfg=build_soccfg(BOARD_PROFILE))
    name = _segment_name(coordinates)
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(SingleFluxoniumSource(settings, env.devices))
        )
        snapshot = owner.snapshot()
        numeric = stack.enter_context(closing(owner.model_facts(snapshot).source))
        batch = _dop853_batch(coordinates, numeric.model((0,)))
        build = numeric_builder(Scene(name, numeric, batch, ("dop853",)))
    return Input(
        name,
        build,
        "Synchronous shaped two-tone segments without readout or idle tail",
        category="real",
        expected_routes={
            "dop853": {
                "dop853": (
                    coordinates.points * coordinates.repeats,
                    float(np.sum(batch.end_ns)),
                )
            }
        },
        manifest={
            "scenario_version": 1,
            "coordinates": coordinates.model_dump(mode="json"),
            "effective_settings": settings.model_dump(mode="json"),
            "snapshot": asdict(snapshot),
            "source_commit": SOURCE_COMMIT,
            "pulse_sequence": "Two simultaneous tones, repeated on each timeline",
            "carrier_method": "DC E1-E0 and E2-E1, GHz, no hardware quantization",
            "gain_method": "Pump .043, Stokes .143; uncalibrated benchmark controls",
            "truncation_limit": "Same-dimension oracle agreement does not establish retained-level convergence",
        },
    )


def _dop853_batch(coordinates: Dop853Coordinates, model: SecularModel) -> TimelineBatch:
    times = np.linspace(0.0, coordinates.duration_ns, coordinates.knots)
    local = times / coordinates.duration_ns
    gains = (
        np.ones(1)
        if coordinates.points == 1
        else np.linspace(0.8, 1.2, coordinates.points)
    )
    tracks = []
    for repeat in range(coordinates.repeats):
        start = repeat * coordinates.duration_ns
        for tone, (port, gain, center) in enumerate(
            (("Pump", 0.043, 0.65), ("Stokes", 0.143, 0.35))
        ):
            samples = (
                np.exp(-0.5 * ((local - center) / (1 / 6)) ** 2)
                if coordinates.envelope == "gaussian"
                else np.sin(np.pi * local) ** 2
                * (0.7 + 0.3 * np.cos((tone + 2) * np.pi * local + 0.2))
            )
            envelope = EnvelopeSpec("sampled", samples.astype(np.complex128), times)
            frequency = float(model.energies_ghz[tone + 1] - model.energies_ghz[tone])
            tracks.append(
                DriveTrack(
                    port=port,
                    start_ns=np.full(coordinates.points, start),
                    duration_ns=np.full(coordinates.points, coordinates.duration_ns),
                    frequency_ghz=np.full(coordinates.points, frequency),
                    phase_rad=np.full(
                        coordinates.points, -2 * np.pi * frequency * start
                    ),
                    amplitude=gain
                    * (gains if tone == 0 else np.ones(coordinates.points)),
                    envelopes=(envelope,),
                    envelope_index=np.zeros(coordinates.points, dtype=np.int64),
                    active=np.ones(coordinates.points, dtype=np.bool_),
                )
            )
    return TimelineBatch(
        points=np.arange(coordinates.points, dtype=np.int64),
        tracks=tuple(tracks),
        flux=FluxSchedule(
            offsets=np.arange(coordinates.points + 1, dtype=np.int64),
            start_ns=np.zeros(coordinates.points),
            level=np.zeros((coordinates.points, 1), dtype=np.int64),
        ),
        end_ns=np.full(
            coordinates.points, coordinates.repeats * coordinates.duration_ns
        ),
    )
