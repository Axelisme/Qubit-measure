"""FTF offline construction from the same YAML and numeric point composition."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.resources.ftf import load_ftf
from zcu_tools.simulate.evolution import (
    BuildPlan,
    EvolutionLibrary,
    IntelOpenCL,
    PlateauFamily,
    ValidationLimits,
)
from zcu_tools.simulate.evolution.library import BuildResult

from .point import plan_point


@dataclass(frozen=True)
class FTFBuildPlan:
    """Optional fixed-DC plateau prebuild schedule, not another physical model.

    duration_ns: positive nearest-2-ns plateau length. ratios: signed complex
    microwave amplitudes in YAML microwave_sources order (flux drives are zero).
    lower/upper: ordered dimensionless common-scale range. degree: 4 or 6.
    limits: absolute publication tolerances. Runtime pulses still come from the
    actual timeline; unrepresented duration/envelope/phase/flux requests go direct.
    """

    duration_ns: float
    ratios: tuple[complex, ...]
    lower: float
    upper: float
    degree: int = 6
    limits: ValidationLimits = field(default_factory=ValidationLimits)


def build_database(
    configuration: Path,
    path: Path,
    plan: FTFBuildPlan,
    compiler_directory: Path,
    *,
    node_limit: int | None = None,
) -> BuildResult:
    """Read FTF YAML, build joint GPU nodes, validate and save to explicit path.

    Uses the same load_ftf/plan_point composition as runtime. DC is YAML startup
    device values. Does not create drivers, GUI, QICK programs or hardware. Bath
    values do not enter U but remain in runtime direct evolution. Paths must be
    explicit, with library path absolute; node_limit permits resumable partial
    work. Intel runtime, invalid input, solver/validation/I/O errors propagate.
    """
    settings = load_ftf(configuration)
    if len(plan.ratios) != len(settings.microwave_sources):
        raise ValueError("prebuild ratios must match YAML microwave_sources")
    if not np.isfinite(plan.ratios).all():
        raise ValueError("prebuild ratios must be finite")
    end_us = plan.duration_ns / 1000
    events = tuple(
        PulseEvent(
            source.gen_ch,
            0.0,
            end_us,
            0.0,
            float(np.rad2deg(np.angle(ratio))),
            float(abs(ratio)),
        )
        for source, ratio in zip(settings.microwave_sources, plan.ratios, strict=True)
    )
    readout = PulseEvent(settings.readout_source.gen_ch, end_us, 0.002, 0.0, 0.0, 0.0)
    window = ReadoutWindow(
        settings.receiver.ro_ch,
        settings.readout_source.gen_ch,
        1,
        1000.0,
        (end_us,),
        (0.0,),
        (0.0,),
        (settings.readout_source.gen_ch,),
    )
    snapshot = EnvironmentSnapshot(
        "ftf-v1", "offline", tuple((d.name, d.initial) for d in settings.devices)
    )
    point = plan_point(
        settings,
        snapshot,
        PointTimeline((), events + (readout,), (window,), end_us + 0.002),
    )
    ratios = tuple(0j for _ in settings.flux_sources) + plan.ratios
    family = PlateauFamily(point.evolution_model, plan.duration_ns, ratios)
    numeric = BuildPlan(family, plan.lower, plan.upper, plan.degree, plan.limits)
    return EvolutionLibrary(path).build(
        numeric, IntelOpenCL(compiler_directory), node_limit=node_limit
    )


def main(argv: list[str] | None = None) -> int:
    """Thin offline CLI: YAML/path and an explicit optional prebuild schedule.

    --ratios supplies one Python complex literal per YAML microwave source.
    Prints progress/result only; no generated script or runtime table expansion.
    Returns 0 on partial/success, lets invalid input/runtime/I/O errors propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("path", type=Path)
    parser.add_argument("--duration-ns", required=True, type=float)
    parser.add_argument("--ratios", required=True, nargs="+", type=complex)
    parser.add_argument("--lower", required=True, type=float)
    parser.add_argument("--upper", required=True, type=float)
    parser.add_argument("--degree", choices=(4, 6), type=int, default=6)
    parser.add_argument("--compiler-directory", required=True, type=Path)
    parser.add_argument("--node-limit", type=int)
    args = parser.parse_args(argv)
    plan = FTFBuildPlan(
        args.duration_ns, tuple(args.ratios), args.lower, args.upper, args.degree
    )
    result = build_database(
        args.configuration,
        args.path,
        plan,
        args.compiler_directory,
        node_limit=args.node_limit,
    )
    print(
        f"nodes={result.completed_nodes} new={result.computed_nodes} published={result.published}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
