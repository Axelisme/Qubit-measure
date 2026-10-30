"""Run an experiment offline, persist its record and export headless figures."""

import argparse
import importlib
import json
import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from reprlib import repr as short_repr
from time import perf_counter
from typing import Any, TypeVar
from urllib.parse import quote

import numpy as np
from pydantic import TypeAdapter

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import load
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.yaml import load_yaml

from . import native_persistence
from .acquisition import (
    capture_device_snapshot,
    capture_software_provenance,
    new_run_id,
)
from .base import Experiment
from .context import RunContext
from .metadata import RunMetadata, RunSnapshot
from .records import RunRecord
from .scope import Scope
from .simulation import SimulationEnvironment, white_noise
from .single_fluxonium import single_fluxonium

_Cfg = TypeVar("_Cfg", bound=ConfigBase)
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class ResultField:
    """One Result field's NumPy shape and dtype name, in memory units."""

    shape: tuple[int, ...]
    dtype: str


@dataclass
class ProbeReport:
    """Observations from one offline probe, including partial failure evidence.

    environment is the supplied environment's public description. native_path is
    the intended native destination; a prior probe's file may exist after failure.
    fields/restored_fields map Result dataclass field names to in-memory array
    descriptors, before/after native restore. roundtrip_equal is exact Result
    shape/dtype/value equality with NaNs equal, or None before comparison.
    analysis_summary is a
    bounded repr of analyze's return, or None before it returns. png_paths lists
    successfully written live/analysis PNGs. timings maps attempted stages to
    elapsed wall seconds. Stages are setup, run, live_png, save, restore, analyze
    and analysis_png. failed_stage and exception are both None on success;
    on failure they retain the stage name and original exception object.
    """

    environment: str
    native_path: Path
    fields: dict[str, ResultField] = field(default_factory=dict)
    restored_fields: dict[str, ResultField] = field(default_factory=dict)
    roundtrip_equal: bool | None = None
    analysis_summary: str | None = None
    png_paths: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    failed_stage: str | None = None
    exception: Exception | None = None


@contextmanager
def _timed(report: ProbeReport, stage: str) -> Generator[None]:
    started = perf_counter()
    try:
        yield
    except Exception:
        report.failed_stage = stage
        raise
    finally:
        report.timings[stage] = perf_counter() - started


def _describe(result: Any) -> dict[str, ResultField]:
    return {
        item.name: ResultField(
            np.asarray(getattr(result, item.name)).shape,
            str(np.asarray(getattr(result, item.name)).dtype),
        )
        for item in fields(result)
    }


def _equal(original: Any, restored: Any) -> bool:
    for item in fields(original):
        before = np.asarray(getattr(original, item.name))
        after = np.asarray(getattr(restored, item.name))
        if before.dtype != after.dtype or not np.array_equal(
            before, after, equal_nan=before.dtype.kind in "fc"
        ):
            return False
    return True


def _acquire(
    experiment: Experiment[_Result, _Cfg],
    cfg: _Cfg,
    environment: SimulationEnvironment,
    plots: Plots,
) -> RunRecord[_Cfg, _Result]:
    with Scope("run") as scope:
        context = RunContext(
            environment.soc,
            environment.soccfg,
            plots,
            environment.devices.get_all_devices(),
            scope,
        )
        provenance = capture_software_provenance(context=context)
        devices = capture_device_snapshot(context.devices)
        started = datetime.now(timezone.utc)
        result = experiment.run(cfg.model_copy(deep=True), context=context)
        scope.stop.raise_if_error()
        finished = datetime.now(timezone.utc)
        spec = experiment.AXES_SPEC
        if spec is None:
            raise ValueError("Probe requires experiment.AXES_SPEC")
        return RunRecord(
            cfg,
            result,
            RunMetadata(
                run_id=new_run_id(started),
                experiment=spec.tag,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                completion="stopped" if scope.cancelled else "complete",
                snapshot=RunSnapshot(roles={}),
                provenance=provenance,
                device_snapshot=devices,
            ),
        )


def _export(plots: Plots, stage: str, report: ProbeReport, directory: Path) -> None:
    plots.finish()
    for name in plots:
        path = directory / f"{stage}-{quote(name, safe='')}.png"
        plots.save(name, path)
        report.png_paths.append(path)


def probe(
    experiment: Experiment[_Result, _Cfg],
    cfg: _Cfg,
    environment: SimulationEnvironment,
    output_dir: Path,
    *,
    options: Any = None,
) -> ProbeReport:
    """Run, save native, restore and analyze one supplied offline experiment.

    cfg must match the experiment's Cfg type. Run and analyze receive isolated
    copies of cfg and options; options defaults to None. AXES_SPEC is required.
    Use caller-created SimulationEnvironment resources; never reseed, rewire or
    close them here.
    Create output_dir and overwrite data.h5 and same-named PNGs on rerun. Figures
    use Agg canvases and no QApplication. PNG names contain live/analysis and
    URL-escaped figure names. Return shapes, exact Result round-trip comparison,
    analysis repr and timings. Stage failures are logged with traceback and
    returned as failed_stage/exception without replacing the original exception.
    Missing AXES_SPEC raises ValueError before starting. Caller owns environment
    cleanup, including when a stage fails.
    """
    spec = experiment.AXES_SPEC
    if spec is None:
        raise ValueError("Probe requires experiment.AXES_SPEC")
    report = ProbeReport(environment.description, output_dir / "data.h5")
    live = Plots(NonPresentingHost())
    analysis = Plots(NonPresentingHost())
    try:
        with _timed(report, "setup"):
            output_dir.mkdir(parents=True, exist_ok=True)
        with _timed(report, "run"):
            record = _acquire(experiment, cfg, environment, live)
            report.fields = _describe(record.result)
        with _timed(report, "live_png"):
            _export(live, "live", report, output_dir)
        with _timed(report, "save"):
            native_persistence.save_run(
                record, report.native_path, spec=spec, replace=True
            )
        with _timed(report, "restore"):
            restored = native_persistence.restore_run(
                load(report.native_path), spec=spec
            ).record
            report.restored_fields = _describe(restored.result)
            report.roundtrip_equal = _equal(record.result, restored.result)
        with _timed(report, "analyze"):
            report.analysis_summary = short_repr(
                experiment.analyze(restored, deepcopy(options), plots=analysis)
            )
        with _timed(report, "analysis_png"):
            _export(analysis, "analysis", report, output_dir)
    except Exception as error:
        logging.getLogger(__name__).exception("Probe failed at %s", report.failed_stage)
        report.exception = error
    finally:
        for plots in (live, analysis):
            plots.finish(present=False)
            plots.release()
    return report


def _route(value: str) -> tuple[int, int]:
    try:
        gen, adc = value.split(":")
        return int(gen), int(adc)
    except ValueError as error:
        raise argparse.ArgumentTypeError("route must be GEN:ADC integers") from error


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Exception):
        return {"type": type(value).__qualname__, "message": str(value)}
    raise TypeError(f"Cannot encode {type(value).__qualname__} in probe report")


def main(argv: Sequence[str] | None = None) -> ProbeReport:
    """Probe a trusted module:class and print its report as JSON to stdout.

    The class must be a zero-argument Experiment with AXES_SPEC. Required flags:
    --cfg YAML_OR_JSON, --environment white_noise|single_fluxonium and
    --output-dir DIRECTORY. --options YAML_OR_JSON is validated by the class's
    Options type when declared, otherwise passed as decoded data. Omitted options
    construct Options() when declared, otherwise pass None.

    White-noise flags are --std-adc, --seed, --n-gens, --n-readouts and repeatable
    --route GEN:ADC. Single-fluxonium requires --environment-config YAML.
    This entry owns and closes its created environment, but creates no GUI.
    Return the same ProbeReport as probe, including stage failures. Argparse
    errors raise SystemExit; import, cfg/options validation and environment
    startup/cleanup errors propagate. Only the python -m shell sets exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="trusted importable module:class")
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument(
        "--environment", choices=("white_noise", "single_fluxonium"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--options", type=Path)
    parser.add_argument("--environment-config", type=Path)
    parser.add_argument("--std-adc", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-gens", type=int, default=2)
    parser.add_argument("--n-readouts", type=int, default=1)
    parser.add_argument("--route", type=_route, action="append")
    args = parser.parse_args(argv)
    if args.environment == "single_fluxonium" and args.environment_config is None:
        parser.error("single_fluxonium requires --environment-config")
    module, separator, name = args.experiment.partition(":")
    if not separator:
        parser.error("experiment must be module:class")
    experiment_type = getattr(importlib.import_module(module), name)
    if not isinstance(experiment_type, type) or not issubclass(
        experiment_type, Experiment
    ):
        raise TypeError("module:class must name an Experiment subclass")
    experiment = experiment_type()
    spec = experiment.AXES_SPEC
    if spec is None:
        raise ValueError("Probe requires experiment.AXES_SPEC")
    cfg = spec.cfg_type.model_validate(load_yaml(args.cfg))
    options_type = getattr(experiment_type, "Options", None)
    if args.options is not None:
        decoded = load_yaml(args.options)
        options = (
            decoded
            if options_type is None
            else TypeAdapter(options_type).validate_python(decoded)
        )
    else:
        options = None if options_type is None else options_type()
    environment = (
        white_noise(
            std_adc=args.std_adc,
            seed=args.seed,
            n_gens=args.n_gens,
            n_readouts=args.n_readouts,
            routes=None if args.route is None else tuple(args.route),
        )
        if args.environment == "white_noise"
        else single_fluxonium(args.environment_config)
    )
    try:
        report = probe(experiment, cfg, environment, args.output_dir, options=options)
    finally:
        try:
            if environment.close_backend is not None:
                environment.close_backend()
        finally:
            environment.devices.close_all_devices()
    print(json.dumps(report, default=_json_default, indent=2))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main().exception is None else 1)
