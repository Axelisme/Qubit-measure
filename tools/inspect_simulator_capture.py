"""Read recorded simulator routing and same-acquire telemetry without solving."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, StrictBool, StrictFloat, TypeAdapter
from zcu_tools.simulate.evolution import (
    CaptureMetadata,
    capture_directory,
    read_capture_metadata,
    read_frozen_manifest,
)


class _Timing(BaseModel):
    """Matched acquire's measured seconds and normal-return flag, all required."""

    complete: StrictBool
    wall_s: StrictFloat
    compile_wall_s: StrictFloat
    evolve_wall_s: StrictFloat
    readout_wall_s: StrictFloat


def _matching_timing(directory: Path, filename: str) -> dict[str, float | bool] | None:
    matches: list[object] = []
    for path in sorted(directory.glob("routing-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("capture_file") == filename:
                    matches.append(row)
    if len(matches) > 1:
        raise ValueError(f"multiple telemetry rows for {filename}")
    if not matches:
        return None
    timing = _Timing.model_validate(matches[0])
    return {
        "complete": timing.complete,
        "wall_s": timing.wall_s,
        "compile_wall_s": timing.compile_wall_s,
        "evolve_wall_s": timing.evolve_wall_s,
        "readout_wall_s": timing.readout_wall_s,
        "remainder_s": timing.wall_s
        - timing.compile_wall_s
        - timing.evolve_wall_s
        - timing.readout_wall_s,
    }


def _prefix_adoptions(
    metadata: CaptureMetadata, siblings: Sequence[CaptureMetadata]
) -> list[dict[str, object]]:
    adoptions: list[dict[str, object]] = []
    for representative in siblings:
        if representative["capture_id"] != metadata["capture_id"]:
            continue
        for index, run in enumerate(representative["prefix_runs"], 1):
            for adopter in run["adopters"]:
                if adopter["batch"] == metadata["batch_index"]:
                    adoptions.append(
                        {
                            "point": adopter["point"],
                            "segment": adopter["segment"],
                            "representative_batch": representative["batch_index"],
                            "representative": run["representative"],
                            "prefix_run": index,
                            "status": f"沿用第{index}個prefix run",
                        }
                    )
    return adoptions


def _batch_report(
    metadata: CaptureMetadata,
    siblings: Sequence[CaptureMetadata],
    conditions: dict[str, object],
) -> dict[str, object]:
    resolved = {
        (route["prefix_run"], route["point"], route["segment"])
        for route in metadata["routes"]
        if route["round"] == "worker"
    }
    pending = [
        {
            "point": route["point"],
            "segment": route["segment"],
            "prefix_run": route["prefix_run"],
        }
        for route in metadata["routes"]
        if route["round"] == "caller"
        and route["family"] is None
        and (route["prefix_run"], route["point"], route["segment"]) not in resolved
    ]
    recorded: dict[str, object] = dict(metadata)
    differences = [key for key, value in conditions.items() if value != recorded[key]]
    return {
        **metadata,
        "condition_differences": differences,
        "evaluation_status": "新條件下重新評估" if differences else "錄製條件未變",
        "routes_status": "下列仍是錄製時選路，未重算",
        "prefix_adoptions": _prefix_adoptions(metadata, siblings),
        "pending": pending,
        "pending_status": "停在deferred" if pending else None,
        "reason": "未知",
        "error": "未提供",
    }


def _parse_families(value: str) -> list[str] | None:
    names = TypeAdapter(list[str] | None).validate_json(value, strict=True)
    return None if names is None else sorted(set(names))


def main(argv: Sequence[str] | None = None) -> int:
    """Print JSON from a capture file/directory and routing JSONL directory.

    Omitted path uses capture_directory(); directories select only immediate
    capture-*.json.gz files in filename order. Telemetry defaults to this tool's
    worktree-root .sim_telemetry, independent of cwd. Exact capture_file basename
    matches supply acquire times; missing/null keys do not match. Unreadable or
    invalid input raises, including duplicate matches naming the manifest.
    Optional budget/families/source/version flags compare only metadata. Families
    is a JSON string array or null for unrestricted. Omitted flags retain each
    recorded condition. Differences label new-condition reevaluation; displayed
    routes remain the original observations, never recalculated. No files,
    models, guards or routing state are changed.
    Return zero on success; argparse reports invalid command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument(
        "--telemetry-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".sim_telemetry",
    )
    parser.add_argument("--budget", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--families", type=_parse_families, default=argparse.SUPPRESS)
    parser.add_argument("--source-identity", default=argparse.SUPPRESS)
    parser.add_argument("--segment-numeric-version", default=argparse.SUPPRESS)
    parser.add_argument(
        "--frozen-physical-version", type=int, default=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    conditions = {
        key: value
        for key, value in vars(args).items()
        if key not in {"path", "telemetry_directory"}
    }
    path = capture_directory() if args.path is None else args.path
    files = sorted(path.glob("capture-*.json.gz")) if path.is_dir() else [path]
    reports: list[dict[str, object]] = []
    for capture in files:
        records = read_frozen_manifest(capture)
        metadata = [read_capture_metadata(record) for record in records]
        report: dict[str, object] = {
            "file": capture.name,
            "status": "錄製實錄，未重算" if records else "沒有evolve被錄到",
            "batches": [_batch_report(item, metadata, conditions) for item in metadata],
        }
        timing = _matching_timing(args.telemetry_directory, capture.name)
        if timing is None:
            report["telemetry_status"] = "未找到同次 telemetry 行"
        else:
            report["timing"] = timing
        reports.append(report)
    print(json.dumps({"captures": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
