"""Validate the curated wall-time scene manifest and resolve selectors."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

_Label = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")]
_Key = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$"
    ),
]
_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_Index = Annotated[StrictInt, Field(ge=0)]
_PositiveCount = Annotated[StrictInt, Field(gt=0)]
_Field = Annotated[StrictStr, StringConstraints(min_length=1)]
_FormPath = Annotated[tuple[_Field | _Index, ...], Field(min_length=1)]
_ResultField = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]


class _ManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Override(_ManifestItem):
    """One bounded Form write. The optional basis is for review only."""

    path: _FormPath
    value: JsonValue
    basis: object = None


class ShotsMetric(_ManifestItem):
    """Judge the whole core duration for a Run with shots and no sweep."""

    kind: Literal["shots"] = "shots"


class ResultAxis(_ManifestItem):
    """Named public Result array and the dimension that stores one sweep axis."""

    field: _ResultField
    dimension: _Index = 0


class ScanMetric(_ManifestItem):
    """Judge core seconds per Cartesian point across Result axes, outermost first."""

    kind: Literal["scan"]
    axes: Annotated[list[ResultAxis], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_axes(self) -> ScanMetric:
        if len({(axis.field, axis.dimension) for axis in self.axes}) != len(self.axes):
            raise ValueError("Duplicate Result sweep axis")
        return self


class EvaluationMetric(_ManifestItem):
    """Count evaluations that acquired, not seeds or planned candidates.

    For peak-gated candidates, set peak_field and counts_field. Peaks above
    one are rejected; NaN peaks are unreached. Counts start as NaN, with
    candidates on the first axis. Any written seed counts mark acquisition.
    Otherwise set single_acquire_candidate_field to the Result matrix of entered
    candidate parameters, with NaN rows for candidates not reached. This mode
    requires one main acquire per entered candidate and no rejections or retries.
    """

    kind: Literal["evaluation"]
    peak_field: _ResultField | None = None
    counts_field: _ResultField | None = None
    single_acquire_candidate_field: _ResultField | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> EvaluationMetric:
        peak_gate = self.peak_field is not None and self.counts_field is not None
        single_acquire = (
            self.single_acquire_candidate_field is not None
            and self.peak_field is None
            and self.counts_field is None
        )
        if not (
            single_acquire
            or (peak_gate and self.single_acquire_candidate_field is None)
        ):
            raise ValueError(
                "Declare a peak/count pair or a single-acquire candidate field"
            )
        return self


type SpeedMetric = Annotated[
    ShotsMetric | ScanMetric | EvaluationMetric, Field(discriminator="kind")
]


class OuterAxisReduction(_ManifestItem):
    """The Form override that shortens the outermost axis, and its full length."""

    path: _FormPath
    full_points: _PositiveCount


class RoundsReduction(_ManifestItem):
    """The integer Form override for reduced rounds, and the full round count."""

    path: _FormPath
    full_rounds: _PositiveCount

    def scale(self, override_value: JsonValue) -> float:
        """Return full / reduced rounds; raise for an absent or invalid reduction."""
        if type(override_value) is not int or not 0 < override_value < self.full_rounds:
            raise ValueError(
                "Rounds reduction requires a positive integer Form override below full_rounds"
            )
        return self.full_rounds / override_value


class Extrapolation(_ManifestItem):
    """Declared reductions for projecting a complete measured Run to full work."""

    outer_axis: OuterAxisReduction | None = None
    rounds: RoundsReduction | None = None

    @model_validator(mode="after")
    def require_reduction(self) -> Extrapolation:
        if self.outer_axis is None and self.rounds is None:
            raise ValueError("Extrapolation requires an outer-axis or rounds reduction")
        return self


class Scene(_ManifestItem):
    """A scene's stable identity, live adapter and authored work.

    The metric declares the speed denominator. The optional extrapolation
    references Form overrides that shorten the Run. The optional basis is for
    review only. A nonempty ftf_exclusion skips Run and supplies the reason.
    """

    scene_id: _Key
    adapter: _Key
    basis: object = None
    metric: SpeedMetric
    overrides: list[Override] = Field(default_factory=list)
    extrapolation: Extrapolation | None = None
    ftf_exclusion: _Text | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Scene:
        paths: set[tuple[str | int, ...]] = set()
        for override in self.overrides:
            if override.path in paths:
                raise ValueError(
                    f"Duplicate override for {self.scene_id!r} at {override.path!r}"
                )
            paths.add(override.path)
        for override in self.overrides:
            for length in range(1, len(override.path)):
                parent = override.path[:length]
                if parent in paths:
                    raise ValueError(
                        f"Overlapping override for {self.scene_id!r}: "
                        f"{parent!r} is a parent of {override.path!r}"
                    )
        return self

    @model_validator(mode="after")
    def validate_reductions(self) -> Scene:
        projection = self.extrapolation
        if projection is None:
            return self
        overrides = {item.path: item.value for item in self.overrides}
        outer, rounds = projection.outer_axis, projection.rounds
        if outer is not None and (
            self.metric.kind != "scan" or outer.path not in overrides
        ):
            raise ValueError(
                "Outer-axis reduction requires a scan metric and a matching Form override"
            )
        if rounds is not None:
            rounds.scale(overrides.get(rounds.path))
            if outer is not None and rounds.path == outer.path:
                raise ValueError(
                    "Outer-axis and rounds reductions need distinct Form overrides"
                )
        return self


class Manifest(_ManifestItem):
    """Shared Working Point and workloads. Unpinned notes are for review only."""

    entry: _Label
    point: _Label
    unpinned: object = None
    scenes: Annotated[list[Scene], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_scene_ids(self) -> Manifest:
        seen = set()
        for scene in self.scenes:
            if scene.scene_id in seen:
                raise ValueError(f"Duplicate scene ID: {scene.scene_id!r}")
            seen.add(scene.scene_id)
        return self

    def select(self, selectors: Sequence[str]) -> set[str]:
        """Resolve exact scene IDs before considering an adapter family."""
        scene_ids = {scene.scene_id for scene in self.scenes}
        if not selectors:
            return scene_ids
        selected: set[str] = set()
        seen: set[str] = set()
        for selector in selectors:
            if selector in seen:
                raise ValueError(f"duplicate selector: {selector!r}")
            seen.add(selector)
            matches = (
                {selector}
                if selector in scene_ids
                else {
                    scene.scene_id
                    for scene in self.scenes
                    if scene.adapter.partition("/")[0] == selector
                }
            )
            if not matches:
                raise ValueError(f"unknown scene ID or adapter family: {selector!r}")
            selected.update(matches)
        return selected


def _json_object(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"Duplicate JSON member: {key!r}")
        value[key] = item
    return value


def _json_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite JSON value: {raw}")
    return value


def read_manifest(raw: bytes) -> Manifest:
    """Parse JSON with unique member names and finite numeric values."""
    return Manifest.model_validate(
        json.loads(
            raw,
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_constant=_json_float,
        )
    )


def _raise_io(error: OSError) -> None:
    raise error


@dataclass(frozen=True)
class Sources:
    """The tool-owned inputs used for dispatch and batch identity."""

    directory: Path

    @property
    def manifest(self) -> Path:
        return self.directory / "case-inputs.json"

    @property
    def settings(self) -> Path:
        return self.directory / "ftf-settings.yaml"

    @property
    def board(self) -> Path:
        return self.directory / "board-profile.yaml"

    @property
    def fixture(self) -> Path:
        return self.directory / "fixture"

    def hashes(self, *, manifest_bytes: bytes) -> dict[str, str]:
        """Hash every file consumed from a checkout kept unchanged for the batch."""
        paths = [self.board, self.settings]
        for parent, _, files in self.fixture.walk(on_error=_raise_io):
            paths.extend(parent / name for name in files)
        return {
            self.manifest.name: sha256(manifest_bytes).hexdigest(),
            **{
                path.relative_to(self.directory).as_posix(): sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(paths)
            },
        }
