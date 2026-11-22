"""Validate the curated wall-time scene manifest and resolve selectors."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated

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
_Field = Annotated[StrictStr, StringConstraints(min_length=1)]


class _ManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Override(_ManifestItem):
    """One bounded Form write and its rationale."""

    path: Annotated[tuple[_Field | _Index, ...], Field(min_length=1)]
    value: JsonValue
    basis: _Text


class Scene(_ManifestItem):
    """A scene's stable identity, live adapter and authored work."""

    scene_id: _Key
    adapter: _Key
    basis: _Text
    overrides: list[Override] = Field(default_factory=list)
    ftf_exclusion: _Text | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Scene:
        paths = set()
        for override in self.overrides:
            if override.path in paths:
                raise ValueError(
                    f"Duplicate override for {self.scene_id!r} at {override.path!r}"
                )
            paths.add(override.path)
        return self


class _Unpinned(_ManifestItem):
    form: _Text
    point: _Text


class Manifest(_ManifestItem):
    """Shared Working Point and hand-selected wall-time workloads."""

    entry: _Label
    point: _Label
    unpinned: _Unpinned
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
