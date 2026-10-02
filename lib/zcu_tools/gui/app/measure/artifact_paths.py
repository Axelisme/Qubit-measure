"""Default filenames and batch collision checks for named image artifacts."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from .artifact_tracker import ArtifactKey, ArtifactKind


def named_image_path(base_path: str, key: ArtifactKey) -> str:
    """Build a distinct image filename without interpreting the figure name as a path.

    ASCII lowercase letters, digits, hyphens and underscores remain literal;
    every other UTF-8 byte is encoded as uppercase ``%HH``. This injective
    encoding survives case-insensitive filesystems without lossy sanitization.
    """
    if key.kind is ArtifactKind.DATA:
        raise ValueError("Data artifacts have no image path")
    if not base_path.strip():
        raise ValueError("Image base path must not be empty")
    name = key.figure_name
    if name is None:
        raise ValueError("Image artifacts require a name")
    try:
        raw = name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Figure name must be UTF-8 encodable") from exc
    encoded = "".join(
        chr(byte)
        if (ord("a") <= byte <= ord("z"))
        or (ord("0") <= byte <= ord("9"))
        or byte in (ord("-"), ord("_"))
        else f"%{byte:02X}"
        for byte in raw
    )
    base = Path(base_path)
    stage = "analysis" if key.kind is ArtifactKind.ANALYSIS else "post"
    return str(base.with_name(f"{base.stem}__{stage}__{encoded}{base.suffix}"))


def require_distinct_paths(paths: tuple[str, ...]) -> None:
    """Reject paths that may alias after resolution, Unicode NFC or case folding.

    This is preflight, not an atomic guarantee against external file changes.
    Explicit destinations are checked as given and are never rewritten here.
    """
    if any(not path.strip() for path in paths):
        raise ValueError("Save destination must not be empty")
    folded = [
        unicodedata.normalize("NFC", str(Path(path).resolve())).casefold()
        for path in paths
    ]
    if len(folded) != len(set(folded)):
        raise ValueError("Save destinations must have distinct paths")
