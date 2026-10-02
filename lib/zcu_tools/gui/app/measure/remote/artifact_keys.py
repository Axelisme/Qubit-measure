"""Wire encoding for named save artifacts shared by tab reads and save commands."""

from __future__ import annotations

from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError


def parse_artifact_key(value: object) -> ArtifactKey:
    if value == "data":
        return ArtifactKey(ArtifactKind.DATA)
    if isinstance(value, str) and ":" in value:
        stage, name = value.split(":", 1)
        if stage == "analysis" and name:
            return ArtifactKey(ArtifactKind.ANALYSIS, name)
        if stage == "post" and name:
            return ArtifactKey(ArtifactKind.POST_ANALYSIS, name)
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        "artifact keys must be data, analysis:<name> or post:<name>",
    )


def artifact_key_wire(key: ArtifactKey) -> str:
    if key.kind is ArtifactKind.DATA:
        return "data"
    stage = "analysis" if key.kind is ArtifactKind.ANALYSIS else "post"
    return f"{stage}:{key.figure_name}"
