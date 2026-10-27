"""Whole parameter batches, converted before binding to current-schema views."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument
from zcu_tools.resources.parameter_upgrade import ParameterDocument


@dataclass(frozen=True)
class ParameterValue:
    """One logical dotted field path and its detached working-unit value.

    path starts with general or a component name. value replaces that complete
    field in the raw batch, before the caller transform and current validation.
    """

    path: str
    value: YamlValue


@dataclass(frozen=True)
class PreparedParameterBatch:
    """Validated document and caller diagnostics from one complete conversion.

    document is ready for EditView. notes are opaque operator messages; the
    migration owner attaches the actual destination path when reporting them.
    """

    document: SetupDocument | PointDocument
    notes: tuple[str, ...]


def prepare_parameter_batch(
    document: ParameterDocument,
    *,
    transform: Callable[[ParameterDocument], ParameterDocument] | None,
    values: Sequence[ParameterValue] = (),
) -> PreparedParameterBatch:
    """Convert a detached full seed/value batch and validate the current schema.

    document contains a complete parameter-container mapping, including general,
    components and schema_version. source_path names setup.yaml for a setup;
    other names select a point. transform is the caller-owned pure conversion,
    or None for an already current batch. It must preserve unrelated fields and
    must not perform I/O. Validate only its output, so historical fields need
    not be accepted by current component models. Return a detached typed model
    in PreparedParameterBatch with caller notes. values overlays logical fields before
    conversion, including historical paths absent from the current schema.
    Callback/schema errors propagate before publication;
    neither the input nor registries are changed.
    """
    candidate = deepcopy(document)
    for update in values:
        segments = update.path.split(".")
        if len(segments) < 2 or any(not segment for segment in segments):
            raise ValueError(f"{update.path!r}: expected a logical dotted field")
        if segments[0] != "general":
            segments.insert(0, "components")
        node: YamlMap = candidate.data
        for segment in segments[:-1]:
            branch = node.setdefault(segment, {})
            if not isinstance(branch, dict):
                raise ValueError(f"{update.path}: {segment} is not a mapping")
            node = branch
        node[segments[-1]] = deepcopy(update.value)
    if transform is not None:
        candidate = transform(candidate)
    model = (
        SetupDocument
        if document.source_path is not None
        and document.source_path.name == "setup.yaml"
        else PointDocument
    )
    validated = model.model_validate(deepcopy(candidate.data))
    return PreparedParameterBatch(validated, candidate.notes)
