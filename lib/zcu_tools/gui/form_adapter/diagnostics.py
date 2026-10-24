"""Seed provenance and complete field diagnostics, independent of GUI selection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from zcu_tools.cfg_model import FieldMeta
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
    FormWritePath,
    expr_text,
    input_leaves,
)


@dataclass(frozen=True)
class SeedSource:
    """One Open/Reset seed: source expression/key or None, and its initial status.

    status is missing for absent/failed defaults, available for resolved source
    seeds, and literal for direct defaults. Later edits do not change this fact.
    """

    source: str | None
    status: Literal["missing", "available", "literal"]


@dataclass(frozen=True)
class FieldDiagnostic:
    """One canonical field, with current error and original seed provenance.

    path includes original list indexes. message is None for no current error.
    name/index identify the deepest model-list element, or None outside lists.
    seed_source is an expression, library key, author description, or None.
    seed_status describes Open/Reset, not current validity. uncalibrated is True
    for an authored uncalibrated seed and None when unknown.
    No value or missing field implies a calibration status.
    """

    path: FormWritePath
    message: str | None
    name: str | None
    index: int | None
    seed_source: str | None
    seed_status: Literal["missing", "available", "literal"]
    uncalibrated: bool | None


def capture_seed_sources(
    raw: Mapping[str, JsonValue],
    evaluated: Mapping[str, JsonValue],
    origins: Mapping[FormWritePath, str],
) -> Mapping[FormWritePath, SeedSource]:
    """Record raw seed expressions and their evaluated Open/Reset availability."""
    values = input_leaves(dict(evaluated))
    sources: dict[FormWritePath, SeedSource] = {}
    leaves = input_leaves(dict(raw))
    for path in dict.fromkeys((*leaves, *origins)):
        value = leaves.get(path)
        expression = expr_text(value)
        origin = next(
            (
                origins[path[:size]]
                for size in range(len(path), 0, -1)
                if path[:size] in origins
            ),
            None,
        )
        source = expression if isinstance(expression, str) else origin
        present = any(leaf[: len(path)] == path for leaf in values)
        status: Literal["missing", "available", "literal"] = "literal"
        if value is None or isinstance(expression, str) or path in origins:
            status = "available" if present and value is not None else "missing"
            if path in origins:
                status = "available" if present else "missing"
        elif origin is not None and not origin.startswith("template:"):
            status = "available"
        sources[path] = SeedSource(source, status)
    return sources


def field_diagnostics(
    input: FormInput[FormModel],
    env: FormEnv,
    errors: FormErrors | None,
    sources: Mapping[FormWritePath, SeedSource],
    metadata: Mapping[FormWritePath, FieldMeta],
) -> tuple[FieldDiagnostic, ...]:
    """Enrich supplied owner errors and all current/seed leaves, without validation.

    Uses canonical model-list descriptions for names. No error rules, lowering,
    capture, defaults hooks, mutation or picker state are involved.
    """
    messages = errors.messages if errors is not None else {}
    paths = dict.fromkeys((*sources, *input_leaves(input.data), *messages, *metadata))
    state = FormState(input)
    elements: dict[FormWritePath, dict[int, str]] = {}
    rows: list[FieldDiagnostic] = []
    for path in paths:
        if path not in messages:
            try:
                state.describe(path, env)
            except FormPathError:
                # Seed ancestry can outlive a subsequent selector/shape change.
                continue
        index = None
        name = None
        for position, segment in enumerate(path):
            if not isinstance(segment, int):
                continue
            parent = path[:position]
            if parent not in elements:
                target = state.describe(parent, env).target
                elements[parent] = (
                    {item.index: item.name for item in (target.elements or ())}
                    if target is not None
                    else {}
                )
            index, name = segment, elements[parent].get(segment)
        meta = metadata.get(path)
        seed = sources.get(path)
        if seed is None:
            ancestor = next(
                (
                    sources[path[:size]]
                    for size in range(len(path) - 1, 0, -1)
                    if path[:size] in sources
                ),
                None,
            )
            seed = SeedSource(
                ancestor.source if ancestor is not None else None, "missing"
            )
        rows.append(
            FieldDiagnostic(
                path,
                messages.get(path),
                name,
                index,
                meta.description
                if meta is not None and meta.description is not None
                else seed.source,
                seed.status,
                None if meta is None or meta.calibration is None else True,
            )
        )
    return tuple(rows)
