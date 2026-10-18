"""Detached Run conditions shared by author handles and execution projections."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import NotRequired, TypedDict


class SweepSources(TypedDict):
    """Provenance labels for a sweep's three public inputs.

    start and stop label the respective supplied endpoints in their native unit.
    expts labels the supplied integer point count. Each label must be non-empty.
    Omitted sweep values use gui_default regardless of these supplied labels.
    """

    start: str
    stop: str
    expts: str


class ActualField(TypedDict):
    """A resolved Run condition with its already chosen provenance.

    value is the resolved scalar, module reference or named sweep values.
    source is the caller's captured parameter/calibration/library/default label,
    or SweepSources for individually labeled sweep inputs. None or omission
    means no source was declared. unit is an asserted native unit,
    never a request to convert value.
    """

    value: object
    source: NotRequired[str | SweepSources | None]
    unit: NotRequired[str | None]


class RecipeActual(TypedDict):
    """Run-before-start capture; later queries do not replace it with live data.

    fields maps public SummaryParameter names to value/source/unit facts.
    Typed cfg paths are used only to select these facts before capture.
    Facts are detached from caller state; no cfg inputs are retained.
    """

    fields: dict[str, ActualField]


def capture_actual(fields: Mapping[str, ActualField]) -> RecipeActual:
    """Detach declared summary facts for this Run.

    Only the declared facts in fields enter the capture.
    fields contains resolved value/source/unit facts, not cfg inputs or edits.
    Preserve source labels and units verbatim; do not read GUI state, retain
    publication, resolve expressions or convert.
    """
    return RecipeActual(
        fields=deepcopy(dict(fields)),
    )
