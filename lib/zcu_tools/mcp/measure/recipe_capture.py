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
    means no source was declared. input retains native raw
    expression/resolution facts when present. unit is an asserted native unit,
    never a request to convert value.
    """

    value: object
    source: NotRequired[str | SweepSources | None]
    input: NotRequired[object]
    unit: NotRequired[str | None]


class RecipeActual(TypedDict):
    """Run-before-start capture; later queries do not replace it with live data.

    cfg_ref is the opaque GUI identity/revision observation used to start Run.
    fields maps public cfg or derived parameter names to their resolved facts.
    source_basis is the GUI's opaque source observation basis.
    publication retains the complete native cfg publication, including raw
    expressions and diagnostics. All members are detached from caller state.
    """

    cfg_ref: object
    fields: dict[str, ActualField]
    source_basis: object
    publication: dict[str, object]


def capture_actual(
    publication: Mapping[str, object], fields: Mapping[str, ActualField]
) -> RecipeActual:
    """Detach one observed cfg publication and its resolved/source field facts.

    publication must contain cfg_ref and source_basis from tab.get_cfg/edit_cfg.
    fields contains caller-resolved conditions, not values to edit into the GUI.
    Missing publication keys raise KeyError. Preserve source labels, units and
    raw inputs verbatim; do not read GUI state, resolve expressions or convert.
    """
    return RecipeActual(
        cfg_ref=deepcopy(publication["cfg_ref"]),
        fields=deepcopy(dict(fields)),
        source_basis=deepcopy(publication["source_basis"]),
        publication=deepcopy(dict(publication)),
    )
