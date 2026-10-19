"""Detached aggregate observations from one Form input and fixed capture."""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from zcu_tools.gui.form import FormErrors, FormValue, FormWritePath

from .logical import LogicalFieldObservation


@dataclass(frozen=True)
class ElementObservation:
    """One element in original list order, independent of participants or picker.

    index is its zero-based original position; path is its absolute canonical
    typed path. name is the owner's literal identity, or the description label
    for ordinary lists. input is detached authored JSON, including invalid
    expressions and missing fields. selections maps absolute canonical paths to
    explicit authored selector values, independently of the raw branch.
    values maps absolute canonical leaf paths
    to available ordinary preview values, not lowered native cfg. References
    remain authored ref fields; use read_module for explicit materialization.
    fields holds requested relative logical observations in request order, with
    same-input FixedElements identity; empty when none were requested.
    errors contains directed element/ancestor errors, or None. Invalid leaves
    do not remove other values or elements. No container aliases live state.
    """

    index: int
    name: str
    path: FormWritePath
    input: dict[str, JsonValue]
    selections: Mapping[FormWritePath, JsonValue]
    values: Mapping[FormWritePath, FormValue]
    errors: FormErrors | None
    fields: tuple[LogicalFieldObservation, ...] = ()


@dataclass(frozen=True)
class ElementsObservation:
    """One readonly model-list observation; not an acceptance or mutation token.

    path names the requested model-list. elements includes every original
    position in list order. errors includes the complete Form preview errors,
    including participants and unrelated fields, or None. Values, input and
    directed errors are produced from the same detached input and fixed env.
    No converter, capture, installation, notification or UI focus runs.
    """

    path: FormWritePath
    elements: tuple[ElementObservation, ...]
    errors: FormErrors | None
