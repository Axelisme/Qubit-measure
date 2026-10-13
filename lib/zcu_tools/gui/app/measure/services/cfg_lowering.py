"""App schema-to-Spec conversion for fixed-capture library structure commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zcu_tools.cfg_model import module_spec_from_raw, waveform_spec_from_raw
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.context_types import LibraryChange

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.session.types import ParameterContext


def lower_module(schema: CfgSchema, parameters: ParameterContext) -> ModuleSpec:
    """Lower with fixed roles/values into a stored ModuleSpec.

    Keep the existing finished-schema policy: linked children lower to captured
    inline data; this command does not introduce a new reference codec.

    Invalid finished schema, unavailable references/expressions and malformed
    module input raise InvalidInputError before any resource write. Unexpected
    source/I/O failures propagate. No live selection is consulted.
    """
    try:
        return module_spec_from_raw(schema_to_raw_dict(schema, parameters))
    except (ValueError, RuntimeError) as exc:
        raise InvalidInputError(f"Invalid module configuration: {exc}") from exc


def lower_waveform(schema: CfgSchema, parameters: ParameterContext) -> WaveformSpec:
    """Lower a fixed-capture waveform schema to Spec with lower_module's failures."""
    try:
        return waveform_spec_from_raw(schema_to_raw_dict(schema, parameters))
    except (ValueError, RuntimeError) as exc:
        raise InvalidInputError(f"Invalid waveform configuration: {exc}") from exc


def lowered_library_change(
    parameters: ParameterContext,
    kind: Literal["module", "waveform"],
    action: Literal["put", "replace"],
    path: str,
    schema: CfgSchema,
    *,
    new_path: str | None = None,
) -> LibraryChange:
    """Build a put/replace LibraryChange whose Spec is lowered from schema.

    kind selects lower_module or lower_waveform; parameters is the fixed capture
    used for lowering and must be the one later passed to apply_library_change.
    path/new_path follow LibraryChange (new_path is required only for replace).
    Lowering failures raise lower_module's errors; LibraryChange rejects an
    invalid action/new_path combination with InvalidInputError. Nothing is
    written.
    """
    if kind == "module":
        return LibraryChange(
            kind, action, path, new_path, module=lower_module(schema, parameters)
        )
    return LibraryChange(
        kind, action, path, new_path, waveform=lower_waveform(schema, parameters)
    )
