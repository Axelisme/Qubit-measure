"""Construct complete library proposals without replacing referenced children."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.cfg_model import CfgInputValue, module_spec_from_raw
from zcu_tools.program.v2.modules import ModuleCfgFactory
from zcu_tools.resources.parameter_fields import set_model_value

from .items import ModuleWriteback

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext
    from zcu_tools.program.v2.modules import AbsModuleCfg


def calibrated_module_writeback(
    parameters: ParameterContext,
    tested_module: AbsModuleCfg,
    *,
    target: str,
    description: str,
    field_updates: Mapping[str, CfgInputValue],
) -> ModuleWriteback:
    """Build a complete replacement from captured library Spec and exact updates.

    parameters is the Run/Load read-only capture. target is a library root or
    module slot (including scalar aliases). tested_module is its materialized
    native Run cfg. field_updates maps exact dotted Spec paths to new values;
    only these paths may differ from the captured materialized library cfg.
    Missing paths, invalid Specs or any other differing field raise ValueError
    or the path owner's error. The proposal retains untouched child keys.
    No files, live selection, provenance, draft or apply state are modified.
    """
    key = parameters.library_key(target)
    library_module = ModuleCfgFactory.from_core(
        parameters.library.get_module(key),
        context=parameters.library.validation_context(),
    )
    differences = _differing_fields(
        library_module.to_dict(),
        tested_module.to_dict(),
        (),
        {tuple(path.split(".")) for path in field_updates},
    )
    if differences:
        raise ValueError(f"Tested module differs from {key}: {', '.join(differences)}")
    candidate = parameters.library.module_spec(key)
    for path, value in field_updates.items():
        set_model_value(candidate, path, value)
    return ModuleWriteback(
        target_name=key,
        description=description,
        proposal=module_spec_from_raw(candidate.to_dict()),
    )


def _differing_fields(
    expected: object,
    actual: object,
    path: tuple[str, ...],
    ignored: set[tuple[str, ...]],
) -> list[str]:
    if path in ignored:
        return []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for name in sorted(expected.keys() | actual.keys()):
            child_path = (*path, name)
            if child_path in ignored:
                continue
            if name not in expected or name not in actual:
                differences.append(".".join(child_path))
            else:
                differences.extend(
                    _differing_fields(expected[name], actual[name], child_path, ignored)
                )
        return differences
    if (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        return [
            field
            for i, (left, right) in enumerate(zip(expected, actual, strict=True))
            for field in _differing_fields(left, right, (*path, str(i)), ignored)
        ]
    return [] if expected == actual else [".".join(path)]
