"""Conservative projection of execution snapshots into an existing GUI cfg."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.main.cfg_schemas import MAIN_PROGRAM_MATERIALIZATION_POLICY
from zcu_tools.gui.cfg import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.lowering import validate_finished_cfg
from zcu_tools.gui.cfg.materialization import materialize_spec_value
from zcu_tools.program.v2.sweep import SweepCfg

_OptionProvider = Callable[[str], Sequence[object]]


def project_loaded_cfg(
    current: CfgSchema,
    snapshot: ExpCfgModel,
    *,
    provide_options: _OptionProvider | None = None,
) -> CfgSchema | None:
    """Return a detached complete candidate, or None if nothing can be adopted.

    This does not recover editing expressions or library identities. Dynamic
    selectors need live options; without a provider they remain unchanged.
    Validation of the complete candidate belongs to the caller.
    """
    candidate = deepcopy(current)
    raw = snapshot.model_dump(mode="python", exclude_none=False)
    if _project_section(candidate.spec, candidate.value, raw, (), provide_options) == 0:
        return None
    return candidate


def _project_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    raw: Mapping[str, object],
    path: tuple[str, ...],
    provide_options: _OptionProvider | None,
) -> int:
    adopted = 0
    for name, child_spec in spec.fields.items():
        if name not in raw:
            continue
        incoming = raw[name]
        child_path = (*path, name)
        if child_path == ("dev",) and isinstance(incoming, Mapping):
            # Runtime devices are keyed by physical name; GUI selectors by role.
            roles: dict[str, list[str]] = {}
            for physical_name, info in incoming.items():
                if isinstance(physical_name, str) and isinstance(info, Mapping):
                    label = info.get("label")
                    if isinstance(label, str):
                        roles.setdefault(label, []).append(physical_name)
            incoming = {
                role: names[0] for role, names in roles.items() if len(names) == 1
            }
        if isinstance(child_spec, CfgSectionSpec):
            child_value = value.fields[name]
            if isinstance(incoming, Mapping) and isinstance(
                child_value, CfgSectionValue
            ):
                adopted += _project_section(
                    child_spec, child_value, incoming, child_path, provide_options
                )
            continue
        supported, replacement = _project_node(
            child_spec, incoming, child_path, provide_options
        )
        if supported:
            value.fields[name] = replacement
            adopted += 1
    return adopted


def _project_node(
    spec: CfgNodeSpec,
    raw: object,
    path: tuple[str, ...],
    provide_options: _OptionProvider | None,
) -> tuple[bool, CfgNodeValue | None]:
    if isinstance(spec, ScalarSpec) and spec.editable:
        if (raw is None and not spec.optional) or (spec.required and raw == ""):
            return False, None
        value = DirectValue(deepcopy(raw))
        schema = CfgSchema(
            spec=CfgSectionSpec(fields={"value": spec}),
            value=CfgSectionValue(fields={"value": value}),
        )
        try:
            validate_finished_cfg(schema, resolve_reference=None)
        except (RuntimeError, TypeError, ValueError):
            return False, None
        if spec.choices_source and raw is not None:
            if provide_options is None:
                return False, None
            options = provide_options(spec.choices_source)
            if isinstance(options, (str, bytes)):
                raise TypeError(
                    "Dynamic choice provider must return an option sequence"
                )
            if raw not in options and not (
                spec.type is str and not spec.required and raw == ""
            ):
                return False, None
        return True, value
    if isinstance(spec, SweepSpec) and spec.editable and isinstance(raw, Mapping):
        try:
            # Strict types and the runtime model's own endpoint/step invariant
            # prevent accepting coerced strings or silently changing a sweep.
            sweep = SweepCfg.model_validate(raw, strict=True)
        except ValueError:
            return False, None
        return True, SweepValue(sweep.start, sweep.stop, sweep.expts)
    if isinstance(spec, ReferenceSpec):
        if raw is None:
            return spec.optional, None
        if not isinstance(raw, Mapping):
            return False, None
        try:
            _require_complete(spec, raw, path)
            wrapper = CfgSectionSpec(fields={"value": spec})
            projected = materialize_spec_value(
                wrapper, {"value": raw}, policy=MAIN_PROGRAM_MATERIALIZATION_POLICY
            )
            validate_finished_cfg(
                CfgSchema(spec=wrapper, value=projected), resolve_reference=None
            )
        except (RuntimeError, TypeError, ValueError):
            return False, None
        return True, projected.fields["value"]
    return False, None


def _require_complete(spec: CfgNodeSpec, raw: object, path: tuple[str, ...]) -> None:
    """Reject inputs that would make the program materializer invent defaults."""
    if isinstance(spec, ReferenceSpec):
        if raw is None and spec.optional:
            return
        if not isinstance(raw, Mapping):
            raise ValueError("Missing reference config")
        discriminator = "type" if spec.kind == "module" else "style"
        if discriminator not in raw:
            raise ValueError("Missing reference discriminator")
        selected = MAIN_PROGRAM_MATERIALIZATION_POLICY.reference_value(path, spec, raw)
        if selected is None:
            raise ValueError("Missing reference shape")
        _require_complete(selected.spec, raw, path)
    elif isinstance(spec, CfgSectionSpec):
        if not isinstance(raw, Mapping):
            raise ValueError("Missing section")
        for name, child in spec.fields.items():
            if name not in raw:
                raise ValueError(f"Missing config field {'.'.join((*path, name))}")
            _require_complete(child, raw[name], (*path, name))
    elif isinstance(spec, ScalarSpec):
        if (raw is None and not spec.optional) or (spec.required and raw == ""):
            raise ValueError("Snapshot scalar is not valid for the reference")
    elif isinstance(spec, LiteralSpec) and raw != spec.value:
        raise ValueError("Snapshot disagrees with locked literal")
