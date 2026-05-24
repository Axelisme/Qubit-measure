from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from .types import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


def _section_to_dict_inner(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
    path: list[str],
) -> dict:
    result: dict[str, Any] = {}
    extra_keys = set(value.fields.keys()) - set(spec.fields.keys())
    if extra_keys:
        section = ".".join(path) or "<root>"
        extras = ", ".join(sorted(extra_keys))
        raise RuntimeError(f"Config section '{section}' has unknown fields: {extras}")
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            if isinstance(node_spec, LiteralSpec):
                result[key] = node_spec.value
                continue
            label = getattr(node_spec, "label", "") or key
            full_path = ".".join([*path, key])
            raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_val, (DirectValue, EvalValue))
            if isinstance(node_val, DirectValue):
                if node_val.is_unset:
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
                result[key] = node_val.value
            else:
                if node_val.resolved is None:
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(
                        f"Config field '{full_path}' ({label}) expression "
                        f"{node_val.expr!r} is unresolved"
                    )
                result[key] = node_val.resolved

        elif isinstance(node_spec, LiteralSpec):
            result[key] = node_spec.value

        elif isinstance(node_spec, SweepSpec):
            assert isinstance(node_val, SweepValue)
            from zcu_tools.notebook.utils import make_sweep

            if node_val.step is not None:
                result[key] = make_sweep(
                    node_val.start, node_val.stop, step=node_val.step
                )
            else:
                result[key] = make_sweep(
                    node_val.start, node_val.stop, expts=node_val.expts
                )

        elif isinstance(node_spec, MultiSweepSpec):
            assert isinstance(node_val, MultiSweepValue)
            from zcu_tools.notebook.utils import make_sweep

            result[key] = {
                axis: (
                    make_sweep(sv.start, sv.stop, step=sv.step)
                    if sv.step is not None
                    else make_sweep(sv.start, sv.stop, expts=sv.expts)
                )
                for axis, sv in node_val.axes.items()
            }

        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            assert isinstance(node_val, (ModuleRefValue, WaveformRefValue))
            if not isinstance(node_val.value, CfgSectionValue):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")
            result[key] = _section_to_dict_inner(
                _find_allowed_spec(node_spec, node_val, ml),
                node_val.value,
                ml,
                [*path, key],
            )

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_val, CfgSectionValue)
            result[key] = _section_to_dict_inner(
                node_spec, node_val, ml, [*path, key]
            )

        else:
            raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")

    return result


def _section_to_dict(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
) -> dict:
    """Public entry point for lowering a section; path starts at root."""
    return _section_to_dict_inner(spec, value, ml, [])


def _find_allowed_spec(
    ref_spec: Union[ModuleRefSpec, WaveformRefSpec],
    ref_val: Union[ModuleRefValue, WaveformRefValue],
    ml: "Optional[ModuleLibrary]",
) -> CfgSectionSpec:
    """Return the CfgSectionSpec from allowed[] that matches chosen_key's label."""
    chosen = ref_val.chosen_key
    if chosen.startswith("<Custom:"):
        if not chosen.endswith(">"):
            raise RuntimeError(f"Invalid custom reference key: {chosen!r}")
        label = chosen[len("<Custom:") : -1]
        for spec in ref_spec.allowed:
            if spec.label == label:
                return spec
        allowed = ", ".join(spec.label for spec in ref_spec.allowed)
        raise RuntimeError(
            f"Unknown custom reference label {label!r}; allowed labels: {allowed}"
        )

    if ml is None:
        raise RuntimeError(
            f"Cannot resolve library reference {chosen!r} without ModuleLibrary"
        )

    from zcu_tools.gui.cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

    if isinstance(ref_spec, ModuleRefSpec):
        if chosen not in ml.modules:
            raise RuntimeError(f"Unknown module reference: {chosen!r}")
        chosen_spec, _ = module_cfg_to_value(ml.modules[chosen])
    else:
        if chosen not in ml.waveforms:
            raise RuntimeError(f"Unknown waveform reference: {chosen!r}")
        chosen_spec, _ = waveform_cfg_to_value(ml.waveforms[chosen])

    for spec in ref_spec.allowed:
        if spec.label == chosen_spec.label:
            return spec
    allowed = ", ".join(spec.label for spec in ref_spec.allowed)
    raise RuntimeError(
        f"Library reference {chosen!r} resolved to unsupported spec "
        f"{chosen_spec.label!r}; allowed labels: {allowed}"
    )


def schema_to_dict(schema: CfgSchema, ml: "Optional[ModuleLibrary]") -> dict:
    """Lower a CfgSchema using the same section lowerer as CfgSchema.to_raw_dict()."""
    return _section_to_dict_inner(schema.spec, schema.value, ml, [])
