"""Schema declaration helpers for autofluxdep experiment nodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    ModuleRefSpec,
    ModuleRefValue,
    NodeCfgSchema,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    align_locked_literals,
    make_default_value,
    str_choice_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.module_adapter import module_cfg_to_value
from zcu_tools.gui.session.types import ExpContext

_MISSING = object()


class NodeSchemaBuilder:
    """Linear builder for an autofluxdep node's typed knob schema.

    The builder owns only the mechanical cfg-tree assembly: path mounting,
    default value wrapping, logical-key projection, and choice-section wiring.
    Experiment policy stays in the node's ``make_default_schema`` and
    ``make_cfg``.
    """

    def __init__(self, *, label: str = "") -> None:
        self._spec = CfgSectionSpec(label=label, fields={})
        self._value = CfgSectionValue(fields={})
        self._logical_paths: dict[str, str] = {}
        self._choice_groups: list[tuple[str, str, Mapping[str, tuple[str, ...]]]] = []

    def field(
        self,
        logical_key: str,
        path: str,
        *,
        spec: CfgNodeSpec,
        default: Any = _MISSING,
    ) -> NodeSchemaBuilder:
        if logical_key in self._logical_paths:
            raise ValueError(f"duplicate logical key {logical_key!r}")
        value = (
            _default_value_for_spec(spec)
            if default is _MISSING
            else _wrap_default(spec, default)
        )
        _set_cfg_path(self._spec, self._value, path, spec, value)
        self._logical_paths[logical_key] = path
        return self

    def logical(self, logical_key: str, path: str) -> NodeSchemaBuilder:
        if logical_key in self._logical_paths:
            raise ValueError(f"duplicate logical key {logical_key!r}")
        self._logical_paths[logical_key] = path
        return self

    def float(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: Any,
        decimals: int | None = None,
        optional: bool = False,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=FloatSpec(
                label=label,
                decimals=decimals,
                optional=optional,
                tooltip=tooltip,
            ),
            default=default,
        )

    def int(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: int,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=IntSpec(label=label, tooltip=tooltip),
            default=default,
        )

    def bool(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: bool,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=ScalarSpec(label=label, type=bool, tooltip=tooltip),
            default=default,
        )

    def choice(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        choices: tuple[str, ...],
        default: str,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=str_choice_spec(label, choices, tooltip=tooltip),
            default=default,
        )

    def sweep(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: SweepValue,
        decimals: int | None = None,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=SweepSpec(label=label, decimals=decimals, tooltip=tooltip),
            default=default,
        )

    def centered_sweep(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: CenteredSweepValue,
        decimals: int | None = None,
        tooltip: str = "",
        center_editable: bool = True,
        center_badge: str = "",
        center_tooltip: str = "",
        locked_center: float | None = None,
    ) -> NodeSchemaBuilder:
        return self.field(
            logical_key,
            path,
            spec=CenteredSweepSpec(
                label=label,
                decimals=decimals,
                tooltip=tooltip,
                center_editable=center_editable,
                center_badge=center_badge,
                center_tooltip=center_tooltip,
                locked_center=locked_center,
            ),
            default=default,
        )

    def acquire_retry(self, default: int) -> NodeSchemaBuilder:
        self._ensure_section("generation.acquisition", "Acquisition guardrails")
        return self.int(
            "acquire_retry",
            "generation.acquisition.acquire_retry",
            label="retry",
            default=default,
            tooltip="Retries for transient program build or acquire failures.",
        )

    def acquisition(
        self,
        *,
        earlystop_snr: float | None,
        acquire_retry: int,
    ) -> NodeSchemaBuilder:
        self._ensure_section("generation.acquisition", "Acquisition guardrails")
        return self.float(
            "earlystop_snr",
            "generation.acquisition.earlystop_snr",
            label="earlystop_snr",
            default=earlystop_snr,
            optional=True,
            tooltip="Stop averaging once completed-round SNR reaches this value.",
        ).acquire_retry(acquire_retry)

    def auto_relax_from_t1(
        self,
        *,
        seed_us: float,
        factor: float,
        minimum_us: float,
    ) -> NodeSchemaBuilder:
        self._ensure_section("generation.relax", "Relax timing")
        self.choice(
            "relax_delay_mode",
            "generation.relax.relax_delay_mode",
            label="delay_mode",
            choices=("auto_t1", "fixed"),
            default="auto_t1",
            tooltip="Auto derives relax delay from T1; fixed keeps Default cfg delay.",
        )
        self.float(
            "t1_seed_us",
            "generation.relax.t1_seed_us",
            label="initial_t1_us",
            default=seed_us,
            tooltip="Initial T1 before measured feedback exists.",
        )
        self.float(
            "relax_factor",
            "generation.relax.relax_factor",
            label="factor",
            default=factor,
            tooltip="Multiplier applied to T1 for auto relax delay.",
        )
        self.float(
            "relax_min_us",
            "generation.relax.relax_min_us",
            label="min_us",
            default=minimum_us,
            tooltip="Minimum auto relax delay.",
        )
        self.choice_group(
            "generation.relax",
            "relax_delay_mode",
            {
                "fixed": (),
                "auto_t1": ("relax_factor", "relax_min_us"),
            },
        )
        return self

    def auto_sweep_stop_from_t1(
        self,
        *,
        stop_factor: float,
        stop_min_us: float,
        stop_max_us: float,
        group_label: str,
    ) -> NodeSchemaBuilder:
        self._ensure_section("generation.sweep", group_label)
        self.choice(
            "sweep_range_mode",
            "generation.sweep.sweep_range_mode",
            label="range_mode",
            choices=("auto_t1", "fixed"),
            default="auto_t1",
            tooltip=(
                "Auto derives the sweep stop from latest trusted T1; "
                "start/expts stay in Default cfg."
            ),
        )
        self.float(
            "sweep_stop_factor",
            "generation.sweep.sweep_stop_factor",
            label="stop_factor",
            default=stop_factor,
            tooltip="T1 multiplier for the auto sweep stop.",
        )
        self.float(
            "sweep_stop_min_us",
            "generation.sweep.sweep_stop_min_us",
            label="stop_min_us",
            default=stop_min_us,
            tooltip="Minimum stop value for the auto T1 sweep.",
        )
        self.float(
            "max_length",
            "generation.sweep.max_length",
            label="max_length",
            default=stop_max_us,
            tooltip="Maximum stop value for the auto T1 sweep.",
        )
        self.choice_group(
            "generation.sweep",
            "sweep_range_mode",
            {
                "fixed": (),
                "auto_t1": (
                    "sweep_stop_factor",
                    "sweep_stop_min_us",
                    "max_length",
                ),
            },
        )
        return self

    def feedback_slot(
        self,
        slot: Any,
        *,
        group: str = "feedback",
        group_label: str | None = None,
    ) -> NodeSchemaBuilder:
        section_path = f"generation.{group}"
        self._ensure_section(section_path, group_label or _section_label(group))
        strategy_key = slot.field_name("strategy")
        if slot.kind == "estimator":
            self.choice(
                strategy_key,
                f"{section_path}.{strategy_key}",
                label="strategy",
                choices=("off", "idw", "last_good"),
                default=str(slot.default_strategy),
                tooltip="Select how trusted samples estimate the next value.",
            )
            self.int(
                slot.field_name("idw_k"),
                f"{section_path}.{slot.field_name('idw_k')}",
                label="idw_k",
                default=int(slot.default_idw_k),
                tooltip="Nearest trusted samples used by IDW estimation.",
            )
            self.float(
                slot.field_name("idw_epsilon"),
                f"{section_path}.{slot.field_name('idw_epsilon')}",
                label="idw_epsilon",
                default=float(slot.default_idw_epsilon),
                tooltip="Small distance floor for IDW weighting.",
            )
            self.float(
                slot.field_name("decay_points"),
                f"{section_path}.{slot.field_name('decay_points')}",
                label="decay_points",
                default=float(slot.default_decay_points),
                tooltip="Flux queries before stale estimates fade out.",
            )
            return self.choice_group(
                section_path,
                strategy_key,
                {
                    "off": (),
                    "idw": (
                        slot.field_name("idw_k"),
                        slot.field_name("idw_epsilon"),
                        slot.field_name("decay_points"),
                    ),
                    "last_good": (slot.field_name("decay_points"),),
                },
            )

        if slot.kind == "controller":
            self.choice(
                strategy_key,
                f"{section_path}.{strategy_key}",
                label="strategy",
                choices=("off", "log_step"),
                default=str(slot.default_strategy),
                tooltip="Select whether controller feedback adjusts the next value.",
            )
            return self.choice_group(
                section_path,
                strategy_key,
                {
                    "off": (),
                    "log_step": (),
                },
            )

        raise ValueError(f"unsupported feedback slot kind: {slot.kind!r}")

    def choice_group(
        self,
        section_path: str,
        selector_key: str,
        choices: Mapping[str, tuple[str, ...]],
    ) -> NodeSchemaBuilder:
        if not choices:
            raise ValueError("choice_group needs at least one choice")
        self._choice_groups.append((section_path, selector_key, choices))
        return self

    def build(self) -> NodeCfgSchema:
        for section_path, selector_key, choices in self._choice_groups:
            _replace_with_choice_section(
                self._spec, section_path, selector_key, choices
            )
        align_locked_literals(self._spec, self._value)
        return NodeCfgSchema(
            CfgSchema(spec=self._spec, value=self._value),
            logical_paths=dict(self._logical_paths),
        )

    def _ensure_section(self, path: str, label: str) -> None:
        _ensure_section(self._spec, self._value, path, label)


def module_ref_default(
    ctx: Any | None,
    spec: ModuleRefSpec,
    *names: str,
    accepted_types: tuple[str, ...] = (),
) -> ModuleRefValue | None:
    linked = module_ref_value_from_ctx(ctx, *names, accepted_types=accepted_types)
    if linked is not None:
        return linked
    value = _default_value_for_spec(spec)
    if value is None:
        return None
    if not isinstance(value, ModuleRefValue):
        raise TypeError(f"ModuleRefSpec default produced {type(value).__name__}")
    return value


def module_ref_value_from_ctx(
    ctx: Any | None,
    *names: str,
    accepted_types: tuple[str, ...] = (),
) -> ModuleRefValue | None:
    if not isinstance(ctx, ExpContext):
        return None
    for name in names:
        try:
            module = ctx.ml.get_module(name)
        except (KeyError, ValueError):
            module = None
        if module is None:
            continue
        if accepted_types and _module_type(module) not in set(accepted_types):
            continue
        _, value = module_cfg_to_value(module)
        return ModuleRefValue(chosen_key=name, value=value)
    return None


def _module_type(module: Any) -> str | None:
    if isinstance(module, Mapping):
        value = module.get("type")
    else:
        value = getattr(module, "type", None)
        if value is None and hasattr(module, "to_dict"):
            raw = module.to_dict()
            if isinstance(raw, Mapping):
                value = raw.get("type")
    return str(value) if value is not None else None


def _default_value_for_spec(spec: CfgNodeSpec) -> Any:
    return make_default_value(CfgSectionSpec(fields={"value": spec})).fields["value"]


def _wrap_default(spec: CfgNodeSpec, default: Any) -> Any:
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec default must be SweepValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, CenteredSweepSpec):
        if not isinstance(default, CenteredSweepValue):
            raise TypeError(
                "CenteredSweepSpec default must be CenteredSweepValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(spec, ScalarSpec):
        if isinstance(default, (DirectValue, EvalValue)):
            return default
        return DirectValue(default)
    if isinstance(spec, ModuleRefSpec):
        if default is None and spec.optional:
            return None
        if not isinstance(default, ModuleRefValue):
            raise TypeError(
                f"ModuleRefSpec default must be ModuleRefValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, WaveformRefSpec):
        if default is None and spec.optional:
            return None
        if not isinstance(default, WaveformRefValue):
            raise TypeError(
                "WaveformRefSpec default must be WaveformRefValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(spec, CfgSectionSpec):
        if not isinstance(default, CfgSectionValue):
            raise TypeError(
                f"CfgSectionSpec default must be CfgSectionValue, got {type(default).__name__}"
            )
        return default
    return default


def _set_cfg_path(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    path: str,
    spec: CfgNodeSpec,
    value: Any,
) -> None:
    parent_spec, parent_value, leaf = _ensure_parent(root_spec, root_value, path)
    if leaf in parent_spec.fields:
        raise ValueError(f"cfg path {path!r} already exists")
    parent_spec.fields[leaf] = spec
    parent_value.fields[leaf] = value


def _ensure_section(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    path: str,
    label: str,
) -> None:
    parent_spec, parent_value, leaf = _ensure_parent(root_spec, root_value, path)
    spec = parent_spec.fields.get(leaf)
    value = parent_value.fields.get(leaf)
    if spec is None:
        parent_spec.fields[leaf] = CfgSectionSpec(label=label, fields={})
        parent_value.fields[leaf] = CfgSectionValue(fields={})
        return
    if not isinstance(spec, CfgSectionSpec) or not isinstance(value, CfgSectionValue):
        raise TypeError(f"cfg path {path!r} is not a section")
    if label and not spec.label:
        parent_spec.fields[leaf] = replace(spec, label=label)


def _ensure_parent(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    path: str,
) -> tuple[CfgSectionSpec, CfgSectionValue, str]:
    parts = _split_path(path)
    spec = root_spec
    value = root_value
    for idx, part in enumerate(parts[:-1]):
        child_spec = spec.fields.get(part)
        child_value = value.fields.get(part)
        if child_spec is None:
            child_spec = CfgSectionSpec(label=_section_label(part), fields={})
            child_value = CfgSectionValue(fields={})
            spec.fields[part] = child_spec
            value.fields[part] = child_value
        if not isinstance(child_spec, CfgSectionSpec) or not isinstance(
            child_value, CfgSectionValue
        ):
            parent = ".".join(parts[: idx + 1])
            raise TypeError(f"cfg path {path!r} cannot descend through {parent!r}")
        spec = child_spec
        value = child_value
    return spec, value, parts[-1]


def _replace_with_choice_section(
    root_spec: CfgSectionSpec,
    section_path: str,
    selector_key: str,
    choices: Mapping[str, tuple[str, ...]],
) -> None:
    parent, leaf = _section_parent(root_spec, section_path)
    section = parent.fields[leaf]
    if not isinstance(section, CfgSectionSpec):
        raise TypeError(f"choice section {section_path!r} is not a section")
    binding = _choice_binding(section, selector_key, choices)
    bindings = section.bindings if isinstance(section, ChoiceSectionSpec) else ()
    parent.fields[leaf] = ChoiceSectionSpec(
        fields=dict(section.fields),
        label=section.label,
        inherit_hook=section.inherit_hook,
        bindings=(*bindings, binding),
    )


def _choice_binding(
    section: CfgSectionSpec,
    selector_key: str,
    choices: Mapping[str, tuple[str, ...]],
) -> ChoiceBinding:
    variants: dict[str, CfgSectionSpec] = {}
    for value, field_keys in choices.items():
        fields: dict[str, CfgNodeSpec] = {}
        for key in field_keys:
            try:
                fields[key] = section.fields[key]
            except KeyError as exc:
                raise ValueError(
                    f"choice {selector_key}={value!r} references unknown field {key!r}"
                ) from exc
        variants[value] = CfgSectionSpec(fields=fields, label=value)
    return ChoiceBinding(selector_key, variants)


def _section_parent(root_spec: CfgSectionSpec, path: str) -> tuple[CfgSectionSpec, str]:
    parts = _split_path(path)
    section = root_spec
    for part in parts[:-1]:
        child = section.fields.get(part)
        if not isinstance(child, CfgSectionSpec):
            raise TypeError(f"cfg path {path!r} cannot descend through {part!r}")
        section = child
    leaf = parts[-1]
    if leaf not in section.fields:
        raise KeyError(f"cfg path {path!r} is missing")
    return section, leaf


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError("cfg path must not be empty")
    return parts


def _section_label(key: str) -> str:
    labels = {
        "modules": "Modules",
        "sweep": "Sweep",
        "generation": "Generation overrides",
        "acquisition": "Acquisition guardrails",
        "relax": "Relax timing",
    }
    return labels.get(key, key.replace("_", " ").title())
