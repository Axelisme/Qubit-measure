"""Autoflux-local domain builder for experiment node schemas."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.cfg.module_adapter import (
    pulse_module_ref_spec,
    pulse_readout_module_ref_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, str_choice_spec
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSectionSpec,
    FloatSpec,
    IntSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarLeafInput,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    read_value_path,
    resolve_spec_path,
)

from ._schema_tree import _default_value_for_spec, _SchemaTree
from .module_values import _seed_module_reference

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.feedback.runtime import FeedbackSlotDecl


class NodeSchemaBuilder:
    """One-shot domain builder for an autofluxdep node's typed cfg schema."""

    def __init__(self, ctx: Any | None = None, *, label: str = "") -> None:
        self._ctx = ctx
        self._tree = _SchemaTree(label=label)
        self._logical_paths: dict[str, str] = {}
        self._built = False

    def pulse(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        library_keys: tuple[str, ...] = (),
        optional: bool = False,
        blank_overrides: Mapping[str, ScalarLeafInput] | None = None,
        overrides: Mapping[str, ScalarLeafInput] | None = None,
        locked: Mapping[str, object] | None = None,
    ) -> NodeSchemaBuilder:
        """Declare a pulse reference with ordered ModuleLibrary adoption."""
        spec = pulse_module_ref_spec(label=label, optional=optional)
        spec, default = self._module_default(
            spec,
            verb="pulse",
            library_keys=library_keys,
            accepted_types=("pulse",),
            blank_overrides=blank_overrides,
            overrides=overrides,
            locked=locked,
        )
        return self._field(logical_key, path, spec=spec, default=default)

    def pulse_readout(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        library_keys: tuple[str, ...] = (),
        optional: bool = False,
        locked: Mapping[str, object] | None = None,
    ) -> NodeSchemaBuilder:
        """Declare a pulse-readout reference with ordered library adoption."""
        spec = pulse_readout_module_ref_spec(label=label, optional=optional)
        spec, default = self._module_default(
            spec,
            verb="pulse_readout",
            library_keys=library_keys,
            accepted_types=("readout/pulse",),
            blank_overrides=None,
            overrides=None,
            locked=locked,
        )
        return self._field(logical_key, path, spec=spec, default=default)

    def float(
        self,
        logical_key: str,
        path: str,
        *,
        label: str,
        default: ScalarLeafInput,
        decimals: int | None = None,
        optional: bool = False,
        tooltip: str = "",
    ) -> NodeSchemaBuilder:
        return self._field(
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
        return self._field(
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
        return self._field(
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
        return self._field(
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
        return self._field(
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
        return self._field(
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

    def knob(self, logical_key: str, existing_path: str) -> NodeSchemaBuilder:
        self._check_mutable()
        self._check_logical_key(logical_key)
        resolve_spec_path(self._tree.spec, existing_path)
        self._logical_paths[logical_key] = existing_path
        return self

    def acquisition(
        self,
        *,
        retry: int,
        early_stop_snr: float | None = None,
    ) -> NodeSchemaBuilder:
        declarations = [
            ("acquire_retry", "generation.acquisition.acquire_retry"),
        ]
        if early_stop_snr is not None:
            declarations.insert(
                0,
                ("earlystop_snr", "generation.acquisition.earlystop_snr"),
            )
        self._preflight_declarations(tuple(declarations))
        trial = self._copy_for_compound()
        trial._ensure_section("generation.acquisition", "Acquisition guardrails")
        if early_stop_snr is not None:
            trial.float(
                "earlystop_snr",
                "generation.acquisition.earlystop_snr",
                label="earlystop_snr",
                default=early_stop_snr,
                optional=True,
                tooltip="Stop averaging once completed-round SNR reaches this value.",
            )
        trial.int(
            "acquire_retry",
            "generation.acquisition.acquire_retry",
            label="retry",
            default=retry,
            tooltip="Retries for transient program build or acquire failures.",
        )
        self._adopt_compound(trial)
        return self

    def auto_relax_from_t1(
        self,
        *,
        seed_us: float,
        factor: float,
        minimum_us: float,
    ) -> NodeSchemaBuilder:
        self._preflight_declarations(
            (
                ("relax_delay_mode", "generation.relax.relax_delay_mode"),
                ("t1_seed_us", "generation.relax.t1_seed_us"),
                ("relax_factor", "generation.relax.relax_factor"),
                ("relax_min_us", "generation.relax.relax_min_us"),
            )
        )
        trial = self._copy_for_compound()
        trial._ensure_section("generation.relax", "Relax timing")
        trial.choice(
            "relax_delay_mode",
            "generation.relax.relax_delay_mode",
            label="delay_mode",
            choices=("auto_t1", "fixed"),
            default="auto_t1",
            tooltip="Auto derives relax delay from T1; fixed keeps Default cfg delay.",
        )
        trial.float(
            "t1_seed_us",
            "generation.relax.t1_seed_us",
            label="initial_t1_us",
            default=seed_us,
            tooltip="Initial T1 before measured feedback exists.",
        )
        trial.float(
            "relax_factor",
            "generation.relax.relax_factor",
            label="factor",
            default=factor,
            tooltip="Multiplier applied to T1 for auto relax delay.",
        )
        trial.float(
            "relax_min_us",
            "generation.relax.relax_min_us",
            label="min_us",
            default=minimum_us,
            tooltip="Minimum auto relax delay.",
        )
        trial.choice_fields(
            "generation.relax",
            "relax_delay_mode",
            {
                "fixed": (),
                "auto_t1": ("relax_factor", "relax_min_us"),
            },
        )
        self._adopt_compound(trial)
        return self

    def feedback_slot(
        self,
        slot: FeedbackSlotDecl,
        *,
        group: str = "feedback",
        group_label: str | None = None,
    ) -> NodeSchemaBuilder:
        if slot.kind not in {"estimator", "controller"}:
            raise ValueError(f"unsupported feedback slot kind: {slot.kind!r}")
        section_path = f"generation.{group}"
        strategy_key = slot.field_name("strategy")
        declarations = [(strategy_key, f"{section_path}.{strategy_key}")]
        if slot.kind == "estimator":
            for field in ("idw_k", "idw_epsilon", "decay_points"):
                key = slot.field_name(field)
                declarations.append((key, f"{section_path}.{key}"))
        self._preflight_declarations(tuple(declarations))

        trial = self._copy_for_compound()
        trial._ensure_section(section_path, group_label or _section_label(group))
        if slot.kind == "estimator":
            trial.choice(
                strategy_key,
                f"{section_path}.{strategy_key}",
                label="strategy",
                choices=("off", "idw", "last_good"),
                default=str(slot.default_strategy),
                tooltip="Select how trusted samples estimate the next value.",
            )
            trial.int(
                slot.field_name("idw_k"),
                f"{section_path}.{slot.field_name('idw_k')}",
                label="idw_k",
                default=slot.default_idw_k,
                tooltip="Nearest trusted samples used by IDW estimation.",
            )
            trial.float(
                slot.field_name("idw_epsilon"),
                f"{section_path}.{slot.field_name('idw_epsilon')}",
                label="idw_epsilon",
                default=slot.default_idw_epsilon,
                tooltip="Small distance floor for IDW weighting.",
            )
            trial.float(
                slot.field_name("decay_points"),
                f"{section_path}.{slot.field_name('decay_points')}",
                label="decay_points",
                default=slot.default_decay_points,
                tooltip="Flux queries before stale estimates fade out.",
            )
            trial.choice_fields(
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
            self._adopt_compound(trial)
            return self
        if slot.kind == "controller":
            trial.choice(
                strategy_key,
                f"{section_path}.{strategy_key}",
                label="strategy",
                choices=("off", "log_step"),
                default=str(slot.default_strategy),
                tooltip="Select whether controller feedback adjusts the next value.",
            )
            trial.choice_fields(
                section_path,
                strategy_key,
                {"off": (), "log_step": ()},
            )
            self._adopt_compound(trial)
            return self
        raise AssertionError("feedback slot kind was preflighted")

    def choice_fields(
        self,
        section_path: str,
        selector_key: str,
        fields_by_choice: Mapping[str, tuple[str, ...]],
        *,
        section_label: str | None = None,
    ) -> NodeSchemaBuilder:
        self._check_mutable()
        self._tree.add_choice_binding(
            section_path,
            selector_key,
            fields_by_choice,
            section_label=section_label,
        )
        return self

    def build(self) -> NodeCfgSchema:
        self._check_mutable()
        schema = self._tree.build()
        node_schema = NodeCfgSchema(schema, logical_paths=dict(self._logical_paths))
        self._built = True
        return node_schema

    def _field(
        self,
        logical_key: str,
        path: str,
        *,
        spec: CfgNodeSpec,
        default: object,
    ) -> NodeSchemaBuilder:
        self._check_mutable()
        self._check_logical_key(logical_key)
        self._tree.declare(path, spec, default)
        self._logical_paths[logical_key] = path
        return self

    def _module_default(
        self,
        spec: ReferenceSpec,
        *,
        verb: str,
        library_keys: tuple[str, ...],
        accepted_types: tuple[str, ...],
        blank_overrides: Mapping[str, ScalarLeafInput] | None,
        overrides: Mapping[str, ScalarLeafInput] | None,
        locked: Mapping[str, object] | None,
    ) -> tuple[ReferenceSpec, ReferenceValue | None]:
        self._check_mutable()
        blank = blank_overrides or {}
        always = overrides or {}
        locks = locked or {}
        self._validate_module_paths(
            spec,
            verb=verb,
            operation="blank_overrides",
            relative_paths=tuple(blank),
        )
        self._validate_module_paths(
            spec,
            verb=verb,
            operation="overrides",
            relative_paths=tuple(always),
        )
        self._validate_module_paths(
            spec,
            verb=verb,
            operation="locked",
            relative_paths=tuple(locks),
        )

        for relative_path, value in locks.items():
            spec = spec.lock_literal(relative_path, value)

        default = _seed_module_reference(
            self._ctx,
            library_keys,
            accepted_types=accepted_types,
        )
        is_blank = default is None
        if default is None:
            fallback = _default_value_for_spec(spec)
            if fallback is not None and not isinstance(fallback, ReferenceValue):
                raise TypeError(
                    f"ReferenceSpec default produced {type(fallback).__name__}"
                )
            default = fallback

        applicable: dict[str, tuple[str, ScalarLeafInput]] = {}
        if is_blank:
            applicable.update(
                (path, ("blank_overrides", value)) for path, value in blank.items()
            )
        applicable.update(
            (path, ("overrides", value)) for path, value in always.items()
        )
        if default is not None:
            for relative_path, (operation, _) in applicable.items():
                try:
                    read_value_path(default.value, relative_path)
                except KeyError as exc:
                    raise KeyError(
                        f"{verb} {operation} path {relative_path!r} is absent from "
                        "the selected module value"
                    ) from exc
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{verb} {operation} path {relative_path!r} cannot be "
                        "applied to the selected module value"
                    ) from exc
            for relative_path, (_, value) in applicable.items():
                default.value.with_field(relative_path, value)
        return spec, default

    @staticmethod
    def _validate_module_paths(
        spec: ReferenceSpec,
        *,
        verb: str,
        operation: str,
        relative_paths: tuple[str, ...],
    ) -> None:
        root = CfgSectionSpec(fields={"module": spec})
        for relative_path in relative_paths:
            if not relative_path:
                raise ValueError(f"{verb} {operation} path must not be empty")
            try:
                leaf = resolve_spec_path(root, f"module.{relative_path}")
            except KeyError as exc:
                raise KeyError(
                    f"{verb} {operation} path {relative_path!r} is missing from "
                    "the original module spec"
                ) from exc
            except TypeError as exc:
                raise TypeError(
                    f"{verb} {operation} path {relative_path!r} resolves "
                    "inconsistently in the original module spec"
                ) from exc
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{verb} {operation} path {relative_path!r} cannot descend "
                    "through the original module spec"
                ) from exc
            if not isinstance(leaf, ScalarSpec):
                raise TypeError(
                    f"{verb} {operation} path {relative_path!r} targets "
                    f"{type(leaf).__name__} in the original module spec; "
                    "expected ScalarSpec"
                )

    def _copy_for_compound(self) -> NodeSchemaBuilder:
        self._check_mutable()
        trial = NodeSchemaBuilder(self._ctx)
        trial._tree = deepcopy(self._tree)
        trial._logical_paths = dict(self._logical_paths)
        return trial

    def _adopt_compound(self, trial: NodeSchemaBuilder) -> None:
        self._tree = trial._tree
        self._logical_paths = trial._logical_paths

    def _preflight_declarations(
        self, declarations: tuple[tuple[str, str], ...]
    ) -> None:
        self._check_mutable()
        declared_keys: set[str] = set()
        for logical_key, _ in declarations:
            self._check_logical_key(logical_key)
            if logical_key in declared_keys:
                raise ValueError(f"duplicate logical key {logical_key!r}")
            declared_keys.add(logical_key)
        self._tree.validate_declarations(tuple(path for _, path in declarations))

    def _ensure_section(self, path: str, label: str) -> None:
        self._check_mutable()
        self._tree.ensure_section(path, label)

    def _check_logical_key(self, logical_key: str) -> None:
        if not logical_key:
            raise ValueError("Node logical key must not be empty")
        if "." in logical_key:
            raise ValueError(f"Node logical key must not contain '.': {logical_key!r}")
        if logical_key in self._logical_paths:
            raise ValueError(f"duplicate logical key {logical_key!r}")

    def _check_mutable(self) -> None:
        if self._built:
            raise RuntimeError("NodeSchemaBuilder is already built; create a new one")


def _section_label(key: str) -> str:
    labels = {
        "acquisition": "Acquisition guardrails",
        "relax": "Relax timing",
    }
    return labels.get(key, key.replace("_", " ").title())
