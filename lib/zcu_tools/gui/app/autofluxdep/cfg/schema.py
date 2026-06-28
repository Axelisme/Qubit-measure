"""Flat node-knob schema — the typed param SSOT each autofluxdep node owns.

A node's user knobs are a *flat* set of scalars (reps / rounds / relax_delay /
earlystop_snr / the pulse "設定頭" ch/nqz/gain/length/freq + the by-name waveform
string) plus, for sweep-defined knobs, a ``SweepSpec`` (qubit_freq's detune).
There is no nesting: unlike a measure cfg (modules / sweep / dev sections), a
node's knobs are flat typed leaves (the node's user-tunable settings).

``flat_node_schema`` builds the ``CfgSchema`` (spec + a complete default value
tree) from a field list. ``NodeCfgSchema`` wraps it as the node's param SSOT:

- ``lower()`` → a flat ``dict[str, value]`` (the lowered knobs ``make_cfg`` reads,
  replacing the old ``params.get(k, default)``). Scalars lower to their value;
  a ``SweepSpec`` lowers to a ``SweepCfg`` (via the framework lowering).
- ``set_field(key, value)`` writes one leaf, fast-failing an unknown key — the
  single typed entry both the cfg form (value-tree leaves) and tests (raw values)
  route through.
- ``with_overrides(params)`` seeds the value tree from a flat dict (a placement's
  construction overrides), fast-failing unknown keys — the boundary that turns a
  ``PlacedNode``'s seed dict into its typed schema.

The lowering uses ``md=None``: node knobs are plain scalars / sweeps with no
``EvalValue`` is allowed on numeric scalar knobs and sweep edges; lowering passes
the active MetaDict through the shared cfg lowering path so expressions resolve at
run time. The schema still holds no module/waveform refs, so ``ml`` is only
threaded through for the shared API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def sweepcfg_to_axis(sweep: Any) -> NDArray[np.float64]:
    """The explicit linspace axis a lowered ``SweepCfg`` (start/stop/expts) samples.

    A ``SweepSpec`` lowers to a ``SweepCfg`` (the FPGA sweep); a Result stores its
    swept axis as an explicit array. Rebuilds it from the cfg's endpoints + count
    so the Result columns match the program sweep exactly — the typed counterpart
    of the prototype's ``parse_detune_sweep`` (now that the axis is sweep-defined,
    not free-text ``start,stop,step``).
    """
    return np.linspace(float(sweep.start), float(sweep.stop), int(sweep.expts))


def _coerce_scalar(value: Any, type_: type) -> Any:
    """Coerce a (possibly text) scalar to its declared type, or None if blank/unset.

    Bridges a raw (possibly text) override into the typed value tree: a
    blank string / None means "unset" (leaf stays None → spec default or the
    node's Fast-Fail guard applies). A present value is coerced to the field's
    physical type (str passes through; int/float/bool parse). A malformed numeric
    fast-fails — typed knobs do NOT silently degrade to a default (the prototype's
    ``parse_*`` fallbacks are removed by this typing).
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        if type_ is str:
            return value
        if type_ is bool:
            return value.strip().lower() in ("1", "true", "yes", "on")
        # int/float: parse, fast-failing a malformed value
        return type_(float(value)) if type_ is int else type_(value)
    if type_ is str:
        return str(value)
    if type_ is bool:
        return bool(value)
    return type_(value)


def str_scalar_spec(
    label: str, *, required: bool = False, optional: bool = False
) -> ScalarSpec:
    """A string scalar field (e.g. a by-name waveform reference).

    The framework's ``IntSpec`` / ``FloatSpec`` sugar covers numerics; a string
    knob (the waveform name a node lowers via ``ml.get_waveform``) needs the bare
    ``ScalarSpec(type=str)``, so this is its explicit counterpart. ``optional``
    lets the knob lower to an omitted key when unset (a node then Fast-Fails on
    the missing waveform name with its own message).
    """
    return ScalarSpec(label=label, type=str, required=required, optional=optional)


# One field declaration: its spec + its default value. The default is the
# hardcoded value the old ``make_cfg`` carried (now the single source of truth).
NodeField = tuple[str, CfgNodeSpec, Any]


def _default_value_for(spec: CfgNodeSpec, default: Any) -> Any:
    """Build the value-tree leaf for ``spec`` from its declared ``default``."""
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec default must be a SweepValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, ScalarSpec):
        return DirectValue(default)
    raise TypeError(f"Unsupported node field spec: {type(spec).__name__}")


def flat_node_schema(fields: tuple[NodeField, ...]) -> CfgSchema:
    """A ``CfgSchema`` for a flat list of ``(key, spec, default)`` node knobs."""
    spec = CfgSectionSpec(fields={key: node_spec for key, node_spec, _ in fields})
    value = CfgSectionValue(
        fields={
            key: _default_value_for(node_spec, default)
            for key, node_spec, default in fields
        }
    )
    return CfgSchema(spec=spec, value=value)


@dataclass
class NodeCfgSchema:
    """A node's typed param SSOT — wraps a flat ``CfgSchema`` of user knobs.

    Owns the defaults (in the value tree) and the types (in the spec). The node's
    ``make_cfg`` lowers it to a flat dict; the set_node_params bridge writes leaves
    through it; a placement's stored param dict seeds it via ``with_overrides``.
    """

    schema: CfgSchema

    @property
    def keys(self) -> tuple[str, ...]:
        """The declared knob keys (the node's typed user knobs)."""
        return tuple(self.schema.spec.fields.keys())

    def set_field(self, key: str, value: Any) -> None:
        """Write one knob leaf into the value tree, fast-failing an unknown key.

        The single typed entry into the SSOT, used by two writers: the typed cfg
        form (which emits value-tree leaves — ``DirectValue`` / ``SweepValue``)
        and tests / ``with_overrides`` (which pass raw scalars). An unknown key is
        a real typo (only declared knobs are writable), so it fast-fails.

        - A ``SweepSpec`` knob accepts a ``SweepValue`` (stored verbatim).
        - A ``ScalarSpec`` knob accepts either a ``DirectValue`` / ``EvalValue``
          (stored verbatim — the form produced it) or a raw value (coerced to
          the field's declared type; a blank/None leaves the leaf unset so the
          spec default or a node's Fast-Fail guard applies).
        """
        if key not in self.schema.spec.fields:
            raise KeyError(
                f"Unknown node param {key!r}; declared: {', '.join(self.keys)}"
            )
        spec = self.schema.spec.fields[key]
        if isinstance(spec, SweepSpec):
            if not isinstance(value, SweepValue):
                raise TypeError(
                    f"Param {key!r} is a sweep; expected a SweepValue, "
                    f"got {type(value).__name__}"
                )
            self.schema.value.fields[key] = value
            return
        assert isinstance(spec, ScalarSpec)
        if isinstance(value, (DirectValue, EvalValue)):
            self.schema.value.fields[key] = value
            return
        self.schema.value.fields[key] = DirectValue(_coerce_scalar(value, spec.type))

    def with_overrides(self, params: Mapping[str, Any]) -> NodeCfgSchema:
        """Seed the value tree from a flat ``params`` dict, fast-failing unknown keys.

        Turns a ``PlacedNode``'s stored param dict into the typed schema: each
        present key overwrites its leaf (the rest keep their declared defaults).
        An unknown key fast-fails — a placement param with no declared knob is a
        contract violation, not a silent extra.
        """
        for key, value in params.items():
            self.set_field(key, value)
        return self

    def lower(
        self, ml: ModuleLibrary | None, md: MetaDict | None = None
    ) -> dict[str, Any]:
        """The lowered flat knob dict ``make_cfg`` / ``make_init_result`` read.

        Scalars lower to their value; a ``SweepSpec`` lowers to a ``SweepCfg``.
        Numeric scalar and sweep-edge ``EvalValue`` leaves resolve against
        ``md`` through the shared cfg lowering path.
        """
        return dict(self.schema.to_raw_dict(md=md, ml=ml))

    def read_knobs(self) -> dict[str, Any]:
        """The current *un-lowered* user knob values, as a JSON-friendly dict.

        Read-only projection of the value tree for an observer (the remote
        bridge's ``node.cfg``): a scalar knob reads to its plain value (or an
        eval marker), a sweep knob reads to ``{start, stop, expts}``. Unlike
        ``lower``, this does NOT build a ``SweepCfg`` (whose internal axis object
        is not JSON-serialisable) and needs no ``ModuleLibrary`` — it reports
        exactly what the user set, not the lowered run cfg.
        """
        knobs: dict[str, Any] = {}
        for key in self.keys:
            leaf = self.schema.value.fields[key]
            if isinstance(leaf, SweepValue):
                knobs[key] = {
                    "start": _knob_scalar_value(leaf.start),
                    "stop": _knob_scalar_value(leaf.stop),
                    "expts": leaf.expts,
                }
            elif isinstance(leaf, DirectValue):
                knobs[key] = leaf.value
            elif isinstance(leaf, EvalValue):
                knobs[key] = _knob_eval_value(leaf)
            else:
                # The value tree for a flat node holds only DirectValue /
                # EvalValue / SweepValue leaves; anything else is a contract breach.
                raise TypeError(
                    f"Knob {key!r} has unexpected value-tree leaf "
                    f"{type(leaf).__name__}; expected DirectValue, EvalValue, "
                    "or SweepValue"
                )
        return knobs


def _knob_scalar_value(value: object) -> object:
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    return value


def _knob_eval_value(value: EvalValue) -> dict[str, object]:
    data: dict[str, object] = {"__kind": "eval", "expr": value.expr}
    if value.resolved is not None:
        data["resolved"] = value.resolved
    if value.error is not None:
        data["error"] = value.error
    return data
