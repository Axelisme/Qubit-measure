"""``CfgBuilder`` — fluent assembly of an adapter's default value tree.

An adapter's ``make_default_value(ctx)`` must return a structurally-complete,
spec-compliant value tree (ADR-0021/0022). Hand-building one means writing a
nested ``CfgSectionValue(fields={...})`` literal and threading the per-role L2
factories through it by hand. ``CfgBuilder`` replaces that with a flat,
path-addressed fluent API:

    def make_default_value(self, ctx):
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=1.0)
            .role("modules.readout", "pulse_readout")        # mount a library/blank ref
            .role("modules.reset", "reset", optional=True)   # library miss -> None
            .role("modules.qub_pulse", "qub_probe")
            .set("modules.qub_pulse.gain", 0.05)             # scalar override
            .set_sweep("sweep.freq", proper_qub_freq_range(ctx, 301))
            .build()
        )

Layering (ADR-0023):

- The builder starts from the **L1** blank tree (``make_default_value(spec)``),
  so completeness is guaranteed by construction — every method only *overwrites*
  an already-present node, it never adds or removes keys.
- ``.role()`` dispatches to the **L2** ``make_<role>_default`` /
  ``make_<role>_ref_default`` factories via the shared ``ROLE_FACTORIES`` table.
  The builder lives in the domain layer precisely so it may know ``ctx`` + roles
  + library lookup; that domain knowledge must not leak into the framework-layer
  ``CfgSectionValue`` data container (whose ``with_field`` stays scalar-only).
- The builder does **zero locking** (ADR-0009): locking a field is a spec-layer
  decision, declared in ``cfg_spec()`` via ``lock_literal``. The builder only
  ever writes the value tree.
- ``.build()`` does **not** validate — validation stays at the cfg boundary
  (``BaseAdapter.make_default_cfg`` / ``CfgSchema.to_raw_dict``).

Every method Fast-Fails on a bad path / kind / type mismatch, pointing at the
offending call rather than surfacing as a later lowering error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import Self

from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarLeafInput,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    make_default_value,
)

from .defaults import ROLE_FACTORIES

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgNodeSpec, ExpContext

# A node the builder mounts at a path: a whole ref node, a sweep, or None (a
# disabled optional ref). Scalars never go through _mount_node — they are set
# via CfgSectionValue.with_field (scalar-leaf only).
_MountNode = Union[ModuleRefValue, WaveformRefValue, SweepValue, None]


class CfgBuilder:
    """Fluent, path-addressed assembler for an adapter default value tree.

    Holds ``ctx`` + the static ``spec`` + an in-place-mutated value tree seeded
    from the L1 blank. One-shot: after ``build()`` any further mutation raises.
    """

    def __init__(self, ctx: ExpContext, spec: CfgSectionSpec) -> None:
        self._ctx = ctx
        self._spec = spec
        self._value: CfgSectionValue = make_default_value(spec)
        self._built = False

    # -- public fluent verbs ------------------------------------------------

    def scalars(self, **values: ScalarLeafInput) -> Self:
        """Set top-level scalar leaves by field name.

        Each key must be a top-level field of the root spec (Fast-Fail on a
        typo). Reuses ``with_field`` (scalar-leaf only); raw scalars are wrapped
        in ``DirectValue``.
        """
        self._check_mutable()
        for key, value in values.items():
            if key not in self._spec.fields:
                raise RuntimeError(
                    f"CfgBuilder.scalars: unknown top-level field {key!r} "
                    f"(available: {', '.join(self._spec.fields)})"
                )
            self._value.with_field(key, value)
        return self

    def set(self, path: str, value: ScalarLeafInput) -> Self:
        """Override the scalar leaf at dotted ``path`` (Fast-Fail on a bad path).

        A spec-aware pre-check rejects a path that does not resolve to a leaf,
        before delegating to ``with_field``'s value-tree descent.
        """
        self._check_mutable()
        if not _spec_path_exists(self._spec, _split(path)):
            raise RuntimeError(
                f"CfgBuilder.set: path {path!r} does not resolve to a field in the spec"
            )
        self._value.with_field(path, value)
        return self

    def role(
        self,
        path: str,
        role_id: str,
        *,
        optional: bool = False,
        prefer_blank: bool = False,
    ) -> Self:
        """Mount a module/waveform ref at ``path`` from the L2 role factories.

        - plain ``.role(path, role_id)`` → the role's *ref* factory (library
          aware), or its *blank* factory if the role has no ref variant.
        - ``prefer_blank=True`` → force the *blank* factory (never adopt a
          library entry; e.g. a readout that must stay inline).
        - ``optional=True`` → the *ref* factory's optional path (library miss →
          ``None``). Requires a ref variant and an ``optional`` spec ref.
        """
        self._check_mutable()
        parts = _split(path)
        ref_spec = self._resolve_ref_spec(parts, path)

        spec = ROLE_FACTORIES.get(role_id)
        if spec is None:
            raise RuntimeError(
                f"CfgBuilder.role: unknown role_id {role_id!r} "
                f"(available: {', '.join(sorted(ROLE_FACTORIES))})"
            )

        if optional:
            if not ref_spec.optional:
                raise RuntimeError(
                    f"CfgBuilder.role: optional=True at {path!r} but the spec ref "
                    "is not optional (a required ref cannot be disabled)"
                )
            if spec.ref is None:
                raise RuntimeError(
                    f"CfgBuilder.role: role {role_id!r} has no library-aware "
                    "(ref) factory, so optional=True (library-miss → None) is "
                    "unsupported"
                )
            node = spec.ref(self._ctx, optional=True)
        elif prefer_blank or spec.ref is None:
            node = spec.blank(self._ctx)
        else:
            node = spec.ref(self._ctx)

        self._check_ref_kind(ref_spec, node, path, role_id)
        self._mount_node(parts, node)
        return self

    def sweep(self, path: str, start: float, stop: float, expts: int) -> Self:
        """Mount a literal-valued sweep at ``path``.

        ``start``/``stop`` must be plain numbers; a sweep with an ``EvalValue``
        edge (md-linked range, e.g. ``q_f ± span``) goes through ``set_sweep``
        with a pre-built ``SweepValue``. This keeps the simple verb unambiguous.
        """
        self._check_mutable()
        for name, edge in (("start", start), ("stop", stop)):
            if not isinstance(edge, (int, float)) or isinstance(edge, bool):
                raise RuntimeError(
                    f"CfgBuilder.sweep: {name} must be a plain number; an "
                    "EvalValue edge requires set_sweep(path, SweepValue(...))"
                )
        self.set_sweep(
            path, SweepValue(start=float(start), stop=float(stop), expts=expts)
        )
        return self

    def set_sweep(self, path: str, sweep: SweepValue) -> Self:
        """Mount a pre-built ``SweepValue`` at ``path`` (edges may be EvalValue)."""
        self._check_mutable()
        parts = _split(path)
        leaf_spec = self._resolve_leaf_spec(parts, path)
        if not isinstance(leaf_spec, SweepSpec):
            raise RuntimeError(
                f"CfgBuilder.set_sweep: spec at {path!r} is "
                f"{type(leaf_spec).__name__}, not a SweepSpec"
            )
        self._mount_node(parts, sweep)
        return self

    def build(self) -> CfgSectionValue:
        """Return the assembled value tree; the builder is spent afterwards."""
        self._check_mutable()
        self._built = True
        return self._value

    # -- internals ----------------------------------------------------------

    def _check_mutable(self) -> None:
        if self._built:
            raise RuntimeError("CfgBuilder is already built; create a new one")

    def _mount_node(self, parts: list[str], node: _MountNode) -> None:
        """Assign a whole node (ref / sweep / None) at the value-tree path.

        Descends ``parts[:-1]`` through the value tree exactly like
        ``with_field`` does, then assigns the node at the final segment. Unlike
        ``with_field`` this assigns a non-scalar node, so it lives on the builder
        rather than widening the framework-layer ``CfgSectionValue`` contract.
        """
        node_value: CfgSectionValue = self._value
        for seg in parts[:-1]:
            child = node_value.fields.get(seg)
            if isinstance(child, (ModuleRefValue, WaveformRefValue)):
                node_value = child.value
            elif isinstance(child, CfgSectionValue):
                node_value = child
            else:
                raise RuntimeError(
                    f"CfgBuilder: cannot descend into {type(child).__name__} "
                    f"at segment {seg!r}"
                )
        node_value.fields[parts[-1]] = node

    def _resolve_leaf_spec(self, parts: list[str], path: str) -> CfgNodeSpec:
        """Descend the spec tree to the spec node at ``path`` (Fast-Fail)."""
        spec: CfgNodeSpec = self._spec
        for i, seg in enumerate(parts):
            if not isinstance(spec, CfgSectionSpec):
                raise RuntimeError(
                    f"CfgBuilder: cannot descend into {type(spec).__name__} "
                    f"at segment {seg!r} of {path!r}"
                )
            child = spec.fields.get(seg)
            if child is None:
                raise RuntimeError(
                    f"CfgBuilder: segment {seg!r} of {path!r} not found "
                    f"(available: {', '.join(spec.fields)})"
                )
            spec = child
            del i
        return spec

    def _resolve_ref_spec(
        self, parts: list[str], path: str
    ) -> Union[ModuleRefSpec, WaveformRefSpec]:
        leaf_spec = self._resolve_leaf_spec(parts, path)
        if not isinstance(leaf_spec, (ModuleRefSpec, WaveformRefSpec)):
            raise RuntimeError(
                f"CfgBuilder.role: spec at {path!r} is "
                f"{type(leaf_spec).__name__}, not a ModuleRefSpec/WaveformRefSpec"
            )
        return leaf_spec

    @staticmethod
    def _check_ref_kind(
        ref_spec: Union[ModuleRefSpec, WaveformRefSpec],
        node: _MountNode,
        path: str,
        role_id: str,
    ) -> None:
        if node is None:
            return  # optional miss → disabled ref (ADR-0021)
        if isinstance(ref_spec, ModuleRefSpec) and not isinstance(node, ModuleRefValue):
            raise RuntimeError(
                f"CfgBuilder.role: spec at {path!r} expects a module ref but role "
                f"{role_id!r} produced {type(node).__name__}"
            )
        if isinstance(ref_spec, WaveformRefSpec) and not isinstance(
            node, WaveformRefValue
        ):
            raise RuntimeError(
                f"CfgBuilder.role: spec at {path!r} expects a waveform ref but role "
                f"{role_id!r} produced {type(node).__name__}"
            )


def _split(path: str) -> list[str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise RuntimeError("CfgBuilder: path must not be empty")
    return parts


def _spec_path_exists(spec: CfgSectionSpec, parts: list[str]) -> bool:
    """True if ``parts`` resolve to a leaf within ``spec`` (duck-typing across
    ModuleRefSpec.allowed shapes, mirroring the spec-layer descent)."""
    head, rest = parts[0], parts[1:]
    child: Optional[CfgNodeSpec] = spec.fields.get(head)
    if child is None:
        return False
    if not rest:
        return True
    if isinstance(child, CfgSectionSpec):
        return _spec_path_exists(child, rest)
    if isinstance(child, (ModuleRefSpec, WaveformRefSpec)):
        return any(_spec_path_exists(shape, rest) for shape in child.allowed)
    return False
