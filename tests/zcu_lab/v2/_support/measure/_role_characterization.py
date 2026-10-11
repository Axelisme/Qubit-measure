"""Shared role characterization context, serializer and golden location."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
)
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.gui.session.value_lookup import EmptyValueLookup, ValueLookup
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument

from zcu_lab.components import (
    ChannelSchema,
    QubitSchema,
    QubitWiringSchema,
    ResonatorSchema,
    WiringSchema,
)

GOLDEN_PATH = Path(__file__).with_name("_role_default_golden.json")

# Frozen representative literals; the synthetic channel=4 is explicitly L01.
# Empty and populated captures both keep scalar defaults deferred.
POPULATED_QUBIT = QubitSchema(
    kind="qubit/transmon",
    freq=4200.0,
    wiring=QubitWiringSchema(L01=ChannelSchema(ch=4)),
    ext={"best_ro_freq": 6300.0, "best_ro_gain": 0.22, "best_ro_length": 2.0},
)
POPULATED_RESONATOR = ResonatorSchema(
    kind="resonator",
    freq=6500.0,
    wiring=WiringSchema(ch=3, ro_ch=1),
    ext={"time_of_fly": 0.7},
)


def make_context(
    *,
    qubit: QubitSchema | None = None,
    resonator: ResonatorSchema | None = None,
    library: ModuleLibraryDocument | None = None,
    values: ValueLookup | None = None,
) -> SessionEnv:
    """Capture bound Q1/R1 facts and a native library for one role scenario.

    Omitted models keep calibration and module slots unset, not unbound aliases.
    Inputs are detached by ParameterContext/LibrarySnapshot. The optional value
    lookup is a live collaborator for Source seeds. Component kinds are installed
    only during point validation and restored before returning; no role registry,
    session selection, file I/O or hardware capability is changed.
    """
    qubit = QubitSchema(kind="qubit/transmon") if qubit is None else qubit
    resonator = ResonatorSchema(kind="resonator") if resonator is None else resonator
    before = component_registry.kinds()
    added: list[str] = []
    try:
        for kind, model in [
            (qubit.kind, QubitSchema),
            (resonator.kind, ResonatorSchema),
        ]:
            if kind not in before:
                component_registry.register(kind, model)
                added.append(kind)
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": {
                    "Q1": qubit.model_dump(mode="json", exclude_unset=True),
                    "R1": resonator.model_dump(mode="json", exclude_unset=True),
                },
            }
        )
    finally:
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before
    parameters = ParameterContext(
        point=point,
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
            if library is None
            else library
        ),
        target=SavedPoint(entry_id="role-characterization", point="working"),
        roles=RoleBinding(
            focus=None,
            declarations={
                "qubit": RoleSpec("qubit/*"),
                "resonator": RoleSpec("resonator"),
            },
            names={"qubit": "Q1", "resonator": "R1"},
        ),
    )
    return SessionEnv(
        parameters=parameters,
        soc=None,
        soccfg=None,
        values=EmptyValueLookup() if values is None else values,
    )


def serialize(node: object) -> Any:
    """Normalize a value-tree node to a JSON-comparable structure that keeps the
    DirectValue/EvalValue distinction and ref/section nesting."""
    if isinstance(node, DirectValue):
        return {"D": node.value}
    if isinstance(node, EvalValue):
        return {"E": node.expr, "r": node.resolved}
    if isinstance(node, SweepValue):
        controls = [
            serialize(value) if isinstance(value, (DirectValue, EvalValue)) else value
            for value in (node.start, node.stop, node.expts, node.step)
        ]
        if isinstance(node.step, (int, float)):
            controls[3] = round(node.step, 9)
        return {"sweep": controls}
    if isinstance(node, ReferenceValue):
        return {
            "ref": node.chosen_key,
            "ov": node.is_overridden,
            "v": serialize(node.value),
        }
    if isinstance(node, CfgSectionValue):
        return {k: serialize(v) for k, v in node.fields.items()}
    return None if node is None else repr(node)


def load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())
