"""Public first-invalid reports over caller-owned drafts."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager

from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSchemaAssembler,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import CfgDraft, ResolvedReference
from zcu_tools.gui.widgets.cfg import CfgFormWidget


class EmptyCatalog:
    """No library entries; custom shapes still belong to the draft."""

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def evaluate_expression(expression: str) -> float:
    raise ValueError(f"Cannot evaluate {expression}")


def provide_options(source_id: str) -> Sequence[object]:
    return ()


@contextmanager
def reporting_form(schema: CfgSchema) -> Generator[tuple[CfgFormWidget, CfgDraft]]:
    draft = CfgDraft(
        schema,
        evaluate_expression=evaluate_expression,
        provide_options=provide_options,
        references=EmptyCatalog(),
    )
    form = CfgFormWidget()
    try:
        form.attach(draft)
        yield form, draft
    finally:
        form.detach()
        draft.close()
        form.close()
        form.deleteLater()


def test_first_invalid_reason_preserves_depth_first_sweep_paths(qapp) -> None:
    assembler = CfgSchemaAssembler()
    # Insertion order differs from lexical order; depth-first traversal wins.
    assembler.declare(
        "z_group.axis",
        SweepSpec(label="Axis"),
        SweepValue(EvalValue("start"), EvalValue("stop"), expts=3),
    )
    assembler.declare(
        "z_group.centered",
        CenteredSweepSpec(label="Centered"),
        CenteredSweepValue(EvalValue("center"), 2.0, expts=3),
    )
    assembler.declare("a_later", ScalarSpec(label="Later", type=float), None)
    with reporting_form(assembler.build()) as (form, draft):
        assert (
            form.first_invalid_reason() == "z_group.axis.start: Cannot evaluate start"
        )
        draft.set_target("z_group.axis.start", 0.0)
        assert form.first_invalid_reason() == "z_group.axis.stop: Cannot evaluate stop"
        draft.set_target("z_group.axis.stop", 1.0)
        assert (
            form.first_invalid_reason()
            == "z_group.centered.center: Cannot evaluate center"
        )
        draft.set_target("z_group.centered.center", 0.5)
        assert form.first_invalid_reason() == "a_later: invalid scalar value"
        draft.set_target("a_later", 1.0)
        assert form.first_invalid_reason() is None
        form.detach()
        assert form.first_invalid_reason() is None
    qapp.processEvents()


def test_first_invalid_reason_preserves_reference_and_scalar_details(qapp) -> None:
    shape = CfgSectionSpec(
        label="Shape",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    assembler = CfgSchemaAssembler()
    assembler.declare(
        "reference",
        ReferenceSpec(kind="test", label="Reference", allowed=[shape]),
        ReferenceValue("lost", CfgSectionValue(fields={"gain": DirectValue(None)})),
    )
    assembler.declare(
        "scalar", ScalarSpec(label="Scalar", type=float), EvalValue("scalar")
    )
    with reporting_form(assembler.build()) as (form, draft):
        # The missing library identity outranks the invalid preserved subtree.
        assert (
            form.first_invalid_reason() == "reference: missing library reference 'lost'"
        )
        draft.set_target("reference.ref", "<Custom:Shape>")
        draft.set_target("reference.gain", EvalValue("gain"))
        assert (
            form.first_invalid_reason()
            == "reference.<Custom:Shape>.gain: Cannot evaluate gain"
        )
        draft.set_target("reference.gain", 1.0)
        assert form.first_invalid_reason() == "scalar: Cannot evaluate scalar"
        draft.set_target("scalar", DirectValue(None))
        assert form.first_invalid_reason() == "scalar: invalid scalar value"
        draft.set_target("scalar", 2.0)
        assert form.first_invalid_reason() is None
    qapp.processEvents()
