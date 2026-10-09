"""Authored alias changes and lazy reads through the public table interface."""

import pytest
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.resources.entry import ComponentSchema, RoleResolutionError, RoleSpec
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class LinkedComponent(ComponentSchema):
    frequency: float
    partner: str | int | None = None


@pytest.fixture
def point() -> PointDocument:
    # The alias reader accepts already validated snapshots, not resource handles.
    return PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
        components={
            "Q1": LinkedComponent(kind="qubit/test", frequency=4.0, partner="R1"),
            "Q2": LinkedComponent(kind="qubit/test", frequency=5.0, partner="R2"),
            "R1": LinkedComponent(kind="resonator", frequency=6.0, partner="A1"),
            "R2": LinkedComponent(kind="resonator", frequency=7.0, partner="A2"),
            "A1": LinkedComponent(kind="amplifier", frequency=8.0),
            "A2": LinkedComponent(kind="amplifier", frequency=9.0),
            "qubit": LinkedComponent(kind="qubit/test", frequency=3.0),
        },
    )


def test_links_follow_changed_earlier_rows_without_expanding_authored_values(
    point: PointDocument,
) -> None:
    table = AliasTable(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.partner"),
            AliasRow("amplifier", "$resonator.partner"),
        )
    )
    assert table.resolve("amplifier", point) == "A1"
    table.replace((AliasRow("qubit", "Q2"), *table.rows[1:]))
    assert table.resolve("resonator", point, kind="resonator") == "R2"
    assert table.resolve("amplifier", point) == "A2"
    assert table.rows[1].value == "$qubit.partner"
    assert table.rows[2].value == "$resonator.partner"


@pytest.mark.parametrize(
    "rows",
    [
        (AliasRow("qubit", "$qubit.partner"),),
        (AliasRow("resonator", "$qubit.partner"), AliasRow("qubit", "Q1")),
        (AliasRow("qubit", "Q1"), AliasRow("qubit", "Q2")),
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$missing.partner")),
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit")),
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit..partner")),
        (AliasRow("_private", "Q1"),),
        (AliasRow("qubit", "Q1.partner"),),
    ],
)
def test_invalid_replacements_preserve_the_complete_old_table(
    point: PointDocument, rows: tuple[AliasRow, ...]
) -> None:
    table = AliasTable((AliasRow("qubit", "Q1"),))
    with pytest.raises(InvalidInputError) as caught:
        table.replace(rows)
    assert caught.value.reason_code == "alias_invalid"
    assert table.rows == (AliasRow("qubit", "Q1"),)
    assert table.resolve("qubit", point) == "Q1"


def test_unused_unbound_or_bad_rows_do_not_prevent_other_alias_reads(
    point: PointDocument,
) -> None:
    table = AliasTable(
        (
            AliasRow("unbound", ""),
            AliasRow("broken", "missing"),
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.partner"),
        )
    )
    assert table.resolve("resonator", point) == "R1"
    with pytest.raises(RoleResolutionError, match="unbound"):
        table.resolve("unbound", point)
    with pytest.raises(RoleResolutionError, match="broken"):
        table.resolve("broken", point)


def test_unbound_alias_never_falls_back_to_a_same_named_component(
    point: PointDocument,
) -> None:
    table = AliasTable((AliasRow("qubit", ""), AliasRow("resonator", "$qubit.partner")))
    declarations = {"qubit": RoleSpec("qubit/*"), "resonator": RoleSpec("resonator")}
    assert table.names(point, declarations) == {}
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("resonator", point)
    assert caught.value.role == "resonator"
    with pytest.raises(RoleResolutionError, match="Unknown alias"):
        table.resolve("missing", point)


@pytest.mark.parametrize("partner", [None, 7, "missing"])
def test_invalid_link_targets_are_explicit_not_unbound(
    point: PointDocument, partner: str | int | None
) -> None:
    point.components["Q1"] = LinkedComponent(
        kind="qubit/test", frequency=4.0, partner=partner
    )
    table = AliasTable(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.partner"))
    )
    with pytest.raises(RoleResolutionError) as caught:
        table.names(point, {"resonator": RoleSpec("resonator")})
    assert caught.value.role == "resonator"


def test_missing_link_field_and_wrong_kind_report_the_requested_alias(
    point: PointDocument,
) -> None:
    table = AliasTable(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.absent"))
    )
    with pytest.raises(RoleResolutionError, match="resonator.*Unreadable link"):
        table.resolve("resonator", point)
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("qubit", point, kind="resonator")
    assert caught.value.role == "qubit"
    assert caught.value.required_kind == "resonator"


def test_one_use_override_rebinds_dependent_rows_without_mutating_the_table(
    point: PointDocument,
) -> None:
    table = AliasTable((AliasRow("qubit", ""), AliasRow("resonator", "$qubit.partner")))
    temporary = table.with_override("qubit", "Q2")
    assert temporary.names(
        point, {"qubit": RoleSpec("qubit/*"), "resonator": RoleSpec("resonator")}
    ) == {"qubit": "Q2", "resonator": "R2"}
    assert table.names(point, {"qubit": RoleSpec("qubit/*")}) == {}
    assert table.rows[0].value == ""
    temporary.replace((AliasRow("qubit", "Q1"), *temporary.rows[1:]))
    assert table.rows[0].value == ""


@pytest.mark.parametrize("target", ["", "Q1.partner", " Q1", "_private"])
def test_invalid_override_is_rejected_without_changing_the_table(target: str) -> None:
    table = AliasTable((AliasRow("qubit", "Q1"),))
    with pytest.raises(InvalidInputError):
        table.with_override("qubit", target)
    assert table.rows == (AliasRow("qubit", "Q1"),)


def test_capture_uses_only_declared_rows_and_does_not_infer_from_via(
    point: PointDocument,
) -> None:
    table = AliasTable((AliasRow("qubit", "Q1"), AliasRow("resonator", "R2")))
    declarations = {
        "qubit": RoleSpec("qubit/*"),
        "resonator": RoleSpec("resonator", "qubit.partner"),
        "absent": RoleSpec("resonator"),
    }
    assert table.names(point, declarations) == {"qubit": "Q1", "resonator": "R2"}
    assert table.with_override("absent", "R1").resolve("absent", point) == "R1"
