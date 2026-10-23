"""Authored alias changes and lazy reads through the public table interface."""

import pytest
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session import aliases
from zcu_tools.gui.session.aliases import AliasRow, AliasTable, ResolvedAliasRow
from zcu_tools.resources.entry import ComponentSchema, RoleResolutionError, RoleSpec
from zcu_tools.resources.entry.schema import (
    PointDocument,
    PointGeneral,
    SetupDocument,
    SetupGeneral,
)


@pytest.mark.parametrize(
    ("required", "expected", "message"),
    [
        ({"qubit"}, {"qubit": "Q1"}, None),
        (
            {"participants"},
            None,
            "Template aliases participants are lists; bind each to one name in Aliases",
        ),
        ({"unbound", "broken", "absent"}, None, "missing: absent, broken, unbound"),
        (
            {"qubit", "participants", "absent"},
            None,
            "Template aliases participants are lists; bind each to one name in Aliases",
        ),
    ],
)
def test_scalar_alias_admission_preserves_lists_and_missing_diagnostics(
    required: set[str], expected: dict[str, str] | None, message: str | None
) -> None:
    rows = (
        ResolvedAliasRow("qubit", "Q1", "Q1"),
        ResolvedAliasRow("participants", ["Q1", "Q2"], ["Q1", "Q2"]),
        ResolvedAliasRow("unbound", "", None, "Alias is not bound"),
        ResolvedAliasRow("broken", "$qubit.missing", None, "Unreadable field"),
    )
    missing_calls: list[set[str]] = []

    def missing_message(missing: set[str]) -> str:
        missing_calls.append(missing)
        return f"missing: {', '.join(sorted(missing))}"

    assert hasattr(aliases, "scalar_alias_bindings"), (
        "Alias owner must expose scalar admission"
    )
    if message is None:
        assert (
            aliases.scalar_alias_bindings(
                rows, required, missing_message=missing_message
            )
            == expected
        )
    else:
        with pytest.raises(InvalidInputError, match=message):
            aliases.scalar_alias_bindings(
                rows, required, missing_message=missing_message
            )
    assert missing_calls == (
        [required] if "absent" in required and "participants" not in required else []
    )
    assert rows[1].resolved == ["Q1", "Q2"]


class LinkedComponent(ComponentSchema):
    frequency: float
    partner: str | int | None = None


@pytest.fixture
def point() -> PointDocument:
    # The alias reader accepts already validated snapshots, not resource handles.
    return PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="2.0",
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


def list_table(rows: tuple[AliasRow, ...]) -> AliasTable:
    """Lists are authored alias values, not malformed scalar inputs."""
    try:
        return AliasTable(rows)
    except (AttributeError, ValueError) as error:
        pytest.fail(f"Ordered alias lists must be accepted: {error}")


def test_ordered_list_resolves_links_and_scalar_reads_refuse_it(
    point: PointDocument,
) -> None:
    table = list_table(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("participants", ["$qubit.partner", "Q2"]),
        )
    )
    assert table.names(point, {}) == {"qubit": "Q1", "participants": ["R1", "Q2"]}
    with pytest.raises(RoleResolutionError, match="list"):
        table.resolve("participants", point)
    assert table.rows[1].value == ["$qubit.partner", "Q2"]


def test_empty_list_is_unbound_and_list_elements_do_not_fallback(
    point: PointDocument,
) -> None:
    table = list_table((AliasRow("participants", []),))
    assert table.names(point, {}) == {}
    with pytest.raises(RoleResolutionError, match="not bound") as empty:
        table.resolve("participants", point)
    assert empty.value.reason == "Alias is not bound"
    table.replace(
        (AliasRow("unbound", ""), AliasRow("participants", ["$unbound.partner", "Q2"]))
    )
    with pytest.raises(RoleResolutionError, match="unbound") as element:
        table.names(point, {})
    assert element.value.reason == "Alias list element is not bound"
    assert "$participants[0]" in str(element.value)
    assert "$unbound.partner" in str(element.value)


@pytest.mark.parametrize("value", [["bad.name"], ["$later.partner"], [""]])
def test_invalid_list_replacement_is_atomic(
    point: PointDocument, value: list[str]
) -> None:
    table = list_table((AliasRow("participants", ["Q2", "Q1"]),))
    with pytest.raises(InvalidInputError):
        table.replace((AliasRow("participants", value),))
    assert table.names(point, {}) == {"participants": ["Q2", "Q1"]}


def test_alias_lists_are_detached_on_install_and_read(point: PointDocument) -> None:
    value = ["Q2", "Q1"]
    table = list_table((AliasRow("participants", value),))
    value.clear()
    shown = table.rows[0].value
    if isinstance(shown, list):
        shown.clear()
    assert table.names(point, {}) == {"participants": ["Q2", "Q1"]}


def test_list_cannot_be_used_as_a_scalar_link_root(point: PointDocument) -> None:
    table = list_table(
        (AliasRow("participants", ["Q1"]), AliasRow("sense", "$participants.partner"))
    )
    with pytest.raises(RoleResolutionError, match="list"):
        table.resolve("sense", point)


def test_setup_alias_links_resolve_against_the_fixed_component_snapshot() -> None:
    setup = SetupDocument.model_construct(
        general=SetupGeneral(
            entry_id="00000000-0000-0000-0000-000000000001",
            created_at="2026-10-09T00:00:00Z",
        ),
        components={
            "R1": LinkedComponent(kind="resonator", frequency=6.0, partner="Q2"),
            "Q2": LinkedComponent(kind="qubit/fluxonium", frequency=4.0),
        },
    )
    table = AliasTable(
        (AliasRow("resonator", "R1"), AliasRow("qubit", "$resonator.partner"))
    )
    assert table.resolve("qubit", setup, kind="qubit/fluxonium") == "Q2"
    assert table.names(setup, {}) == {"resonator": "R1", "qubit": "Q2"}
    with pytest.raises(RoleResolutionError):
        table.resolve("qubit", setup, kind="resonator")


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
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.partner"),
        )
    )
    assert table.resolve("resonator", point) == "R1"
    with pytest.raises(RoleResolutionError, match="unbound"):
        table.resolve("unbound", point)


def test_unbound_alias_never_falls_back_to_a_same_named_component(
    point: PointDocument,
) -> None:
    table = AliasTable((AliasRow("qubit", ""), AliasRow("resonator", "$qubit.partner")))
    declarations = {"qubit": RoleSpec("qubit/*"), "resonator": RoleSpec("resonator")}
    assert table.names(point, declarations) == {}
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("resonator", point)
    assert caught.value.role == "resonator"
    with pytest.raises(RoleResolutionError) as unknown:
        table.resolve("missing", point)
    assert unknown.value.reason == "Unknown alias"


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
    with pytest.raises(RoleResolutionError) as unreadable:
        table.resolve("resonator", point)
    assert unreadable.value.role == "resonator"
    assert "Unreadable link" in unreadable.value.reason
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("qubit", point, kind="resonator")
    assert caught.value.role == "qubit"
    assert caught.value.required_kind == "resonator"


def test_linked_unbound_alias_names_the_empty_row_to_bind(
    point: PointDocument,
) -> None:
    table = AliasTable((AliasRow("qubit", ""), AliasRow("resonator", "$qubit.partner")))
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("resonator", point)
    message = str(caught.value)
    assert caught.value.reason == "Alias is not bound"
    assert "$resonator" in message and "$qubit" in message
    assert "bind" in message and "Aliases" in message
    assert "None" not in message and "focus=" not in message


@pytest.mark.parametrize(
    ("partner", "kind", "named"),
    [
        (None, None, "Q1.absent"),
        (7, None, "Q1.partner"),
        ("R1", "amplifier", "R1"),
    ],
    ids=["missing-field", "non-name-target", "wrong-kind"],
)
def test_link_and_kind_failures_name_the_alias_and_keep_their_reason(
    point: PointDocument, partner: int | str | None, kind: str | None, named: str
) -> None:
    link = "$qubit.absent" if partner is None else "$qubit.partner"
    point.components["Q1"] = LinkedComponent(
        kind="qubit/test", frequency=4.0, partner=partner
    )
    table = AliasTable((AliasRow("qubit", "Q1"), AliasRow("resonator", link)))
    with pytest.raises(RoleResolutionError) as caught:
        table.resolve("resonator", point, kind=kind)
    message = str(caught.value)
    assert caught.value.role == "resonator"
    assert "$resonator" in message and named in message
    if kind is not None:
        assert kind in message
    assert "None" not in message and "focus=" not in message


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


def test_capture_includes_custom_rows_without_inference_from_via(
    point: PointDocument,
) -> None:
    table = AliasTable(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "R2"),
            AliasRow("custom_sensor", "$qubit.partner"),
            AliasRow("unused", ""),
        )
    )
    declarations = {
        "qubit": RoleSpec("qubit/*"),
        "resonator": RoleSpec("resonator", "qubit.partner"),
        "absent": RoleSpec("resonator"),
    }
    assert table.names(point, declarations) == {
        "qubit": "Q1",
        "resonator": "R2",
        "custom_sensor": "R1",
    }
    assert table.with_override("absent", "R1").resolve("absent", point) == "R1"
