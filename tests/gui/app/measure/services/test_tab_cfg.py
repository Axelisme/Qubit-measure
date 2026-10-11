"""Tab cfg lifetime through app lookup and resource capabilities, without Qt."""

from collections.abc import Sequence

import pytest
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgObservation,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgResolution,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr


class EmptyReferences:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def resolution() -> CfgResolution:
    return CfgResolution(
        (),
        lambda expression: 0.0,
        lambda source_id: (),
        EmptyReferences(),
        lambda name: 0.0,
        validate_scalar_expr,
    )


def defaults() -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue({"value": DirectValue(1.0)}),
    )


def make_owner() -> TabCfgResources:
    return TabCfgResources(
        mutation_allowed=lambda tab_id: True,
    )


def test_headless_create_edit_and_accept_use_one_resource() -> None:
    owner = make_owner()
    resource = owner.create("tab", defaults, resolution=resolution)
    editing = owner.lookup("tab")
    editing.observe()
    changed = editing.edit((CfgEdit(("value",), 7.0),))
    assert changed.status is CfgStatus.VALID
    assert resource.observe() == changed
    accepted = owner.acceptance("tab").accept()
    assert accepted.values == {"value": 7.0}


def test_restore_skips_defaults_but_reset_uses_them() -> None:
    owner = make_owner()
    calls: list[str] = []

    def fresh() -> CfgSchema:
        calls.append("defaults")
        return defaults()

    initial = defaults()
    initial.value.fields["value"] = DirectValue(8.0)
    owner.create("restored", fresh, resolution=resolution, initial=initial)
    assert calls == []
    editing = owner.lookup("restored")
    editing.observe()
    assert owner.acceptance("restored").accept().values == {"value": 8.0}
    editing.reset()
    assert calls == ["defaults"]
    assert owner.acceptance("restored").accept().values == {"value": 1.0}


def test_retire_revokes_retained_handles_and_recreate_has_new_identity() -> None:
    owner = make_owner()
    owner.create("tab", defaults, resolution=resolution)
    old = owner.lookup("tab")
    acceptance = owner.acceptance("tab")
    old.observe()
    owner.retire("tab")
    for read in (
        old.observe,
        lambda: acceptance.accept(),
        lambda: owner.lookup("tab"),
    ):
        with pytest.raises(CfgPreconditionError) as caught:
            read()
        assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    owner.create("tab", defaults, resolution=resolution)


def test_duplicate_create_does_not_replace_existing_resource() -> None:
    owner = make_owner()
    owner.create("tab", defaults, resolution=resolution)
    before = owner.lookup("tab").observe()
    with pytest.raises(RuntimeError, match="already owns"):
        owner.create("tab", defaults, resolution=resolution)
    assert owner.lookup("tab").observe() == before


def test_failed_creation_does_not_reserve_tab_identity() -> None:
    failing = True

    def source() -> CfgResolution:
        if failing:
            raise RuntimeError("source unavailable")
        return resolution()

    owner = TabCfgResources(mutation_allowed=lambda tab_id: True)
    with pytest.raises(RuntimeError, match="source unavailable"):
        owner.create("tab", defaults, resolution=source)
    with pytest.raises(CfgPreconditionError):
        owner.lookup("tab")
    failing = False
    owner.create("tab", defaults, resolution=source)
    assert owner.lookup("tab").observe().status is CfgStatus.VALID


def test_busy_policy_is_bound_to_the_correct_tab_and_does_not_block_refresh() -> None:
    running: str | None = None
    owner = TabCfgResources(
        mutation_allowed=lambda tab_id: tab_id != running,
    )
    owner.create("running", defaults, resolution=resolution)
    owner.create("other", defaults, resolution=resolution)
    running = "running"
    busy = owner.lookup("running")
    with pytest.raises(CfgPreconditionError) as caught:
        busy.edit(())
    assert caught.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    busy.refresh()
    other = owner.lookup("other")
    other.edit((CfgEdit(("value",), 7.0),))
    assert owner.acceptance("other").accept().values == {"value": 7.0}


def test_retire_during_notification_keeps_association_and_handle_alive() -> None:
    owner = make_owner()
    owner.create("tab", defaults, resolution=resolution)
    editing = owner.lookup("tab")
    failures: list[CfgPreconditionReason] = []

    def callback(observation: CfgObservation) -> None:
        with pytest.raises(CfgPreconditionError) as caught:
            owner.retire("tab")
        failures.append(caught.value.reason)
        assert owner.lookup("tab").observe() == observation

    unsubscribe = editing.watch(callback)
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    unsubscribe()
    owner.retire("tab")
    with pytest.raises(CfgPreconditionError):
        editing.observe()
