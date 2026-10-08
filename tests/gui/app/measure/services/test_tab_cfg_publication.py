"""Cross-tab publications retain captures and isolate source failures."""

from collections.abc import Callable
from dataclasses import replace

import pytest
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgStatus,
)

from tests.gui.app.measure._cfg_fakes import (
    PublishedHost,
    cfg_resolution,
    cfg_resources,
    parameter_capture,
)


def source_state() -> State:
    return State(SessionEnv(parameters=parameter_capture(x=1.0), soc=None, soccfg=None))


def expression_schema() -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue(fields={"value": EvalValue("general.ext.x")}),
    )


def test_source_refresh_publishes_all_tabs_before_any_notification() -> None:
    state = source_state()
    capture = state.session_env.parameters

    def resolution():
        return cfg_resolution(state, capture)

    owner = cfg_resources(state)
    first = owner.create("first", expression_schema, resolution=resolution)
    second = owner.create("second", expression_schema, resolution=resolution)
    observations = []
    initial = first.observe().ref

    def observe_pair(publication) -> None:
        if publication.ref != initial:
            observations.append((first.observe(), second.observe()))

    unsubscribe = first.watch(observe_pair)
    state.set_context(replace(state.session_env, parameters=parameter_capture(x=9.0)))
    state.version.bump("context")
    publications = owner.refresh_all()
    unsubscribe()

    assert len(observations) == 1
    assert observations[0] == publications
    assert publications[0].source_basis == publications[1].source_basis
    assert publications[0].ref.revision == publications[1].ref.revision == 1
    assert first.accept(publications[0].ref.revision).values == {"value": 1.0}
    assert second.accept(publications[1].ref.revision).values == {"value": 1.0}


@pytest.mark.parametrize("batch", [False, True])
def test_notification_blocks_other_tab_commands_and_lifetime(batch: bool) -> None:
    state = source_state()
    capture = state.session_env.parameters

    def resolution():
        return cfg_resolution(state, capture)

    owner = cfg_resources(state)
    first = owner.create("first", expression_schema, resolution=resolution)
    second = owner.create("second", expression_schema, resolution=resolution)
    initial = first.observe().ref
    failures = []

    def subscriber(publication) -> None:
        if publication.ref == initial:
            return
        commands: tuple[Callable[[], object], ...] = (
            lambda: second.edit(
                second.observe().ref.revision, (CfgEdit(("value",), 5.0),)
            ),
            lambda: second.accept(second.observe().ref.revision),
            lambda: owner.retire("second"),
            lambda: owner.create("third", expression_schema, resolution=resolution),
            lambda: owner.refresh_all(),
        )
        for command in commands:
            try:
                command()
            except CfgPreconditionError as exc:
                failures.append(exc.reason)

    unsubscribe = first.watch(subscriber)
    if batch:
        owner.refresh_all()
    else:
        first.edit(initial.revision, (CfgEdit(("value",), 2.0),))
    unsubscribe()

    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 5
    second.edit(second.observe().ref.revision, (CfgEdit(("value",), 5.0),))
    assert second.accept(second.observe().ref.revision).values == {"value": 5.0}
    owner.retire("second")
    owner.create("third", expression_schema, resolution=resolution)


def test_source_fault_publishes_unavailable_for_every_tab_then_recovers(
    monkeypatch,
) -> None:
    state = source_state()
    capture = state.session_env.parameters

    def resolution():
        return cfg_resolution(state, capture)

    owner = cfg_resources(state)
    first = owner.create("first", expression_schema, resolution=resolution)
    second = owner.create("second", expression_schema, resolution=resolution)
    initial = first.observe().ref
    observations = []

    def subscriber(publication) -> None:
        if publication.ref != initial:
            observations.append((first.observe().status, second.observe().status))

    def fail_options(self):
        raise RuntimeError("Published source is unavailable")

    unsubscribe = first.watch(subscriber)
    with monkeypatch.context() as patch:
        patch.setattr(PublishedHost, "list_device_names", fail_options)
        publications = owner.refresh_all()
    unsubscribe()

    assert observations == [(CfgStatus.UNAVAILABLE, CfgStatus.UNAVAILABLE)]
    assert all(
        publication.status is CfgStatus.UNAVAILABLE for publication in publications
    )
    for cfg in (first, second):
        with pytest.raises(CfgPreconditionError):
            cfg.accept(cfg.observe().ref.revision)
    recovered = owner.refresh_all()
    assert all(publication.status is CfgStatus.VALID for publication in recovered)
    assert recovered[0].ref.revision == recovered[1].ref.revision == 2
