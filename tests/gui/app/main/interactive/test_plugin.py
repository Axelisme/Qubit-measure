"""The plugin's typed action and declared command share one commit rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from zcu_tools.gui.app.main.interactive import Action, Command, PluginDefinition
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec, validate_params
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@dataclass(frozen=True)
class Pick:
    half: float
    integer: float


def _plugin() -> tuple[
    PluginDefinition[Pick, tuple[float, float]], Action[Pick, float]
]:
    def move(state: Pick, position: float) -> Pick:
        if position >= state.integer:
            raise ValueError("half must precede integer")
        return Pick(position, state.integer)

    action = Action[Pick, float](move)
    command = Command[Pick](
        "move_half",
        (ParamSpec("position", JsonType.NUMBER),),
        lambda session, args: action.execute(session, cast(float, args["position"])),
    )

    def can_finish(state: Pick) -> None:
        if state.half == 0:
            raise FailedPreconditionError("pick a half line")

    return (
        PluginDefinition(
            "flux-pick",
            Pick(0, 10),
            (command,),
            can_finish,
            lambda state: (state.half, state.integer),
        ),
        action,
    )


def test_gui_typed_action_and_command_use_one_state_transition() -> None:
    plugin, action = _plugin()
    session = plugin.open(ManualOwnerScheduler())
    notifications: list[Pick] = []
    session.subscribe(lambda: notifications.append(session.snapshot()))

    action.execute(session, 2.0)
    params = validate_params(plugin.commands[0].params, {"position": 3.0})
    plugin.execute_command(session, "move_half", params)
    assert notifications == [Pick(2.0, 10), Pick(3.0, 10)]
    assert plugin.finish(session) == (3.0, 10)
    with pytest.raises(FailedPreconditionError):
        action.execute(session, 4.0)


def test_invalid_action_and_unavailable_finish_do_not_close_or_partially_write() -> (
    None
):
    plugin, action = _plugin()
    session = plugin.open(ManualOwnerScheduler())
    with pytest.raises(FailedPreconditionError, match="pick a half"):
        plugin.finish(session)
    with pytest.raises(ValueError, match="half must precede"):
        plugin.execute_command(session, "move_half", {"position": 11.0})
    assert session.snapshot() == Pick(0, 10)
    assert action.execute(session, 2.0) == Pick(2.0, 10)
    with pytest.raises(InvalidInputError, match="unknown interactive command"):
        plugin.execute_command(session, "not_declared", {})


def test_result_failure_leaves_terminal_gate_closed() -> None:
    plugin, _ = _plugin()

    def fail_result(state: Pick) -> tuple[float, float]:
        raise RuntimeError("result failed")

    broken = PluginDefinition(
        plugin.plugin_id,
        Pick(2, 10),
        plugin.commands,
        plugin.can_finish,
        fail_result,
    )
    session = broken.open(ManualOwnerScheduler())
    with pytest.raises(RuntimeError, match="result failed"):
        broken.finish(session)
    with pytest.raises(FailedPreconditionError):
        plugin.execute_command(session, "move_half", {"position": 3.0})


def test_duplicate_and_reserved_commands_rejected_before_open() -> None:
    plugin, _ = _plugin()
    with pytest.raises(ValueError, match="duplicate"):
        PluginDefinition(
            "flux-pick",
            Pick(1, 10),
            plugin.commands * 2,
            plugin.can_finish,
            plugin.build_result,
        )
    with pytest.raises(ValueError, match="reserved"):
        PluginDefinition(
            "flux-pick",
            Pick(1, 10),
            (Command("done", (), lambda session, args: None),),
            plugin.can_finish,
            plugin.build_result,
        )
