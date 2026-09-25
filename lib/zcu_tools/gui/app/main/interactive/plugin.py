"""Qt-free plugin actions, commands and terminal result contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.param_spec import ParamSpec
from zcu_tools.gui.session.ports import OwnerScheduler

from .session import Session

S = TypeVar("S")
P = TypeVar("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class Action(Generic[S, P]):
    calculate: Callable[[S, P], S]

    def execute(self, session: Session[S], params: P) -> S:
        """The GUI and command handler call this same validation/commit path."""
        return session.commit(lambda state: self.calculate(state, params))


@dataclass(frozen=True, slots=True)
class Command(Generic[S]):
    """A stable command id, its shared wire ParamSpec, and a typed action bridge.

    The GUI invokes the typed Action directly. Remote validates the request with
    gui.remote.param_spec.validate_params before calling the bridge. Both call
    the same Action; the domain policy never depends on a wire codec.
    """

    name: str
    params: tuple[ParamSpec, ...]
    execute: Callable[[Session[S], Mapping[str, object]], object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("interactive command name must be non-empty")
        if self.name == "done":
            raise ValueError("interactive command 'done' is reserved")
        names = [param.name for param in self.params]
        if len(names) != len(set(names)):
            raise ValueError(
                f"duplicate parameters for interactive command {self.name!r}"
            )
        if not callable(self.execute):
            raise TypeError("interactive command execute must be callable")


@dataclass(frozen=True, slots=True)
class PluginDefinition(Generic[S, R]):
    plugin_id: str
    seed: S
    commands: tuple[Command[S], ...]
    can_finish: Callable[[S], None]
    build_result: Callable[[S], R]

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("interactive plugin id must be non-empty")
        names = [command.name for command in self.commands]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate interactive commands for {self.plugin_id!r}")
        if not callable(self.can_finish) or not callable(self.build_result):
            raise TypeError("interactive plugin terminal callbacks must be callable")

    def open(self, owner: OwnerScheduler) -> Session[S]:
        """Open one independent framework-owned session for these captured inputs."""
        return Session(self.seed, owner)

    def execute_command(
        self, session: Session[S], name: str, params: Mapping[str, object]
    ) -> object:
        """Dispatch already-decoded params; the command invokes the shared action."""
        session.ensure_input_open()
        for command in self.commands:
            if command.name == name:
                return command.execute(session, params)
        raise InvalidInputError(f"unknown interactive command {name!r}")

    def finish(self, session: Session[S]) -> R:
        """Validate before closing input; result failure leaves the gate terminal."""
        session.ensure_input_open()
        committed = session.snapshot()
        self.can_finish(committed)
        session.close_input()
        return self.build_result(committed)
