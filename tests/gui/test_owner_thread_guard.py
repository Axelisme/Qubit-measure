from __future__ import annotations

import threading
from collections.abc import Callable
from unittest.mock import MagicMock

from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.project import ProjectInfo


def _foreign_error(callback: Callable[[], None]) -> BaseException:
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            callback()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    assert len(errors) == 1
    return errors[0]


def test_measure_state_rejects_foreign_thread_mutation() -> None:
    original = MagicMock()
    state = State(original)

    error = _foreign_error(lambda: state.set_context(MagicMock()))

    assert isinstance(error, RuntimeError)
    assert str(error) == "State mutation must run on its owner thread"
    assert state.exp_context is original


def test_autoflux_state_rejects_foreign_thread_mutation() -> None:
    state = AutoFluxDepState(MagicMock())

    error = _foreign_error(lambda: state.set_flux_values([1.0]))

    assert isinstance(error, RuntimeError)
    assert state.flux_values == []


def test_fluxdep_state_rejects_foreign_thread_mutation() -> None:
    state = FluxDepState()
    original = state.project

    error = _foreign_error(lambda: state.set_project(ProjectInfo(chip_name="new")))

    assert isinstance(error, RuntimeError)
    assert state.project is original
    assert state.version.get("project") == 0


def test_dispersive_state_rejects_foreign_thread_mutation() -> None:
    state = DispersiveState()
    original = state.project

    error = _foreign_error(lambda: state.set_project(ProjectInfo(chip_name="new")))

    assert isinstance(error, RuntimeError)
    assert state.project is original
    assert state.version.get("project") == 0
