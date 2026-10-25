"""GUI composition CLIs parse help and reject malformed arguments before assembly."""

from collections.abc import Callable

import pytest

from apps import dispersive, fluxdep, measure


@pytest.mark.parametrize("main", [measure.main, fluxdep.main, dispersive.main])
def test_help_exits_without_gui_assembly(
    main: Callable[[list[str]], int], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as failure:
        main(["--help"])
    assert failure.value.code == 0
    assert "--project-root" in capsys.readouterr().out


@pytest.mark.parametrize("main", [fluxdep.main, dispersive.main])
def test_invalid_port_exits_before_gui_assembly(
    main: Callable[[list[str]], int], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as failure:
        main(["--control-port", "not-an-integer"])
    assert failure.value.code == 2
    assert "invalid int value" in capsys.readouterr().err
