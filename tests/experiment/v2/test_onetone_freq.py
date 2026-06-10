from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2.onetone import freq as freq_module
from zcu_tools.experiment.v2.onetone.freq import FreqExp


class _FakeLivePlot1D:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self) -> _FakeLivePlot1D:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def update(self, *_: object, **__: object) -> None:
        pass


def test_freq_exp_passes_stop_checker_to_acquire(monkeypatch) -> None:
    captured: dict[str, object] = {}

    cfg = MagicMock()
    cfg.rounds = 3
    cfg.modules.readout.pulse_cfg.ch = 0
    cfg.modules.readout.pulse_cfg.ro_ch = 0
    cfg.modules.readout.ro_cfg.ro_ch = 0
    cfg.sweep.freq = MagicMock()

    monkeypatch.setattr(freq_module, "setup_devices", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        freq_module, "sweep2array", lambda *_args, **_kwargs: np.arange(3.0)
    )
    monkeypatch.setattr(freq_module, "sweep2param", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(freq_module, "LivePlot1D", _FakeLivePlot1D)

    class FakeProgram:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def acquire(self, *_args: object, **kwargs: object):
            captured["stop_checkers"] = kwargs["stop_checkers"]
            return [np.ones((3, 2), dtype=np.float64)]

    monkeypatch.setattr(freq_module, "ModularProgramV2", FakeProgram)

    FreqExp().run(MagicMock(), MagicMock(), cfg)

    stop_checkers = captured["stop_checkers"]
    assert isinstance(stop_checkers, list)
    assert len(stop_checkers) == 1
    assert callable(stop_checkers[0])
