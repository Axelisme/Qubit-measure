"""Worker width at the public six-fluxonium environment factory seam."""

import pytest
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import runtime, six_fluxonium

from tests.program._profiles import profile_soccfg
from tests.simulators.six_fluxonium.config._data import six_document


def test_factory_forwards_workers_to_every_cavity_and_preserves_serial_default(
    monkeypatch,
):
    widths = []

    class ObservedEngine(Engine):
        def __init__(self, workers: int, **kwargs):
            widths.append(workers)
            super().__init__(workers, **kwargs)

    # Environment exposes no engine width; observe construction, retaining real Engine.
    monkeypatch.setattr(runtime, "Engine", ObservedEngine)
    config = parse_six_fluxonium(six_document())
    soccfg = profile_soccfg("six_fluxonium")
    for workers in (None, 1, 2):
        widths.clear()
        env = (
            six_fluxonium(config, soccfg=soccfg)
            if workers is None
            else six_fluxonium(config, soccfg=soccfg, workers=workers)
        )
        try:
            assert widths == [1 if workers is None else workers] * 6
        finally:
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()
    # Construction opens no worker pool; all numerical work stays out of this test.
    with pytest.raises(ValueError, match="workers/chunks must be positive"):
        six_fluxonium(config, soccfg=soccfg, workers=0)
