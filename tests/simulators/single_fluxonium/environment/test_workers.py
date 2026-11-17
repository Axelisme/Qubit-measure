"""Worker width at the public single-fluxonium environment factory seam."""

import pytest
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulators.single_fluxonium.environment import runtime, single_fluxonium

from tests.program._profiles import profile_soccfg
from tests.simulators.single_fluxonium.environment._inputs import settings


def test_factory_forwards_workers_and_preserves_serial_default(monkeypatch):
    widths = []

    class ObservedEngine(Engine):
        def __init__(self, workers: int, **kwargs):
            widths.append(workers)
            super().__init__(workers, **kwargs)

    # Environment exposes no engine width; observe construction, retaining real Engine.
    monkeypatch.setattr(runtime, "Engine", ObservedEngine)
    config = settings()
    soccfg = profile_soccfg("single_fluxonium")
    for workers in (None, 1, 2):
        env = (
            single_fluxonium(config, soccfg=soccfg)
            if workers is None
            else single_fluxonium(config, soccfg=soccfg, workers=workers)
        )
        try:
            assert widths[-1] == (1 if workers is None else workers)
        finally:
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()
    # Construction opens no worker pool; all numerical work stays out of this test.
    with pytest.raises(ValueError, match="workers/chunks must be positive"):
        single_fluxonium(config, soccfg=soccfg, workers=0)
