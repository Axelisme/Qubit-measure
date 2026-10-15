"""Reject real Mux modules before the bound environment generates any data.

The program uses the compile-only mixmux/PFB profile. The environment retains
its fixed profile: this negative seam does not make either profile interchangeable
or provide mux physics.
"""

import pytest
from zcu_tools.core.simulation import RoundRequest, RoundSignals
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.control import Repeat

from tests.program.v2._mux_support import mux_soccfg, native_mux


@pytest.mark.parametrize("method", ["acquire", "acquire_decimated"])
@pytest.mark.parametrize("placement", ["direct", "repeat"])
def test_simulation_acquire_rejects_mux_with_type_and_name(method, placement):
    mux = native_mux().build("mux_probe")
    module = Repeat("container", 2).add_content(mux) if placement == "repeat" else mux
    prog = ModularProgramV2(mux_soccfg(), ProgramV2Cfg(reps=2), modules=[module])
    env = white_noise(n_gens=2, n_readouts=9, routes=((0, 4), (0, 1)))
    calls: list[RoundRequest] = []

    def execute(request: RoundRequest) -> RoundSignals:
        calls.append(request)
        raise AssertionError("unsupported mux must fail before backend execution")

    env.soc.binding.execute = execute
    try:
        with pytest.raises(ValueError, match="unsupported") as caught:
            getattr(prog, method)(env.soc, progress=False)
        message = str(caught.value)
        assert "unsupported" in message.lower()
        assert type(mux).__name__ in message
        assert "mux_probe" in message
        assert calls == []
    finally:
        env.devices.close_all_devices()
