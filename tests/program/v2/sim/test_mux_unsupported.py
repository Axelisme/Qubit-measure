"""Mux has no physical simulator; public acquisition must reject it.

The existing simulation factory is used only for rejection. When it is retired,
move this collaborator to the bound environment and keep the same assertions.
"""

import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.sim import SimParams

from tests.program.v2._mux_support import RecordingProgram, mux_soccfg, native_mux


@pytest.mark.parametrize("method", ["acquire", "acquire_decimated"])
def test_simulation_acquire_rejects_mux_with_type_and_name(method):
    prog = RecordingProgram(mux_soccfg(), [native_mux().build("mux_probe")])
    soc, _ = make_mock_soc(sim=SimParams())
    with pytest.raises(ValueError) as caught:
        getattr(prog, method)(soc, progress=False)
    message = str(caught.value)
    assert "unsupported" in message.lower()
    assert "MuxReadout" in message
    assert "mux_probe" in message
