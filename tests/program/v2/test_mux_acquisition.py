"""Public builder/acquire integration with fixed mux, not physical simulation."""

import numpy as np
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modules import readout

from tests.program.v2._mux_support import AcquisitionSoc, mux_soccfg, native_mux


def test_builder_converter_uses_initialized_copy_and_normalized_acquire():
    soccfg = mux_soccfg()
    soc = AcquisitionSoc(soccfg.get_cfg())
    original = native_mux().build("mux")
    buffer = SignalBuffer((2,), dtype=np.complex128)
    stop = StopSignal()
    with Schedule(ProgramV2Cfg(reps=3, rounds=2), buffer, stop=stop) as sched:
        builder = sched.prog_builder(soc, soccfg).add(original)
        prog = builder.build()
        module = next(m for m in prog.modules if m.name == "mux")
        assert isinstance(module, readout.MuxReadout)
        assert module is not original

        def convert(raw):
            iq = module.collect_iq(prog, raw, read_index=0)
            return iq[..., 0] + 1j * iq[..., 1]

        result = builder.run_program(prog, raw2signal_fn=convert)
    stop.raise_if_error()
    np.testing.assert_array_equal(result, [5 - 5j, 2 - 2j])
    np.testing.assert_array_equal(buffer.data, result)
