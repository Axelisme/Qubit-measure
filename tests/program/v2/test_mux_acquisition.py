"""Public builder/acquire integration with fixed mux, not physical simulation."""

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import readout
from zcu_tools.program.v2.modules.control import Repeat

from tests.program.v2._mux_support import AcquisitionSoc, mux_soccfg, native_mux


class ShotSentinelSoc(AcquisitionSoc):
    """Expose distinct round/channel/shot/read values through real acquisition."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rounds_started = 0

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot):
        self.rounds_started += 1
        super().start_readout(total_shots, counter_addr, ch_list, reads_per_shot)

    def poll_data(self, totaltime=0.1, timeout=None):
        if self.delivered:
            return []
        self.delivered = True
        data = []
        for ch, n in zip(self.channels, self.reads, strict=True):
            values = self.rounds_started * 1000 + ch * 100 + np.arange(self.shots * n)
            length = self.us2cycles(0.413, ro_ch=ch)
            data.append(np.stack([values, -values], axis=-1) * length)
        return [(self.shots, (data, {}))]


def test_raw_getter_uses_last_acquired_round_with_per_adc_length_normalization():
    soccfg = mux_soccfg()
    # Distinct ADC clocks give distinct rounded integration lengths. DC offset
    # remains in raw IQ; only averaged acquisition applies QICK offset removal.
    soccfg["readouts"][1]["f_output"] = 153.6
    soccfg["readouts"][1]["iq_offset"] = 7.0
    soc = ShotSentinelSoc(soccfg.get_cfg())
    module = native_mux().build("mux")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=3, rounds=2),
        [Repeat("reads", 2).add_content(module)],
        sweep=[("preparation", 4)],
    )
    with pytest.raises(ValueError, match="mux.*raw"):
        module.collect_raw_iq(prog)
    # Repeated trigger execution needs an explicit QICK runtime read count.
    prog.set_reads_per_shot(2)
    averaged = prog.acquire(soc, progress=False)
    raw = prog.get_raw()
    iq = module.collect_raw_iq(prog)
    assert iq.dtype == np.float64
    assert iq.shape == (3, 4, 2, 2, 2)
    assert prog.ro_chs[4]["length"] != prog.ro_chs[1]["length"]
    np.testing.assert_array_equal(iq[0, 0, 0], [[2400, -2400], [2100, -2100]])
    np.testing.assert_array_equal(iq[2, 3, 1], [[2423, -2423], [2123, -2123]])
    np.testing.assert_array_equal(
        module.collect_raw_iq(prog, read_index=1), iq[:, :, 1]
    )
    np.testing.assert_array_equal(module.collect_raw_iq(prog), iq)
    assert prog.get_raw() is raw
    assert soc.rounds_started == 2
    # Averaged collection still uses all rounds and reps and QICK offset removal.
    np.testing.assert_array_equal(
        module.collect_iq(prog, averaged, read_index=1)[0],
        [[1909, -1909], [1602, -1616]],
    )


def test_builder_converter_uses_initialized_copy_and_normalized_acquire():
    soccfg = mux_soccfg()
    soc = AcquisitionSoc(soccfg.get_cfg())
    original = native_mux().build("mux")
    buffer = SignalBuffer((2,), dtype=np.complex128)
    stop = StopSignal()
    result = buffer.data
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
