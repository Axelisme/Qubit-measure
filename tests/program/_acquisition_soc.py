from zcu_tools.program.v2 import TimelineProgramV2
from tests.program._timeline import compile_modules
"""Deterministic transport for the unbound, ordinary QICK acquisition pipeline."""

import numpy as np
from qick import QickConfig
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


class AcquisitionSoc(QickConfig):
    """No simulation binding: zero buffers unless caller supplies poll_data."""

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot):
        del counter_addr, ch_list
        self._shots = total_shots
        self._reads = reads_per_shot
        self._delivered = False

    def poll_data(self, totaltime=0.1, timeout=None):
        del totaltime, timeout
        if self._delivered:
            return []
        self._delivered = True
        return [
            (
                self._shots,
                (
                    [
                        np.zeros((self._shots * n, 2), dtype=np.int64)
                        for n in self._reads
                    ],
                    {},
                ),
            )
        ]

    def get_tproc_counter(self, addr):
        del addr
        return 2**31 - 1

    def get_decimated(self, ch, address, length):
        del ch, address
        return np.zeros((length, 2), dtype=np.float64)

    def get_accumulated(self, ch, address, length):
        del ch, address
        return np.zeros((length, 2), dtype=np.int64)

    def stop_tproc(self, *args, **kwargs): ...
    def start_tproc(self, *args, **kwargs): ...
    def start_src(self, *args, **kwargs): ...
    def reload_mem(self, *args, **kwargs): ...
    def clear_tproc_counter(self, *args, **kwargs): ...
    def prepare_round(self, *args, **kwargs): ...
    def cleanup_round(self, *args, **kwargs): ...
    def load_bin_program(self, *args, **kwargs): ...
    def load_envelope(self, *args, **kwargs): ...
    def load_weights(self, *args, **kwargs): ...
    def set_nyquist(self, *args, **kwargs): ...
    def set_mixer_freq(self, *args, **kwargs): ...
    def configure_readout(self, *args, **kwargs): ...
    def config_avg(self, *args, **kwargs): ...
    def config_buf(self, *args, **kwargs): ...
    def enable_buf(self, *args, **kwargs): ...


def make_acquisition_soc():
    cfg = make_mock_soccfg()
    return AcquisitionSoc(cfg.get_cfg()), cfg


def make_chunk_acquisition(
    *,
    reps: int = 4,
    reads: int = 2,
    reps_innermost: bool = False,
    sweep: list[tuple[str, int]] | None = None,
) -> tuple[AcquisitionSoc, TimelineProgramV2]:
    """Build unbound single-ADC shots with repeated reads and optional point sweeps."""
    soc, soccfg = make_acquisition_soc()
    readouts = [
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            gain=0.1 * (index + 1),
            waveform=ConstWaveformCfg(length=1.0),
            ro_ch=0,
            ro_length=1.0,
            freq=100.0,
        ).build(f"readout_{index}")
        for index in range(reads)
    ]
    program = compile_modules(
        soccfg,
        modules=readouts,
        reps=reps,
        reps_innermost=reps_innermost,
        sweep=None if sweep is None else [(name, count) for name, count in sweep],
    )
    return soc, program
