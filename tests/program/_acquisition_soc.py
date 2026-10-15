"""Deterministic transport for the unbound, ordinary QICK acquisition pipeline."""

import numpy as np
from qick import QickConfig
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


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
