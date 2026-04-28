from __future__ import annotations

import time

import numpy as np
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from qick.qick_asm import get_version


def _build_mock_cfg(n_gens: int = 2, n_readouts: int = 1) -> dict:
    """Build a minimal but structurally-valid QickConfig dict for testing.

    Uses axis_signal_gen_v6 generators (no mixer, HAS_MIXER=False, SAMPS_PER_CLK=16)
    and axis_readout_v2 readouts.  The tProc revision is dynamically set to the
    latest value in QickProgramV2.ASM_REVISIONS so the fixture stays valid after
    QICK library upgrades.

    Clock arithmetic:
      refclk = 245.76 MHz
      gen:     fs=6144.0, fs_mult=25, fs_div=1, interpolation=1, fdds_div=1
               f_dds = fs = refclk*25/1 = 6144.0 MHz
      readout: fs=2457.6, fs_mult=10, fs_div=1, decimation=1, fdds_div=1
               f_dds = fs / decimation = 2457.6 MHz, b_dds=32
    """

    _REFCLK = 245.76

    _GEN_FS = _REFCLK * 25
    _GEN_FS_MULT = 25
    _GEN_FS_DIV = 1
    _GEN_INTERPOLATION = 1
    _GEN_FDDS_DIV = _GEN_FS_DIV * _GEN_INTERPOLATION

    _RO_FS = _REFCLK * 10
    _RO_FS_MULT = 10
    _RO_FS_DIV = 1
    _RO_DECIMATION = 1
    _RO_FDDS_DIV = _RO_FS_DIV * _RO_DECIMATION
    _RO_F_DDS = _RO_FS / _RO_DECIMATION
    _RO_F_OUTPUT = _RO_FS / (_RO_DECIMATION * 8)

    def _gen(dac: str, ch_idx: int) -> dict:
        return {
            "type": "axis_signal_gen_v6",
            "dac": dac,
            "fs": _GEN_FS,
            "fs_mult": _GEN_FS_MULT,
            "fs_div": _GEN_FS_DIV,
            "interpolation": _GEN_INTERPOLATION,
            "f_fabric": _REFCLK * 25 / 16,
            "f_dds": _GEN_FS,
            "b_dds": 32,
            "fdds_div": _GEN_FDDS_DIV,
            "samps_per_clk": 16,
            "maxlen": 65536,
            "complex_env": True,
            "has_dds": True,
            "has_mixer": False,
            "maxv": 2**15 - 2,
            "maxv_scale": 1.0,
            "b_phase": 32,
            "tproc_ch": ch_idx,
        }

    def _readout(adc: str, ch_idx: int) -> dict:
        return {
            "type": "axis_readout_v2",
            "ro_type": "axis_readout_v2",
            "adc": adc,
            "fs": _RO_FS,
            "fs_mult": _RO_FS_MULT,
            "fs_div": _RO_FS_DIV,
            "decimation": _RO_DECIMATION,
            "f_output": _RO_F_OUTPUT,
            "f_dds": _RO_F_DDS,
            "b_dds": 32,
            "fdds_div": _RO_FDDS_DIV,
            "buf_maxlen": 8192,
            "avg_maxlen": 8192,
            "avgbuf_type": "axis_avg_buffer",
            "avgbuf_version": "v2",
            "has_edge_counter": False,
            "has_weights": False,
            "has_outsel": True,
            "iq_offset": 0.0,
            "tproc_ctrl": ch_idx,
            "tproc_ch": ch_idx,
            "trigger_type": "dport",
            "trigger_port": 0,
            "trigger_bit": 0,
        }

    dac_names = [f"{i:02d}" for i in range(n_gens)]
    adc_names = [f"{i:02d}" for i in range(n_readouts)]

    return {
        "board": "ZCU216",
        "sw_version": get_version(),
        "fw_timestamp": "2025-01-01",
        "refclk_freq": 245.76,
        "extra_description": [],
        "tprocs": [
            {
                "type": "qick_processor",
                "revision": max(QickProgramV2.ASM_REVISIONS),
                "f_time": 430.08,
                "f_core": 430.08,
                "pmem_size": 4096,
                "wmem_size": 8192,
                "dmem_size": 2048,
                "dreg_qty": 16,
                "output_pins": [],
                "start_pin": "none",
                "stop_pin": "none",
            }
        ],
        "gens": [_gen(d, i) for i, d in enumerate(dac_names)],
        "readouts": [_readout(a, i) for i, a in enumerate(adc_names)],
        "rf": {
            "dacs": {
                d: {
                    "fs": _GEN_FS,
                    "fs_mult": _GEN_FS_MULT,
                    "fs_div": _GEN_FS_DIV,
                    "interpolation": _GEN_INTERPOLATION,
                    "f_fabric": _GEN_FS / 16,
                }
                for d in dac_names
            },
            "adcs": {
                a: {
                    "fs": _RO_FS,
                    "fs_mult": _RO_FS_MULT,
                    "fs_div": _RO_FS_DIV,
                    "decimation": _RO_DECIMATION,
                    "f_fabric": _RO_F_OUTPUT,
                    "coupling": "AC",
                }
                for a in adc_names
            },
            "clk_groups": [],
        },
    }


def make_mock_soccfg(n_gens: int = 2, n_readouts: int = 1) -> QickConfig:
    """Build a minimal but structurally-valid QickConfig for compile-phase testing."""
    return QickConfig(_build_mock_cfg(n_gens, n_readouts))


class MockQickSoc(QickConfig):
    """QickConfig + stubbed hardware methods that return random fake data.

    Implements just enough of the QickSoc surface to make
    AveragerProgramV2.acquire / acquire_decimated / run_rounds run end-to-end
    without real hardware. All hardware control methods are no-ops; data
    acquisition methods return random arrays of the correct shape.
    """

    _BIG_COUNT = 2**31 - 1

    def __init__(self, cfg_dict: dict) -> None:
        super().__init__(cfg_dict)
        self._cfg_dict = cfg_dict
        self._readout_state = None
        self._poll_done = False

    # --- no-op hardware control ---
    # Accept *args/**kwargs so signature drift across QICK versions doesn't
    # break the mock (e.g. config_avg gained an `edge_counting` kwarg).
    def stop_tproc(self, *args, **kwargs) -> None: ...
    def start_tproc(self, *args, **kwargs) -> None: ...
    def start_src(self, *args, **kwargs) -> None: ...
    def reload_mem(self, *args, **kwargs) -> None: ...
    def clear_tproc_counter(self, *args, **kwargs) -> None: ...
    def prepare_round(self, *args, **kwargs) -> None: ...
    def cleanup_round(self, *args, **kwargs) -> None: ...
    def load_bin_program(self, *args, **kwargs) -> None: ...
    def load_envelope(self, *args, **kwargs) -> None: ...
    def load_weights(self, *args, **kwargs) -> None: ...
    def load_pulse(self, *args, **kwargs) -> None: ...
    def load_pulses(self, *args, **kwargs) -> None: ...
    def set_nyquist(self, *args, **kwargs) -> None: ...
    def set_mixer_freq(self, *args, **kwargs) -> None: ...
    def config_mux_gen(self, *args, **kwargs) -> None: ...
    def configure_readout(self, *args, **kwargs) -> None: ...
    def config_mux_readout(self, *args, **kwargs) -> None: ...
    def config_avg(self, *args, **kwargs) -> None: ...
    def config_buf(self, *args, **kwargs) -> None: ...
    def enable_buf(self, *args, **kwargs) -> None: ...

    def get_tproc_counter(self, addr: int) -> int:
        # Return huge so `while count < total_count` exits on first iteration
        # for decimated/run_rounds paths. Accumulated path doesn't compare
        # this — it relies on poll_data() filling total_count.
        return self._BIG_COUNT

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot) -> None:
        self._readout_state = (int(total_shots), list(ch_list), list(reads_per_shot))
        self._poll_done = False

    def poll_data(self, totaltime: float = 0.1, timeout=None):
        if self._poll_done or self._readout_state is None:
            return []
        total_shots, _ch_list, reads_per_shot = self._readout_state

        data = [
            np.random.randint(
                -(2**15), 2**15, size=(total_shots * n, 2), dtype=np.int64
            )
            for n in reads_per_shot
        ]

        time.sleep(0.0001 * np.asarray(data).size)

        self._poll_done = True
        return [(total_shots, (data, {}))]

    def get_decimated(self, ch, address, length):
        return np.random.randn(length, 2).astype(float)

    def get_accumulated(self, ch, address, length):
        return np.random.randint(-(2**15), 2**15, size=(length, 2), dtype=np.int64)


def make_mock_soc(n_gens: int = 2, n_readouts: int = 1) -> MockQickSoc:
    """Build a MockQickSoc supporting end-to-end acquire() with random fake data."""
    return MockQickSoc(_build_mock_cfg(n_gens, n_readouts))
