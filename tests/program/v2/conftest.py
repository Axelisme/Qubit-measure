from __future__ import annotations

import pytest
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from qick.qick_asm import get_version


def make_mock_soccfg(n_gens: int = 2, n_readouts: int = 1) -> QickConfig:
    """Build a minimal but structurally-valid QickConfig for compile-phase testing.

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

    # refclk = 245.76 MHz (standard ZCU216 reference)
    _REFCLK = 245.76

    # gen: fs_mult=25, fs_div=1 → fs = 245.76*25 = 6144.0, interpolation=1 → f_dds=6144.0
    _GEN_FS = _REFCLK * 25  # 6144.0 MHz
    _GEN_FS_MULT = 25
    _GEN_FS_DIV = 1
    _GEN_INTERPOLATION = 1
    _GEN_FDDS_DIV = _GEN_FS_DIV * _GEN_INTERPOLATION  # 1

    # readout: fs_mult=10, fs_div=1 → fs = 245.76*10 = 2457.6, decimation=1 → f_dds=2457.6
    _RO_FS = _REFCLK * 10  # 2457.6 MHz
    _RO_FS_MULT = 10
    _RO_FS_DIV = 1
    _RO_DECIMATION = 1
    _RO_FDDS_DIV = _RO_FS_DIV * _RO_DECIMATION  # 1
    _RO_F_DDS = _RO_FS / _RO_DECIMATION  # 2457.6 MHz
    _RO_F_OUTPUT = _RO_FS / (_RO_DECIMATION * 8)  # AxisReadoutV2 DOWNSAMPLING=8 → 307.2 MHz

    def _gen(dac: str, ch_idx: int) -> dict:
        return {
            "type": "axis_signal_gen_v6",
            "dac": dac,
            "fs": _GEN_FS,
            "fs_mult": _GEN_FS_MULT,
            "fs_div": _GEN_FS_DIV,
            "interpolation": _GEN_INTERPOLATION,
            "f_fabric": _REFCLK * 25 / 16,  # f_fabric ≈ 384.0 MHz (fs/samps_per_clk)
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

    cfg = {
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
            "dacs": {d: {"fs": _GEN_FS, "fs_mult": _GEN_FS_MULT, "fs_div": _GEN_FS_DIV, "interpolation": _GEN_INTERPOLATION, "f_fabric": _GEN_FS / 16} for d in dac_names},
            "adcs": {a: {"fs": _RO_FS, "fs_mult": _RO_FS_MULT, "fs_div": _RO_FS_DIV, "decimation": _RO_DECIMATION, "f_fabric": _RO_F_OUTPUT, "coupling": "AC"} for a in adc_names},
            "clk_groups": [],
        },
    }
    return QickConfig(cfg)


@pytest.fixture
def mock_soccfg() -> QickConfig:
    return make_mock_soccfg()
