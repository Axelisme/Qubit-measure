from __future__ import annotations

import pytest
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from qick.qick_asm import get_version


def make_mock_soccfg(n_gens: int = 2, n_readouts: int = 1) -> QickConfig:
    """Build a minimal but structurally-valid QickConfig for compile-phase testing.

    Uses axis_sg_int4_v1 generators (the simplest type in QickProgramV2.gentypes)
    and axis_readout_v2 readouts.  The tProc revision is dynamically set to the
    latest value in QickProgramV2.ASM_REVISIONS so the fixture stays valid after
    QICK library upgrades.
    """

    def _gen(dac: str, ch_idx: int) -> dict:
        return {
            "type": "axis_sg_int4_v1",
            "dac": dac,
            "fs": 6553.6,
            "f_fabric": 430.08,
            "f_dds": 6553.6,
            "b_dds": 32,
            "samps_per_clk": 4,
            "maxlen": 65536,
            "complex_env": True,
            "has_dds": True,
            "tproc_ch": ch_idx,
        }

    def _readout(adc: str, ch_idx: int) -> dict:
        return {
            "type": "axis_readout_v2",
            "ro_type": "axis_readout_v2",
            "adc": adc,
            "fs": 2457.6,
            "f_output": 430.08,
            "f_dds": 1000.0,
            "b_dds": 24,
            "buf_maxlen": 8192,
            "avg_maxlen": 8192,
            "avgbuf_type": "axis_avg_buffer",
            "avgbuf_version": "v2",
            "has_edge_counter": False,
            "has_weights": False,
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
            "dacs": {d: {"fs": 6553.6} for d in dac_names},
            "adcs": {a: {"fs": 2457.6, "coupling": "AC"} for a in adc_names},
            "clk_groups": [],
        },
    }
    return QickConfig(cfg)


@pytest.fixture
def mock_soccfg() -> QickConfig:
    return make_mock_soccfg()
