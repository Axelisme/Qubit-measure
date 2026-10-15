from __future__ import annotations

from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from qick.qick_asm import get_version


def _build_mock_cfg(n_gens: int = 2, n_readouts: int = 1) -> dict[str, object]:
    """Build a minimal but structurally-valid QickConfig dict for testing.

    Uses axis_signal_gen_v6 generators (no mixer, HAS_MIXER=False, SAMPS_PER_CLK=16)
    and axis_readout_v2 readouts.  The tProc revision is dynamically set to the
    latest value in QickProgramV2.ASM_REVISIONS so the fixture stays valid after
    QICK library upgrades.

    Clock arithmetic:
      refclk = 245.76 MHz
      gen:     fs=12288.0, fs_mult=50, fs_div=1, interpolation=1, fdds_div=1
               f_dds = fs = refclk*50/1 = 12288.0 MHz
      readout: fs=2457.6, fs_mult=10, fs_div=1, decimation=1, fdds_div=1
               f_dds = fs / decimation = 2457.6 MHz, b_dds=32

    The gen f_dds is deliberately high (12288 MHz, a clean 50*refclk).  With
    interpolation==1 QICK applies no Nyquist check, so a played tone folds by
    ``f mod f_dds`` on the analyzer's absolute frequency axis (not an fs/2
    reflection).  Pushing f_dds to 12288 keeps the whole fluxonium working set —
    f01 (~4 GHz), the dressed resonator (~7 GHz), and 6 GHz-class readouts with
    several-hundred-MHz sweeps — comfortably below f_dds, so absolute frequencies
    are reported un-folded.  f_dds is NOT a free parameter: it must stay
    self-consistent with ``refclk * fs_mult / (fs_div * interpolation)`` or
    QICK's freq2reg/reg2freq quantisation desyncs from f_dds.
    """

    _REFCLK = 245.76

    _GEN_FS = _REFCLK * 50
    _GEN_FS_MULT = 50
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

    def _gen(dac: str, ch_idx: int) -> dict[str, object]:
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

    def _readout(adc: str, ch_idx: int) -> dict[str, object]:
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
    """Offline compile profile: v6 generators, v2 ADCs, tProc v2, no mixers/mux.

    Only channel counts are configurable. Generator clock is 12288 MHz with
    16 samples/clock; ADC output is 307.2 MHz. DDS and phase use 32 bits.
    Wiring belongs to SimulationSoc, not this compile-only helper. No hardware
    or acquisition source is constructed. Negative counts fail with ValueError.
    """
    """Build a minimal but structurally-valid QickConfig for compile-phase testing."""
    if (
        isinstance(n_gens, bool)
        or isinstance(n_readouts, bool)
        or n_gens < 0
        or n_readouts < 0
    ):
        raise ValueError("channel counts must be nonnegative integers")
    return QickConfig(_build_mock_cfg(n_gens, n_readouts))
