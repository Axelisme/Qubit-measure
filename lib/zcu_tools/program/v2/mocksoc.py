from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

# Default poll pacing for the white-noise mock path (no SimParams).
# When a SimParams is attached, its poll_latency field overrides this value.
# Set to 0 to skip the sleep entirely (avoids the syscall overhead of sleep(0)).
_DEFAULT_POLL_LATENCY: float = 1e-7

import numpy as np
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from qick.qick_asm import get_version

from .sim._profiling import PerfStats, elapsed_ms, perf_now

if TYPE_CHECKING:
    from .sim import SimParams
    from .sim.engine import SimEngine

logger = logging.getLogger(__name__)
_POLL_DATA_PERF = PerfStats("worker.sim.poll_data", logger, slow_ms=50.0)


def _build_mock_cfg(n_gens: int = 2, n_readouts: int = 1) -> dict:
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

    def __init__(self, cfg_dict: dict, sim: SimParams | None = None) -> None:
        super().__init__(cfg_dict)
        self._cfg_dict = cfg_dict
        self._readout_state = None
        self._poll_done = False

        # When set (via make_mock_soc(sim=...)), MyProgramV2.acquire detects this
        # and routes through the SimEngine instead of the white-noise path (D1).
        #
        # FLUX-AWARE-MOCK copy-on-input: keep an *internal* copy of the SimParams
        # so that set_flux_device (and any future per-soc mutation) never writes
        # through to the caller's instance.  The GUI mock-connect path passes the
        # shared singleton DEFAULT_SIMPARAM (params.py), so mutating it in place
        # would alias across every mock soc; the copy makes each soc own its params.
        self._sim_params: SimParams | None = (
            sim.model_copy() if sim is not None else None
        )

        # The SimEngine compute handle, injected by MyProgramV2.acquire on the sim
        # path (set_sim_engine).  poll_data computes one round *lazily* off it —
        # nothing is pre-computed, so an early-stopping run never computes a round
        # it does not poll.  The round counter tracks which round to ask for next;
        # the engine redraws independent noise per round (deterministic grid is
        # cached inside the engine on first call).
        self._sim_engine: SimEngine | None = None
        self._sim_round_idx: int = 0

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

    @property
    def sim_params(self) -> SimParams | None:
        """This soc's internal SimParams copy (None on the white-noise path).

        FLUX-AWARE-MOCK: read-only accessor so the GUI mock-connect path can derive
        a matching FluxoniumPredictor from the *actual* simulated physics (EJ/EC/EL
        + flux affine) rather than re-reading the DEFAULT_SIMPARAM constant — keeping
        the predictor consistent if a future connect parameterises the sim.
        Returns the soc-owned copy (copy-on-input); callers must not mutate it.
        """
        return self._sim_params

    def set_flux_device(self, name: str | None) -> None:
        """Bind (or unbind) the operating-flux source device (FLUX-AWARE-MOCK).

        Sets ``flux_device`` on this soc's *internal* SimParams copy, so the next
        acquire's SimEngine reads the named ``FakeDevice``'s live value (mapped
        through ``value_to_flux``) instead of the fixed reduced flux = 1.0.  The
        device need not be registered yet — binding only records the name;
        resolution and the FakeDevice check happen lazily at acquire time
        (engine._operating_signal), so this can be called before the device is
        connected.  Pass ``name=None`` to fall back to the fixed operating point.

        Raises if this soc carries no SimParams: a flux_device binding is
        meaningless on the white-noise mock (no SimEngine reads it), so silently
        accepting it would hide a wiring mistake (fast-fail).  ``with_updates``
        re-validates and returns a fresh instance, preserving the copy-on-input
        isolation.
        """

        if self._sim_params is None:
            raise RuntimeError(
                "set_flux_device requires a SimParams-backed mock soc; this soc "
                "was built without sim (white-noise path), so a flux_device "
                "binding has no SimEngine to read it"
            )
        self._sim_params = self._sim_params.with_updates(flux_device=name)

    def set_sim_engine(self, engine: SimEngine) -> None:
        """Attach the SimEngine compute handle for this acquire() (sim path).

        Called by MyProgramV2.acquire on the sim path *before* the real round
        loop runs.  Nothing is computed here — poll_data computes each round
        lazily off this engine in poll order, so early-stop saves the unpolled
        rounds' compute.
        """

        self._sim_engine = engine
        self._sim_round_idx = 0

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot) -> None:
        self._readout_state = (int(total_shots), list(ch_list), list(reads_per_shot))
        self._poll_done = False

    def poll_data(self, totaltime: float = 0.1, timeout=None):
        if self._poll_done or self._readout_state is None:
            return []
        profile_start = perf_now()
        total_shots, _ch_list, reads_per_shot = self._readout_state
        compute_ms = 0.0
        sleep_ms = 0.0

        if self._sim_engine is not None:
            # Sim path: compute *this* round's raw buffer lazily off the engine
            # (deterministic grid cached inside the engine; only noise is fresh),
            # then flatten to the (total_shots*nreads, 2) per-channel shape the
            # accumulated loop writes into acc_buf via a flat reshape.  No budget
            # ceiling: the engine computes any round on demand, so a run that
            # early-stops simply never asks for the rounds it skips.
            compute_start = perf_now()
            round_buf = self._sim_engine.compute_round(self._sim_round_idx)
            compute_ms = elapsed_ms(compute_start)
            self._sim_round_idx += 1
            data = [ch.reshape(-1, 2) for ch in round_buf]
        else:
            compute_start = perf_now()
            data = [
                np.random.randint(
                    -(2**15), 2**15, size=(total_shots * n, 2), dtype=np.int64
                )
                for n in reads_per_shot
            ]
            compute_ms = elapsed_ms(compute_start)

        # Synthetic poll pacing — not physics; sim path uses SimParams.poll_latency,
        # white-noise path uses the module-level _DEFAULT_POLL_LATENCY constant.
        # Skip the sleep call entirely when latency == 0 to avoid syscall noise.
        latency = (
            self._sim_params.poll_latency
            if self._sim_params is not None
            else _DEFAULT_POLL_LATENCY
        )
        if latency > 0.0:
            sleep_start = perf_now()
            time.sleep(latency * np.asarray(data).size)
            sleep_ms = elapsed_ms(sleep_start)

        self._poll_done = True
        _POLL_DATA_PERF.record(
            elapsed_ms(profile_start),
            detail=(
                f"sim={self._sim_engine is not None} total_shots={total_shots} "
                f"reads_per_shot={reads_per_shot} compute_ms={compute_ms:.1f} "
                f"sleep_ms={sleep_ms:.1f}"
            ),
        )
        return [(total_shots, (data, {}))]

    def get_decimated(self, ch, address, length):
        # Sim path: render this round's time-domain trace off the engine (model A,
        # timeFly-shifted readout envelope × steady mixed S21) and slice to the
        # requested length; each call redraws fresh noise so software-averaging the
        # rounds improves SNR (mirrors poll_data's accumulated path).  With no
        # engine attached (D1) keep the white-noise stub unchanged.
        if self._sim_engine is not None:
            (trace,) = self._sim_engine.compute_decimated()
            return trace[:length].astype(float)
        return np.random.randn(length, 2).astype(float)

    def get_accumulated(self, ch, address, length):
        # acc_buf is built alongside dec_buf by the decimated finish_round, but
        # lookback never consumes it (it reads only get_decimated), so even on the
        # sim path the accumulated companion stays white noise (D1) — it is safe
        # because nothing downstream reads it for a decimated acquire.
        return np.random.randint(-(2**15), 2**15, size=(length, 2), dtype=np.int64)


def make_mock_soc(
    n_gens: int = 2,
    n_readouts: int = 1,
    sim: SimParams | None = None,
) -> tuple[MockQickSoc, QickConfig]:
    """Build a MockQickSoc supporting end-to-end acquire().

    With ``sim=None`` (D1) the soc returns white-noise fake data and acquire
    follows the unchanged real path.  With a ``SimParams``, MyProgramV2.acquire
    detects ``soc._sim_params`` and routes through the SimEngine to produce
    physically-realistic data.
    """
    mock_cfg = _build_mock_cfg(n_gens, n_readouts)
    return MockQickSoc(mock_cfg, sim=sim), QickConfig(mock_cfg)
