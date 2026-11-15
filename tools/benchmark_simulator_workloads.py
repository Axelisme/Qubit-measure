"""Versioned synthetic board, physical inputs and public-acquire timelines.

No lab definitions, default YAML or test fixtures are loaded. Change workload
version when timeline semantics change, configuration_version when inputs change.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.simulate.evolution import RoutingHistory
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.ftf.environment import ftf
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import single_fluxonium
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import six_fluxonium


@dataclass(frozen=True)
class AcquireSession:
    """Prepared acquire and routing-clear callbacks; yielding context owns cleanup.

    acquire executes the complete public program acquire, not setup/compilation.
    clear_history invalidates routing preparations without resetting RNG/models.
    """

    acquire: Callable[[], object]
    clear_history: Callable[[], None]


class Workload(BaseModel):
    """Named synthetic timeline version and complete JSON board/model/program inputs.

    version identifies timeline semantics. configuration_version identifies input
    revisions. configuration contains board, model and program settings; its full
    content is hashed and retained in the result. name selects a supported timeline.
    """

    name: str
    version: int
    configuration_version: int
    configuration: dict[str, JsonValue]


def _board(*, mux: bool) -> dict[str, JsonValue]:
    return {
        "board": "ZCU216",
        "sw_version": "0.2.436",
        "fw_timestamp": "2025-01-01",
        "refclk_freq": 245.76,
        "extra_description": [],
        "tprocs": [
            {
                "type": "qick_processor",
                "revision": 27,
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
        "generators": [
            {
                "type": "axis_signal_gen_v6",
                "fs_mult": 50,
                "fs_div": 1,
                "interpolation": 1,
                "fabric_div": 32,
                "rf_fabric_div": 16,
                "samps_per_clk": 16,
                "b_dds": 32,
                "b_phase": 32,
                "maxlen": 65536,
                "complex_env": True,
                "has_dds": True,
                "has_mixer": False,
                "maxv": 32766,
                "maxv_scale": 1.0,
                "dac": f"{ch:02d}",
                "tproc_ch": ch,
                "tmux_ch": None,
            }
            for ch in range(8)
        ],
        "readouts": [
            {
                "type": "axis_readout_v2",
                "ro_type": "axis_readout_v2",
                "fs_mult": 10,
                "fs_div": 1,
                "decimation": 1,
                "samps_per_clk": 8,
                "b_dds": 32,
                "buf_maxlen": 8192,
                "avg_maxlen": 8192,
                "avgbuf_type": "axis_avg_buffer",
                "avgbuf_version": "v2",
                "has_edge_counter": False,
                "has_weights": False,
                "has_outsel": True,
                "iq_offset": 0.0,
                "coupling": "AC",
                "adc": f"{ch:02d}",
                "tproc_ctrl": None if mux and ch < 6 else 8 + ch,
                "tmux_ch": None,
                "trigger_type": "dport",
                "trigger_port": 0,
                "trigger_bit": ch,
                "tproc_ch": ch,
                "b_phase": 32 if mux and ch < 6 else None,
            }
            for ch in range(7)
        ],
        "rf": {"dac_power": "VOP", "clk_groups": []},
        "mux": [
            {
                "gen_ch": 0,
                "ro_chs": list(range(6)),
                "generator": {
                    "type": "axis_sg_mixmux8_v1",
                    "has_mixer": True,
                    "n_tones": 8,
                    "has_gain": True,
                    "has_phase": True,
                    "interpolation": 8,
                },
                "readout": {
                    "type": "axis_pfb_readout_v4",
                    "ro_type": "axis_pfb_readout_v4",
                    "ro_fullpath": "pfb0",
                    "pfb_nout": 8,
                    "pfb_nch": 64,
                    "pfb_ch_offset": 32,
                    "pfb_dds_on_output": True,
                    "has_outsel": False,
                },
            }
        ]
        if mux
        else [],
    }


def _single() -> dict[str, JsonValue]:
    return {
        "format": "zcu.single-fluxonium",
        "version": 1,
        "chip": {
            "qubit": {"ej_ghz": 4, "ec_ghz": 1, "el_ghz": 1},
            "temperature_k": 0.03,
        },
        "dc_bias": {
            "offset_phi0": 0.45,
            "source": {
                "name": "benchmark_bias",
                "unit": "A",
                "limits": [-1, 1],
                "initial": 0,
                "ramp_step": 0.01,
                "flux_per_unit": 0.1,
            },
        },
        "drive_lines": {
            "mw": {
                "name": "mw",
                "gen_ch": 1,
                "charge_drive_ghz_per_gain": {"real": 0.1, "imag": -0.02},
            }
        },
        "flux_line": {
            "name": "ff",
            "gen_ch": 2,
            "flux_phi0_per_gain": 0.05,
            "delay_ns": 2,
            "tau_ns": 2,
        },
        "readout": {
            "drive": {"gen_ch": 0, "input_per_gain": 10},
            "resonator": {
                "thermal_occupation": None,
                "frequency_ghz": 7,
                "charge_coefficient_ghz": 0.01,
                "kappa_external_mhz": 1,
                "kappa_internal_mhz": 0.1,
            },
            "adc": {
                "background": {"ripple_period_mhz": 1000, "pivot_ghz": 0},
                "ro_ch": 0,
                "gain": 10000,
                "noise_psd_adc2_ns": [[0.01, 0], [0, 0.01]],
            },
        },
        "numerics": {
            "projection_prune_floor": 1e-9,
            "slow_envelope_segments_per_period": 20,
            "truncation": {"native_cutoff": 20, "levels": 3},
            "solver": {
                "rtol": 1e-8,
                "atol": 1e-10,
                "max_step_ns": 0.5,
                "bath_segment_ns": 4,
            },
            "seed": 20261112,
            "time_scale": 0,
        },
        "drift": None,
    }


def _six() -> dict[str, JsonValue]:
    single = parse_single_fluxonium(_single())
    qubits: dict[str, JsonValue] = {}
    for index in range(6):
        qubits[f"Q{index + 1}"] = {
            "format": "zcu.single-fluxonium",
            "version": 1,
            "chip": single.chip.model_dump(mode="json"),
            "dc_bias": {
                "source": {
                    "name": f"Q{index + 1}_bias",
                    "unit": "A",
                    "limits": [-1, 1],
                    "initial": 0,
                    "ramp_step": 0.01,
                },
                "offset": -0.45,
                "period": 1,
                "coil_offset": 0,
                "coil_period": 2,
            },
            "drive_line": {
                "name": f"mw{index}",
                "gen_ch": index + 1,
                "charge_drive_ghz_per_gain": {"real": 0.1, "imag": 0},
            },
            "readout": {
                "drive": {"gen_ch": 0, "input_per_gain": 10},
                "resonator": {
                    **single.readout.resonator.model_dump(mode="json"),
                    "frequency_ghz": 6.5 + 0.125 * index,
                },
                "adc": {**single.readout.adc.model_dump(mode="json"), "ro_ch": index},
            },
            "numerics": single.numerics.model_dump(mode="json"),
            "drift": None,
        }
    return {
        "format": "zcu.six-fluxonium",
        "version": 1,
        "qubits": qubits,
        "mux_gen_ch": 0,
        "pulse_gen_ch": 7,
        "pulse_ro_ch": 6,
        "dc_source": {
            "name": "coil_bias",
            "unit": "A",
            "limits": [-1, 1],
            "initial": 0,
            "ramp_step": 0.01,
        },
        "numerics": {
            "routing_cache_bytes": 4 * 1024**2,
            "slow_envelope_segments_per_period": 20,
        },
    }


def _ftf() -> dict[str, JsonValue]:
    return {
        "format": "zcu.ftf",
        "version": 2,
        "chip": {
            "f1": {"ej_ghz": 3, "ec_ghz": 0.8, "el_ghz": 0.6},
            "coupler": {"ejmax_ghz": 10, "ec_ghz": 0.3, "asymmetry": 0.2, "q_cap": 1e6},
            "f2": {"ej_ghz": 3.3, "ec_ghz": 0.7, "el_ghz": 0.55},
            "temperature_k": 0.03,
            "g_f1_coupler_ghz": 0.03,
            "g_f2_coupler_ghz": 0.04,
            "g_f1_f2_ghz": -0.02,
        },
        "dc_bias": {
            "offset_phi0": {"f1": 0.3, "coupler": 0.2, "f2": 0.4},
            "sources": {
                "benchmark_ftf_bias": {
                    "unit": "V",
                    "limits": [-1, 1],
                    "initial": 0,
                    "ramp_step": 0.01,
                    "flux_per_unit": {"f1": 0, "coupler": 0.1, "f2": 0},
                }
            },
        },
        "drive_lines": {
            f"mw{ch}": {
                "gen_ch": ch,
                "charge_drive_ghz_per_gain": {
                    "f1": {"real": 0.02 if ch == 1 else 0.005, "imag": 0},
                    "coupler": {"real": 0, "imag": 0},
                    "f2": {"real": 0.005 if ch == 1 else 0.02, "imag": 0},
                },
            }
            for ch in (1, 2)
        },
        "readout": {
            "drive": {"gen_ch": 0, "input_per_gain": 1},
            "resonators": {
                "r1": {
                    "frequency_ghz": 6,
                    "charge_coefficients_ghz": {"f1": 0.03, "coupler": 0.01, "f2": 0},
                    "kappa_external_mhz": 1,
                    "kappa_internal_mhz": 0.1,
                    "thermal_occupation": 0,
                },
                "r2": {
                    "frequency_ghz": 6.2,
                    "charge_coefficients_ghz": {"f1": 0, "coupler": 0.01, "f2": 0.03},
                    "kappa_external_mhz": 1,
                    "kappa_internal_mhz": 0.1,
                    "thermal_occupation": 0,
                },
            },
            "adc": {
                "background": {"ripple_period_mhz": 1000, "pivot_ghz": 0},
                "ro_ch": 0,
                "gain": 10000,
                "noise_psd_adc2_ns": [[0.01, 0], [0, 0.01]],
            },
        },
        "numerics": {
            "projection_weight_floor": 1e-9,
            "truncation": {
                "f1": {"native_cutoff": 12, "levels": 2},
                "coupler": {"charge_cutoff": 3, "levels": 2},
                "f2": {"native_cutoff": 12, "levels": 2},
                "retained_levels": 4,
            },
            "solver": {
                "rtol": 1e-8,
                "atol": 1e-10,
                "max_step_ns": 0.5,
                "bath_segment_ns": 4,
            },
            "seed": 20261112,
            "time_scale": 0,
            "telemetry": {"enabled": False},
        },
        "drift": None,
    }


def _stirap() -> Workload:
    model = _single()
    model["drive_lines"] = {
        name: {
            "name": name,
            "gen_ch": ch,
            "charge_drive_ghz_per_gain": {"real": 1.0, "imag": 0.0},
        }
        for name, ch in (("pump", 1), ("stokes", 2))
    }
    model["flux_line"] = None
    model["numerics"] = {
        "projection_prune_floor": 1e-9,
        "slow_envelope_segments_per_period": 20,
        "truncation": {"native_cutoff": 80, "levels": 20},
        "solver": {
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step_ns": 0.5,
            "bath_segment_ns": 4,
        },
        "seed": 20261112,
        "time_scale": 0,
    }
    return Workload(
        name="single_stirap_shaped",
        version=1,
        configuration_version=1,
        configuration={
            "board": _board(mux=False),
            "model": parse_single_fluxonium(model).model_dump(mode="json"),
            "program": {
                "reps": 32,
                "gains": [0.0344, 0.043, 0.0516],
                "drive_length_us": 0.42,
                "readout_length_us": 0.032,
                "shaped_repeats": 1,
                "gaussian_sigma_us": 0.07,
                # Native-80 EJ/EC/EL=4/1/1 GHz, flux=.45: E1-E0 and E2-E1.
                "drive_frequency_mhz": 1433.4218883789758,
                "second_frequency_mhz": 2904.5631072147653,
                "readout_frequency_mhz": 7000,
                "second_gain": 0.143,
                "readout_gain": 0.1,
                "mux_frequencies_mhz": [],
                "mux_mixer_frequency_mhz": 0,
                "initial_delay_us": 0,
                "relax_delay_us": 0,
                "first_tone_offset_us": 0.105,
                "second_tone_offset_us": 0,
            },
        },
    )


def make_workloads() -> list[Workload]:
    """Return the versioned single/six/FTF inputs, then the STIRAP-shaped scan.

    Model parsers expand defaults into the retained configuration. Board and
    program values are synthetic and do not depend on repository YAML or caches.
    """
    single = parse_single_fluxonium(_single()).model_dump(mode="json")
    six = parse_six_fluxonium(_six()).model_dump(mode="json")
    circuit = parse_ftf(_ftf()).model_dump(mode="json")
    return [
        Workload(
            name=name,
            version=2 if name.endswith("shaped") else 1,
            configuration_version=2 if name.endswith("shaped") else 1,
            configuration={
                "board": _board(mux=name == "six_mux"),
                "model": model,
                "program": {
                    "reps": 32,
                    "gains": [0.16, 0.24, 0.32]
                    if name.endswith("shaped")
                    else [0.08, 0.12, 0.16],
                    "drive_length_us": 0.024,
                    "readout_length_us": 0.032,
                    "shaped_repeats": 4,
                    "gaussian_sigma_us": 0.0048,
                    "drive_frequency_mhz": 350,
                    "second_frequency_mhz": 550 if name.startswith("ftf") else 0,
                    "readout_frequency_mhz": 6000 if name.startswith("ftf") else 7000,
                    "second_gain": 0.2 if name == "ftf_overlap_shaped" else 0.1,
                    "readout_gain": 0.1,
                    "mux_frequencies_mhz": [6500 + 125 * ch for ch in range(6)],
                    "mux_mixer_frequency_mhz": 6800,
                    "initial_delay_us": 0,
                    "relax_delay_us": 0,
                    "second_tone_offset_us": 0.008
                    if name == "ftf_overlap_shaped"
                    else 0,
                },
            },
        )
        for name, model in [
            ("single_drive_flux_ramp", single),
            ("single_repeated_shaped", single),
            ("six_mux", six),
            ("ftf_simultaneous_tones", circuit),
            ("ftf_overlap_shaped", circuit),
        ]
    ] + [_stirap()]


class _ProgramInputs(BaseModel):
    reps: int
    gains: list[float]
    drive_length_us: float
    readout_length_us: float
    shaped_repeats: int
    first_tone_offset_us: float = 0
    second_tone_offset_us: float
    gaussian_sigma_us: float
    drive_frequency_mhz: float
    second_frequency_mhz: float
    readout_frequency_mhz: float
    second_gain: float
    readout_gain: float
    mux_frequencies_mhz: list[float]
    mux_mixer_frequency_mhz: float
    initial_delay_us: float
    relax_delay_us: float


def _program(workload: Workload, builder: TimelineBuilder) -> TimelineProgramV2:
    inputs = _ProgramInputs.model_validate(workload.configuration["program"])
    gain = builder.add_table_sweep("gain", inputs.gains)
    if workload.name == "six_mux":
        for ch in range(1, 7):
            builder.emit(
                PulseCfg(
                    ch=ch,
                    nqz=1,
                    freq=inputs.drive_frequency_mhz,
                    gain=gain,
                    waveform=ConstWaveformCfg(length=inputs.drive_length_us),
                ).build(f"drive{ch}")
            )
        readout = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=inputs.mux_mixer_frequency_mhz,
            length=inputs.readout_length_us,
            ro_length=inputs.readout_length_us,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=freq, gain=inputs.readout_gain)
                for ch, freq in enumerate(inputs.mux_frequencies_mhz)
            ],
        )
        builder.emit(readout.build("readout"))
    else:
        shaped = workload.name in {
            "single_repeated_shaped",
            "ftf_overlap_shaped",
            "single_stirap_shaped",
        }
        waveform = (
            GaussWaveformCfg(
                length=inputs.drive_length_us, sigma=inputs.gaussian_sigma_us
            )
            if shaped
            else ConstWaveformCfg(length=inputs.drive_length_us)
        )
        first = PulseCfg(
            ch=1,
            nqz=1,
            freq=inputs.drive_frequency_mhz,
            gain=gain,
            pre_delay=inputs.first_tone_offset_us,
            waveform=waveform,
        ).build("first")
        if workload.name.startswith("ftf") or workload.name in {
            "single_drive_flux_ramp",
            "single_stirap_shaped",
        }:
            builder.fork("default", into="second")
            second = PulseCfg(
                ch=2,
                nqz=1,
                freq=inputs.second_frequency_mhz,
                gain=gain
                if workload.name == "single_drive_flux_ramp"
                else inputs.second_gain,
                pre_delay=inputs.second_tone_offset_us,
                waveform=waveform,
            ).build("second")
            builder.emit(first)
            builder.emit(second, timeline="second")
            builder.join("second", into="default")
        else:
            for _ in range(inputs.shaped_repeats):
                builder.emit(first)
        builder.emit(
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=inputs.readout_frequency_mhz,
                gain=inputs.readout_gain,
                waveform=ConstWaveformCfg(length=inputs.readout_length_us),
                ro_length=inputs.readout_length_us,
            ).build("readout")
        )
    builder.delay_by_timeline()
    return builder.build()


@contextmanager
def open_session(workload: Workload) -> Generator[AcquireSession]:
    """Build a fresh synthetic environment and compiled program, yield acquire/clear.

    Supported names come from make_workloads; other names raise ValueError before
    environment construction. No warmup runs here. Closing always releases the
    numeric backend and fake devices, including failed acquires. Configuration,
    compilation and acquisition errors propagate to the caller.
    """
    if workload.name not in {item.name for item in make_workloads()}:
        raise ValueError(f"unsupported workload: {workload.name}")
    board = workload.configuration["board"]
    if not isinstance(board, dict):
        raise ValueError("board must be a mapping")
    soccfg = build_soccfg(board)
    history = RoutingHistory()
    if workload.name.startswith("single_"):
        env = single_fluxonium(
            parse_single_fluxonium(workload.configuration["model"]),
            soccfg=soccfg,
            history=history,
        )
    elif workload.name == "six_mux":
        env = six_fluxonium(
            parse_six_fluxonium(workload.configuration["model"]),
            soccfg=soccfg,
            history=history,
        )
    elif workload.name.startswith("ftf_"):
        env = ftf(
            parse_ftf(workload.configuration["model"]), soccfg=soccfg, history=history
        )
    else:
        raise ValueError(f"unsupported workload: {workload.name}")
    try:
        inputs = _ProgramInputs.model_validate(workload.configuration["program"])
        program = _program(
            workload,
            TimelineBuilder(
                soccfg,
                reps=inputs.reps,
                initial_delay=inputs.initial_delay_us,
                relax_delay=inputs.relax_delay_us,
            ),
        )
        yield AcquireSession(
            lambda: program.acquire(env.soc, progress=False), history.clear
        )
    finally:
        try:
            if env.close_backend is not None:
                env.close_backend()
        finally:
            env.devices.close_all_devices()
