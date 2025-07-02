from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D, LivePlotter2D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    check_no_post_delay,
    make_readout,
    make_reset,
)

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


def measure_reset_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
    sweep_cfg = cfg["sweep"]["freq"]

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Pulse(
                "reset_probe_pulse",
                cfg={
                    **cfg["reset_probe_pulse"],
                    "freq": sweep2param("freq", sweep_cfg),
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    fpts = sweep2array(sweep_cfg)  # predicted frequency points
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
        signal2real=reset_signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("reset_probe_pulse", "freq", as_array=True)
    assert isinstance(fpts, np.ndarray), "fpts should be an array"

    return fpts, signals


def measure_reset_time(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    len_params = sweep2param("length", cfg["sweep"]["length"])
    if reset_cfg["type"] == "pulse":
        reset_pulse = reset_cfg["pulse_cfg"]

        reset_pulse["length"] = len_params
    elif reset_cfg["type"] == "two_pulse":
        reset_pulse1 = reset_cfg["pulse_cfg1"]
        reset_pulse2 = reset_cfg["pulse_cfg2"]

        reset_pulse1["length"] = len_params
        reset_pulse2["length"] = len_params
    else:
        raise ValueError("Reset pulse must be pulse or two pulse")

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(name="init_pulse", cfg=cfg.get("init_pulse")),
            make_reset("tested_reset", reset_cfg=reset_cfg),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse gains
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Length (us)", "Amplitude"),
        ticks=(lens,),
        signal2real=reset_signal2real,
    )

    # get the actual pulse length
    if reset_cfg["type"] == "pulse":
        real_lens = prog.get_pulse_param("reset_pulse", "length", as_array=True)
    elif reset_cfg["type"] == "two_pulse":
        real_lens = prog.get_pulse_param("reset_pulse1", "length", as_array=True)
    else:
        raise ValueError("Reset pulse must be pulse or two pulse")

    assert isinstance(real_lens, np.ndarray), "real_lens should be an array"
    # TODO: better way to do this?
    real_lens += lens[0] - real_lens[0]  # add back the side length of the pulse

    return real_lens, signals  # lens


def measure_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_cfg = cfg["tested_reset"]
    init_pulse = cfg["init_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["sweep"] = {
        "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
        "gain": gain_sweep,
    }

    init_pulse["gain"] = sweep2param("gain", gain_sweep)

    # TODO: better way to do this?
    reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
    if reset_cfg["type"] == "pulse":
        reset_pulse = reset_cfg["pulse_cfg"]

        reset_pulse["gain"] = reset_factor * reset_pulse["gain"]
        reset_pulse["length"] = reset_factor * reset_pulse["length"] + 0.005
        if reset_pulse["style"] == "flat_top":  # prevent negative length in flat part
            reset_pulse["length"] += reset_pulse["raise_pulse"]["length"]
    elif reset_cfg["type"] == "two_pulse":
        reset_pulse1 = reset_cfg["pulse_cfg1"]
        reset_pulse2 = reset_cfg["pulse_cfg2"]

        reset_pulse1["gain"] = reset_factor * reset_pulse1["gain"]
        reset_pulse2["gain"] = reset_factor * reset_pulse2["gain"]
        reset_pulse1["length"] = reset_factor * reset_pulse1["length"] + 0.005
        reset_pulse2["length"] = reset_factor * reset_pulse2["length"] + 0.005
        if reset_pulse1["style"] == "flat_top":  # prevent negative length in flat part
            reset_pulse1["length"] += reset_pulse1["raise_pulse"]["length"]
        if reset_pulse2["style"] == "flat_top":  # prevent negative length in flat part
            reset_pulse2["length"] += reset_pulse2["raise_pulse"]["length"]

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            make_reset("tested_reset", reset_cfg=reset_cfg),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    pdrs = sweep2array(gain_sweep)  # predicted amplitudes
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter1D("Pulse gain", "Amplitude", num_lines=2),
        ticks=(pdrs,),
        signal2real=reset_signal2real,
    )

    pdrs = prog.get_pulse_param("init_pulse", "gain", as_array=True)
    assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

    return pdrs, signals


def measure_mux_reset_freq(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    check_no_post_delay(cfg["reset_probe_pulse1"], "reset_probe_pulse1")

    # force freq1 to be the outer loop
    cfg["sweep"] = {
        "freq1": cfg["sweep"]["freq1"],
        "freq2": cfg["sweep"]["freq2"],
    }

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Pulse(
                "reset_probe_pulse1",
                cfg={
                    **cfg["reset_probe_pulse1"],
                    "freq": sweep2param("freq1", cfg["sweep"]["freq1"]),
                },
            ),
            Pulse(
                "reset_probe_pulse2",
                cfg={
                    **cfg["reset_probe_pulse2"],
                    "freq": sweep2param("freq2", cfg["sweep"]["freq2"]),
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    fpts1 = sweep2array(cfg["sweep"]["freq1"])  # predicted frequency points
    fpts2 = sweep2array(cfg["sweep"]["freq2"])  # predicted frequency points
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)"),
        ticks=(fpts1, fpts2),
        signal2real=lambda x: np.abs(x - np.mean(x)),
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("reset_probe_pulse1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("reset_probe_pulse2", "freq", as_array=True)
    assert isinstance(fpts1, np.ndarray), "fpts1 should be an array"
    assert isinstance(fpts2, np.ndarray), "fpts2 should be an array"

    return fpts1, fpts2, signals


def measure_mux_reset_pdr(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    check_no_post_delay(cfg["reset_probe_pulse1"], "reset_probe_pulse1")

    # force gain1 to be the outer loop
    cfg["sweep"] = {
        "gain1": cfg["sweep"]["gain1"],
        "gain2": cfg["sweep"]["gain2"],
    }

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Pulse(
                "reset_probe_pulse1",
                cfg={
                    **cfg["reset_probe_pulse1"],
                    "gain": sweep2param("gain1", cfg["sweep"]["gain1"]),
                },
            ),
            Pulse(
                "reset_probe_pulse2",
                cfg={
                    **cfg["reset_probe_pulse2"],
                    "gain": sweep2param("gain2", cfg["sweep"]["gain2"]),
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    pdrs1 = sweep2array(cfg["sweep"]["gain1"])  # predicted amplitudes
    pdrs2 = sweep2array(cfg["sweep"]["gain2"])  # predicted amplitudes

    ref_i = 0 if pdrs1[0] < pdrs1[-1] else -1
    ref_j = 0 if pdrs2[0] < pdrs2[-1] else -1
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter2D("Gain1", "Gain2"),
        ticks=(pdrs1, pdrs2),
        signal2real=lambda x: np.abs(x - x[ref_i][ref_j]),
    )

    # get the actual frequency points
    pdrs1 = prog.get_pulse_param("reset_probe_pulse1", "gain", as_array=True)
    pdrs2 = prog.get_pulse_param("reset_probe_pulse2", "gain", as_array=True)
    assert isinstance(pdrs1, np.ndarray), "pdrs1 should be an array"
    assert isinstance(pdrs2, np.ndarray), "pdrs2 should be an array"

    return pdrs1, pdrs2, signals
