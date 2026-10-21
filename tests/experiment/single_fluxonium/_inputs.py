"""Small synthetic settings and compiled template inputs, without lab imports."""

import numpy as np

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    EventSlot,
    ReadoutSlot,
    TimelineTemplate,
)
from zcu_tools.resources.single_fluxonium import parse_single_fluxonium
from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings


def settings() -> SingleFluxoniumSettings:
    return parse_single_fluxonium(
        {
            "format": "zcu.single-fluxonium",
            "version": 1,
            "chip": {"qubit": {"ej_ghz": 4, "ec_ghz": 1, "el_ghz": 1}, "temperature_k": 0},
            "dc_bias": {
                "offset_phi0": 0.5,
                "source": {"name": "single_test_bias", "unit": "A", "limits": [-1, 1],
                           "initial": 0, "ramp_step": 0.01, "flux_per_unit": 0.1},
            },
            "drive_line": {"name": "mw", "gen_ch": 1,
                           "charge_drive_ghz_per_gain": {"real": 0.1, "imag": -0.02}},
            "flux_line": {"name": "ff", "gen_ch": 2, "flux_phi0_per_gain": 0.2,
                          "delay_ns": 2, "tau_ns": 2},
            "readout": {
                "drive": {"gen_ch": 0, "input_per_gain": 10},
                "resonator": {"frequency_ghz": 7, "charge_coefficient_ghz": 0,
                              "kappa_external_mhz": 1, "kappa_internal_mhz": 0},
                "adc": {"ro_ch": 0, "gain": 1, "noise_psd_adc2_ns": [[0, 0], [0, 0]]},
            },
            "numerics": {"truncation": {"native_cutoff": 12, "levels": 4},
                         "solver": {"rtol": 1e-8, "atol": 1e-10, "max_step_ns": 0.1,
                                    "bath_segment_ns": 2}, "seed": 123},
        }
    )


def snapshot(value: float = 0) -> EnvironmentSnapshot:
    return EnvironmentSnapshot("single-fluxonium-v1", "synthetic", (("single_test_bias", value),))


def slot(channel: int, start_ns: float, duration_ns: float, *,
         frequency_mhz: float = 0, phase_deg: float = 0, gain: float = 1) -> EventSlot:
    return EventSlot(
        channel, "synthetic", (), np.full(2, start_ns / 1000),
        np.full(2, duration_ns / 1000), np.full(2, frequency_mhz),
        np.full(2, phase_deg), np.full(2, gain), (None,),
        np.zeros(2, dtype=np.int64), np.ones(2, dtype=bool),
    )


def template() -> TimelineTemplate:
    readout = slot(0, 16, 8, frequency_mhz=7000, phase_deg=30, gain=0.1)
    window = ReadoutSlot(0, 4, 1000, np.full((2, 1), 0.018),
                         np.full((2, 1), 7000), np.zeros((2, 1)),
                         np.zeros((2, 1), dtype=np.int64))
    return TimelineTemplate(
        np.array([0, 3], dtype=np.int64),
        (slot(1, 2, 6, frequency_mhz=250, phase_deg=90, gain=-0.2),
         slot(2, 0, 8, gain=0.5), readout), (window,), np.full(2, 0.024),
    )
