"""Small synthetic numeric fixture; no repository example or lab dependencies."""


def ftf_document() -> dict[str, object]:
    """Minimal v2 sample with signed named calibrations and a generator hole."""
    return {
        "format": "zcu.ftf",
        "version": 2,
        "chip": {
            "f1": {"ej_ghz": 3.0, "ec_ghz": 0.8, "el_ghz": 0.6},
            "coupler": {
                "ejmax_ghz": 10.0,
                "ec_ghz": 0.3,
                "asymmetry": 0.2,
                "q_cap": 1000000.0,
            },
            "f2": {"ej_ghz": 3.3, "ec_ghz": 0.7, "el_ghz": 0.55},
            "temperature_k": 0.0,
            "g_f1_coupler_ghz": 0.03,
            "g_f2_coupler_ghz": 0.04,
            "g_f1_f2_ghz": -0.02,
        },
        "dc_bias": {
            "offset_phi0": {"f1": 0.3, "coupler": 0.2, "f2": 0.4},
            "sources": {
                "bias": {
                    "unit": "V",
                    "limits": [-1.0, 1.0],
                    "initial": 0.2,
                    "ramp_step": 0.1,
                    "flux_per_unit": {"f1": 0.0, "coupler": 0.1, "f2": 0.0},
                }
            },
        },
        "drive_lines": {
            "mw": {
                "gen_ch": 2,
                "charge_drive_ghz_per_gain": {
                    "f1": {"real": 0.02, "imag": 0.01},
                    "coupler": {"real": 0.0, "imag": 0.0},
                    "f2": {"real": -0.01, "imag": 0.02},
                },
            }
        },
        "readout": {
            "drive": {"gen_ch": 0, "input_per_gain": 1.0},
            "resonators": {
                "r1": {
                    "frequency_ghz": 6.0,
                    "charge_coefficients_ghz": {"f1": 0.03, "coupler": 0.01, "f2": 0.0},
                    "kappa_external_mhz": 1.0,
                    "kappa_internal_mhz": 0.1,
                    "thermal_occupation": 0.0,
                },
                "r2": {
                    "frequency_ghz": 6.2,
                    "charge_coefficients_ghz": {"f1": 0.0, "coupler": 0.01, "f2": 0.03},
                    "kappa_external_mhz": 1.0,
                    "kappa_internal_mhz": 0.2,
                    "thermal_occupation": 0.0,
                },
            },
            "adc": {
                "ro_ch": 0,
                "gain": 10000.0,
                "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]],
            },
        },
        "numerics": {
            "truncation": {
                "f1": {"levels": 2, "native_cutoff": 8},
                "coupler": {"levels": 2, "charge_cutoff": 3},
                "f2": {"levels": 2, "native_cutoff": 8},
                "retained_levels": 8,
            },
            "solver": {
                "rtol": 1e-08,
                "atol": 1e-10,
                "max_step_ns": 1.0,
                "bath_segment_ns": 4.0,
            },
            "seed": 123,
        },
    }


def dual_readout_document(channels: tuple[int, ...] = (4, 1)) -> dict[str, object]:
    """Pulse plus independent mux source with distinct receiver calibrations."""
    document = ftf_document()
    readout = document["readout"]
    if not isinstance(readout, dict):
        raise TypeError("fixture requires a readout mapping")
    readout["mux"] = {
        "gen_ch": 1,
        "input_per_gain": -2.0,
        "adc": [
            {
                "ro_ch": ch,
                "gain": 1000.0 + i,
                "phase_deg": 10.0 + i,
                "offset_adc": [0.1 + i, -0.2],
                "noise_psd_adc2_ns": [[1.0 + i, 0.0], [0.0, 2.0]],
                "time_of_flight_ns": 20.0 + i,
            }
            for i, ch in enumerate(channels)
        ],
    }
    return document


def mux_document(channels=(4, 1)) -> dict[str, object]:
    """The same tiny circuit with explicit mux receiver bindings."""
    document = ftf_document()
    readout = document["readout"]
    if not isinstance(readout, dict):
        raise TypeError("fixture requires a readout mapping")
    readout["mux"] = {
        "gen_ch": 1,
        "input_per_gain": 1.0,
        "adc": [
            {
                "ro_ch": ch,
                "gain": 10000.0 + 1000 * i,
                "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]],
            }
            for i, ch in enumerate(channels)
        ],
    }
    return document
