"""Small synthetic numeric fixture; no repository example or lab dependencies."""


def ftf_document() -> dict[str, object]:
    return {
        "format": "zcu.ftf",
        "version": 1,
        "channels": {"generators": 3, "readouts": 1},
        "model": {
            "f1": {
                "ej_ghz": 3.0,
                "ec_ghz": 0.8,
                "el_ghz": 0.6,
                "native_cutoff": 8,
                "levels": 2,
            },
            "coupler": {
                "ejmax_ghz": 10.0,
                "ec_ghz": 0.3,
                "asymmetry": 0.2,
                "charge_cutoff": 3,
                "levels": 2,
                "q_cap": 1e6,
            },
            "f2": {
                "ej_ghz": 3.3,
                "ec_ghz": 0.7,
                "el_ghz": 0.55,
                "native_cutoff": 8,
                "levels": 2,
            },
            "g1c_ghz": 0.03,
            "g2c_ghz": 0.04,
            "g12_ghz": -0.02,
            "retained_levels": 8,
            "temperature_k": 0.0,
        },
        "devices": [
            {
                "name": "bias",
                "unit": "V",
                "limits": [-1.0, 1.0],
                "initial": 0.2,
                "rampstep": 0.1,
            }
        ],
        "dc_mapping": {
            "offset": [0.3, 0.2, 0.4],
            "columns": [
                {"source": "bias", "unit": "Phi0/V", "coefficients": [0.0, 0.1, 0.0]}
            ],
        },
        "flux_sources": [
            {"name": "flux", "gen_ch": 1, "coefficients_phi0_per_gain": [0.0, 0.1, 0.0]}
        ],
        "microwave_sources": [
            {
                "name": "mw",
                "gen_ch": 2,
                "coefficients_ghz_per_gain": [[0.02, 0.01], [0.0, 0.0], [-0.01, 0.02]],
            }
        ],
        "readout_source": {"gen_ch": 0, "input_per_gain": 1.0},
        "receiver": {
            "ro_ch": 0,
            "gain": 10000.0,
            "noise_psd_adc2_ns": [[0.0, 0.0], [0.0, 0.0]],
        },
        "resonators": [
            {
                "frequency_ghz": 6.0,
                "charge_coefficients_ghz": [0.03, 0.01, 0.0],
                "kappa_external_mhz": 1.0,
                "kappa_internal_mhz": 0.1,
                "thermal_occupation": 0.0,
            },
            {
                "frequency_ghz": 6.2,
                "charge_coefficients_ghz": [0.0, 0.01, 0.03],
                "kappa_external_mhz": 1.0,
                "kappa_internal_mhz": 0.2,
                "thermal_occupation": 0.0,
            },
        ],
        "solver": {
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step_ns": 1.0,
            "bath_segment_ns": 4.0,
        },
        "seed": 123,
    }


def ftf_v2_document() -> dict[str, object]:
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
