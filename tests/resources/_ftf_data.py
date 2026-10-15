"""Small synthetic numeric fixture; no repository example or lab dependencies."""


def ftf_document() -> dict[str, object]:
    return {
        "format": "zcu.ftf", "version": 1,
        "channels": {"generators": 3, "readouts": 1},
        "model": {
            "f1": {"ej_ghz": 3., "ec_ghz": .8, "el_ghz": .6, "native_cutoff": 8, "levels": 2},
            "coupler": {"ejmax_ghz": 10., "ec_ghz": .3, "asymmetry": .2, "charge_cutoff": 3, "levels": 2, "q_cap": 1e6},
            "f2": {"ej_ghz": 3.3, "ec_ghz": .7, "el_ghz": .55, "native_cutoff": 8, "levels": 2},
            "g1c_ghz": .03, "g2c_ghz": .04, "g12_ghz": -.02,
            "retained_levels": 8, "temperature_k": 0.,
        },
        "devices": [{"name": "bias", "unit": "V", "limits": [-1., 1.], "initial": .2, "rampstep": .1}],
        "dc_mapping": {"offset": [.3, .2, .4], "columns": [{"source": "bias", "unit": "Phi0/V", "coefficients": [0., .1, 0.]}]},
        "flux_sources": [{"name": "flux", "gen_ch": 1, "coefficients_phi0_per_gain": [0., .1, 0.]}],
        "microwave_sources": [{"name": "mw", "gen_ch": 2, "coefficients_ghz_per_gain": [[.02, .01], [0., 0.], [-.01, .02]]}],
        "readout_source": {"gen_ch": 0, "input_per_gain": 1.},
        "receiver": {"ro_ch": 0, "gain": 10000., "noise_psd_adc2_ns": [[0., 0.], [0., 0.]]},
        "resonators": [
            {"frequency_ghz": 6., "charge_coefficients_ghz": [.03, .01, 0.], "kappa_external_mhz": 1., "kappa_internal_mhz": .1, "thermal_occupation": 0.},
            {"frequency_ghz": 6.2, "charge_coefficients_ghz": [0., .01, .03], "kappa_external_mhz": 1., "kappa_internal_mhz": .2, "thermal_occupation": 0.},
        ],
        "solver": {"rtol": 1e-8, "atol": 1e-10, "max_step_ns": 1., "bath_segment_ns": 4.},
        "seed": 123,
    }
