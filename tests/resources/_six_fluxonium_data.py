"""Synthetic six-cavity material, independent of lab defaults."""

from tests.experiment.single_fluxonium._inputs import settings


def six_document():
    qubits = {}
    for index in range(6):
        value = settings().model_dump(mode="json")
        del value["flux_line"]
        value["dc_bias"] = {"offset_phi0": 0.4 + 0.02 * index, "flux_per_unit": 0.1 * (index + 1)}
        value["drive_line"].update(name=f"mw_{index}", gen_ch=index + 1)
        value["readout"]["resonator"].update(frequency_ghz=6.8 + 0.125 * index)
        value["readout"]["adc"].update(ro_ch=index, gain=10000)
        value["numerics"].update(min_round_duration_s=0)
        qubits[f"Q{index + 1}"] = value
    return {
        "format": "zcu.six-fluxonium",
        "version": 1,
        "mux_gen_ch": 0,
        "pulse_gen_ch": 13,
        "pulse_ro_ch": 6,
        "dc_source": {"name": "flux_bias", "unit": "A", "limits": [-1, 1], "initial": 0, "ramp_step": 0.01},
        "qubits": qubits,
    }
