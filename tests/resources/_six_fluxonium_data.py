"""Synthetic six-cavity material, independent of lab defaults."""

from tests.experiment.single_fluxonium._inputs import settings


def six_document():
    qubits = {}
    for index in range(6):
        value = settings().model_dump(mode="json")
        value["dc_bias"]["source"]["name"] = f"flux_bias_{index + 1}"
        value["drive_line"].update(name=f"mw_{index}", gen_ch=index + 1)
        value["flux_line"].update(name=f"ff_{index}", gen_ch=index + 7)
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
        "qubits": qubits,
    }
