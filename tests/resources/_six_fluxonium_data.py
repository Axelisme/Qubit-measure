"""Synthetic six-cavity material, independent of lab defaults."""

from typing import NotRequired, TypedDict

from tests.experiment.single_fluxonium._inputs import settings


class ChipDocument(TypedDict):
    qubit: dict[str, float]
    temperature_k: float


class ReadoutDocument(TypedDict):
    drive: dict[str, float]
    resonator: dict[str, float]
    adc: dict[str, object]


class BiasDocument(TypedDict):
    source: dict[str, object]
    offset: float
    period: float
    coil_offset: float
    coil_period: float


class QubitDocument(TypedDict):
    format: str
    version: int
    chip: ChipDocument
    dc_bias: BiasDocument
    drive_line: dict[str, object]
    readout: ReadoutDocument
    numerics: dict[str, object]


class SixDocument(TypedDict):
    format: str
    version: int
    mux_gen_ch: int
    pulse_gen_ch: int
    pulse_ro_ch: int
    dc_source: dict[str, object]
    qubits: dict[str, QubitDocument]
    surprise: NotRequired[bool]


def six_document() -> SixDocument:
    qubits: dict[str, QubitDocument] = {}
    for index in range(6):
        value = settings().model_dump(mode="json")
        del value["flux_line"]
        period = 1.0 + 0.1 * index
        coil_period = -2.0 - 0.2 * index
        value["dc_bias"] = BiasDocument(
            source={
                "name": f"Q{index + 1}_bias",
                "unit": "A",
                "limits": [-1, 1],
                "initial": 0,
                "ramp_step": 0.01,
            },
            offset=-(0.25 + 0.01 * index) * period,
            period=period,
            coil_offset=-0.15 * coil_period,
            coil_period=coil_period,
        )
        value["drive_line"].update(name=f"mw_{index}", gen_ch=index + 1)
        value["readout"]["resonator"].update(frequency_ghz=6.8 + 0.125 * index)
        value["readout"]["adc"].update(ro_ch=index, gain=10000)
        value["numerics"].update(time_scale=0)
        qubits[f"Q{index + 1}"] = QubitDocument(
            format=value["format"],
            version=value["version"],
            chip=value["chip"],
            dc_bias=value["dc_bias"],
            drive_line=value["drive_line"],
            readout=value["readout"],
            numerics=value["numerics"],
        )
    return {
        "format": "zcu.six-fluxonium",
        "version": 1,
        "mux_gen_ch": 0,
        "pulse_gen_ch": 13,
        "pulse_ro_ch": 6,
        "dc_source": {
            "name": "coil_bias",
            "unit": "A",
            "limits": [-1, 1],
            "initial": 0,
            "ramp_step": 0.01,
        },
        "qubits": qubits,
    }
