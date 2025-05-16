from copy import deepcopy
from typing import Any, Dict, Optional


def declare_pulse(
    prog,
    pulse: Dict[str, Any],
    waveform: str,
    ro_ch: Optional[int] = None,
    **kwargs,
):
    prog.declare_gen(pulse["ch"], nqz=pulse["nqz"])
    create_waveform(prog, waveform, pulse)

    add_pulse(prog, pulse, waveform=waveform, ro_ch=ro_ch, **kwargs)

    # add map from waveform to pulse cfg in case we need to access it later
    prog.pulse_map[waveform] = deepcopy(pulse)


def create_waveform(prog, name: str, pulse: Dict[str, Any]):
    ch: int = pulse["ch"]
    style: str = pulse["style"]

    even = style == "flat_top"
    if style == "flat_top":
        pulse = pulse["raise_pulse"]
    length: float = pulse["length"]
    wav_style: str = pulse["style"]

    if wav_style == "const":
        if style == "flat_top":
            raise ValueError("Flat top with constant raise style is not supported")
    elif wav_style == "gauss":
        prog.add_gauss(ch, name, sigma=pulse["sigma"], length=length, even_length=even)
    elif wav_style == "drag":
        prog.add_DRAG(
            ch,
            name,
            sigma=pulse["sigma"],
            length=length,
            delta=pulse["delta"],
            alpha=pulse["alpha"],
            even_length=even,
        )
    elif wav_style == "cosine":
        prog.add_cosine(ch, name, length=length, even_length=even)
    elif wav_style == "flat_top":
        raise ValueError("Nested flat top pulses are not supported")
    else:
        raise ValueError(f"Unknown waveform style: {wav_style}")


def add_pulse(
    prog,
    pulse: Dict[str, Any],
    waveform: str,
    ro_ch: Optional[int] = None,
    **kwargs,
):
    ch: int = pulse["ch"]
    style: str = pulse["style"]

    wav_kwargs = dict(
        style=style, freq=pulse["freq"], phase=pulse["phase"], gain=pulse["gain"]
    )

    if style == "const":
        wav_kwargs["length"] = pulse["length"]
    else:
        assert waveform is not None, f"Waveform is required for {style} pulse"

        wav_kwargs["envelope"] = waveform
        if style == "flat_top":
            # the length register for flat_top only contain the flat part
            wav_kwargs["length"] = pulse["length"] - pulse["raise_pulse"]["length"]

        if style in ["gauss", "cosine", "drag", "arb"]:
            wav_kwargs["style"] = "arb"

    prog.add_pulse(ch, waveform, ro_ch=ro_ch, **wav_kwargs, **kwargs)
