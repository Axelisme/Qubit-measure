from typing import Any, Dict, Optional

from myqick.asm_v2 import AveragerProgramV2


def trigger_pulse(prog, pulse: Dict[str, Any]) -> None:
    pre_delay = pulse.get("pre_delay")
    post_delay = pulse.get("post_delay")

    if pre_delay is not None:
        prog.delay_auto(pre_delay, ros=False)

    prog.pulse(pulse["ch"], "qub_pulse")

    if post_delay is not None:
        prog.delay_auto(post_delay, ros=False)


def create_waveform(prog: AveragerProgramV2, name: str, pulse: Dict[str, Any]):
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
    prog: AveragerProgramV2,
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

    if "mask" in pulse:
        wav_kwargs["mask"] = pulse["mask"]

    prog.add_pulse(ch, waveform, ro_ch=ro_ch, **wav_kwargs, **kwargs)
