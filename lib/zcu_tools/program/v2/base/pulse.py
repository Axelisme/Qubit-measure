from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .program import MyProgramV2


def force_no_post_delay(pulse: Dict[str, Any], name: str) -> None:
    if pulse.get("post_delay") is not None:
        warnings.warn(f"{name} has a post_delay, which will be ignored")
    pulse["post_delay"] = None


def trigger_pulse(prog: MyProgramV2, pulse: Dict[str, Any], name: str) -> None:
    ch = pulse["ch"]
    post_delay = pulse.get("post_delay", 0.0)
    t = pulse.get("t", 0.0)

    prog.pulse(ch, name, t=t, tag=f"ch{ch}_{name}")

    if post_delay is not None:
        prog.delay_auto(post_delay, ros=False, tag=f"ch{ch}_{name}_post_delay")


def create_waveform(prog: MyProgramV2, name: str, pulse: Dict[str, Any]) -> None:
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
    prog: MyProgramV2,
    pulse: Dict[str, Any],
    waveform: str,
    ro_ch: Optional[int] = None,
    **kwargs,
) -> None:
    ch: int = pulse["ch"]
    style: str = pulse["style"]

    wav_kwargs = dict(
        style=style, freq=pulse["freq"], phase=pulse["phase"], gain=pulse["gain"]
    )

    if style == "const":
        wav_kwargs["length"] = pulse["length"]
    else:
        wav_kwargs["envelope"] = waveform
        if style == "flat_top":
            # the length register for flat_top only contain the flat part
            wav_kwargs["length"] = pulse["length"] - pulse["raise_pulse"]["length"]

        if style in ["gauss", "cosine", "drag", "arb"]:
            wav_kwargs["style"] = "arb"

    if "mask" in pulse:
        wav_kwargs["mask"] = pulse["mask"]

    prog.add_pulse(ch, waveform, ro_ch=ro_ch, **wav_kwargs, **kwargs)
