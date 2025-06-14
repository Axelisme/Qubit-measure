from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

from .module import Module

if TYPE_CHECKING:
    from .program import MyProgramV2


def force_no_post_delay(pulse: Dict[str, Any], name: str) -> None:
    if pulse.get("post_delay") is not None:
        warnings.warn(f"{name} has a post_delay, which will be ignored")
    pulse["post_delay"] = None


class Pulse(Module):
    def __init__(self, name: str, cfg: dict, ro_ch: Optional[int] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.ro_ch = ro_ch

    def init(self, prog: MyProgramV2) -> None:
        prog.declare_gen(
            self.cfg["ch"],
            nqz=self.cfg["nqz"],
            mixer_freq=self.cfg.get("mixer_freq"),
            mux_freqs=self.cfg.get("mux_freqs"),
            mux_gains=self.cfg.get("mux_gains"),
            mux_phases=self.cfg.get("mux_phases"),
            ro_ch=self.cfg.get("ro_ch"),
        )
        self.create_waveform(prog)

        self.add_pulse(prog)

    def create_waveform(self, prog: MyProgramV2) -> None:
        cfg = self.cfg

        ch: int = cfg["ch"]
        style: str = cfg["style"]

        even = style == "flat_top"
        if style == "flat_top":
            cfg = cfg["raise_pulse"]
        length: float = cfg["length"]
        wav_style: str = cfg["style"]

        if wav_style == "const":
            if style == "flat_top":
                raise ValueError("Flat top with constant raise style is not supported")
        elif wav_style == "gauss":
            prog.add_gauss(
                ch, self.name, sigma=cfg["sigma"], length=length, even_length=even
            )
        elif wav_style == "drag":
            prog.add_DRAG(
                ch,
                self.name,
                sigma=cfg["sigma"],
                length=length,
                delta=cfg["delta"],
                alpha=cfg["alpha"],
                even_length=even,
            )
        elif wav_style == "cosine":
            prog.add_cosine(ch, self.name, length=length, even_length=even)
        elif wav_style == "flat_top":
            raise ValueError("Nested flat top pulses are not supported")
        else:
            raise ValueError(f"Unknown waveform style: {wav_style}")

    def add_pulse(self, prog: MyProgramV2) -> None:
        cfg = self.cfg

        ch: int = cfg["ch"]
        style: str = cfg["style"]

        wav_kwargs = dict(
            style=style, freq=cfg["freq"], phase=cfg["phase"], gain=cfg["gain"]
        )

        if style == "const":
            wav_kwargs["length"] = cfg["length"]
        else:
            wav_kwargs["envelope"] = self.name
            if style == "flat_top":
                # the length register for flat_top only contain the flat part
                wav_kwargs["length"] = cfg["length"] - cfg["raise_pulse"]["length"]

            if style in ["gauss", "cosine", "drag", "arb"]:
                wav_kwargs["style"] = "arb"

        if "mask" in cfg:
            wav_kwargs["mask"] = cfg["mask"]

        prog.add_pulse(ch, self.name, ro_ch=self.ro_ch, **wav_kwargs)

    def run(self, prog: MyProgramV2) -> None:
        cfg = self.cfg

        prog.pulse(
            cfg["ch"],
            self.name,
            t=cfg.get("t", 0.0),
            tag=self.name,
        )

        post_delay = cfg.get("post_delay", 0.0)
        if post_delay is not None:
            prog.delay_auto(post_delay, ros=False, tag=f"{self.name}_post_delay")
