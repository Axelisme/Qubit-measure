from typing import Optional

from ..base import MyProgramV2, add_pulse, create_waveform
from .base import Module


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
            ro_ch=self.ro_ch,
        )

        create_waveform(prog, self.name, self.cfg)
        add_pulse(prog, self.cfg, self.name, ro_ch=self.ro_ch)

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
