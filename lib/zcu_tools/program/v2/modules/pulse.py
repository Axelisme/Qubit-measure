from __future__ import annotations
import warnings
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
from copy import deepcopy

import qick.asm_v2 as qasm

from ..base import MyProgramV2, add_pulse, create_waveform
from .base import Module, str2module

if TYPE_CHECKING:
    from ..modules import ModuleLibrary


def check_no_post_delay(cfg: Dict[str, Any], name: str) -> None:
    if cfg.get("post_delay") is not None:
        warnings.warn(
            f"{name} has post_delay, this may potentially make two pulses not overlap. "
            "\nForce set post_delay to None."
        )
    cfg["post_delay"] = None


class DelayOn(qasm.Macro):
    def __init__(self, gen_ch: int, t: float) -> None:
        self.gen_ch = gen_ch
        self.t = t

    def preprocess(self, prog: MyProgramV2) -> None:
        cur_t = prog.get_timestamp(gen_ch=self.gen_ch)
        prog.set_timestamp(cur_t + self.t, gen_ch=self.gen_ch)

    def expand(self, prog: MyProgramV2) -> None:
        return []


class Pulse(Module):
    def __init__(
        self,
        name: str,
        cfg: Optional[Dict[str, Any]],
        ro_ch: Optional[int] = None,
        pulse_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.ro_ch = ro_ch

        if pulse_name is None:
            self.pulse_name = name
        else:
            self.pulse_name = pulse_name

    @classmethod
    def derive_cfg(
        cls, ml: ModuleLibrary, module_cfg: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if module_cfg is None:
            return None

        module_cfg = str2module(ml, module_cfg)

        # if it also a pulse cfg, exclude raise_pulse in flat_top
        module_cfg.setdefault("phase", 0.0)
        module_cfg.setdefault("t", 0.0)
        module_cfg.setdefault("post_delay", 0.0)

        GAUSS_RATIO = 0.25

        style = module_cfg["style"]
        if style == "gauss":
            length = module_cfg.get("length")
            if length is not None:
                module_cfg.setdefault("sigma", length * GAUSS_RATIO)
        elif style == "flat_top":
            raise_pulse = module_cfg["raise_pulse"]
            if raise_pulse["style"] == "gauss":
                raise_length = raise_pulse.get("length")
                if raise_length is not None:
                    raise_pulse.setdefault("sigma", raise_length * GAUSS_RATIO)

        return module_cfg

    def init(self, prog: MyProgramV2) -> None:
        if self.cfg is None:
            return

        # if pulse already declared, skip
        if self.has_registered(prog, self.pulse_name):
            return

        self.init_pulse(prog, self.pulse_name)

    # -----------------------
    # TODO: better way to share pulse between modules

    def register(self, prog: MyProgramV2, name: str) -> None:
        if not hasattr(prog, "_module_pulse_list"):
            prog._module_pulse_list = []
        prog._module_pulse_list.append(name)

    def has_registered(self, prog: MyProgramV2, name: str) -> bool:
        if not hasattr(prog, "_module_pulse_list"):
            return False
        return name in prog._module_pulse_list

    # -----------------------

    def init_pulse(self, prog: MyProgramV2, name: str) -> None:
        ro_ch = self.ro_ch

        prog.declare_gen(
            self.cfg["ch"],
            nqz=self.cfg["nqz"],
            mixer_freq=self.cfg.get("mixer_freq"),
            mux_freqs=self.cfg.get("mux_freqs"),
            mux_gains=self.cfg.get("mux_gains"),
            mux_phases=self.cfg.get("mux_phases"),
            ro_ch=ro_ch,
        )

        create_waveform(prog, name, self.cfg)
        add_pulse(prog, self.cfg, name, ro_ch=self.ro_ch)

        self.register(prog, name)

    def run(self, prog: MyProgramV2) -> None:
        cfg = self.cfg

        if cfg is None:
            return

        t = cfg["t"]

        # directly set puls(t = t, ...) will use absolute time
        # this make t relative to last pulse end,
        # TODO: other non-hacky way to do this?
        if t != "auto":
            prog.append_macro(DelayOn(cfg["ch"], t))
            t = "auto"

        prog.pulse(cfg["ch"], self.pulse_name, t=t, tag=self.name)

        post_delay = cfg["post_delay"]
        if post_delay is not None:
            prog.delay_auto(post_delay, ros=False, tag=f"{self.name}_post_delay")
