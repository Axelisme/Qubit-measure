import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

from qick.asm_v2 import QickParam

from ..base import MyProgramV2
from .base import Module
from .waveform import make_waveform, set_waveform_param


def check_block_mode(name: str, cfg: Dict[str, Any], want_block: bool) -> None:
    if cfg.get("block_mode") != want_block:
        warnings.warn(
            f"{name} block_mode is {cfg.get('block_mode')}, this may not be what you want"
        )


class Pulse(Module):
    def __init__(
        self,
        name: str,
        cfg: Optional[Dict[str, Any]],
        pulse_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

        self.pulse_name = name if pulse_name is None else pulse_name

    def init(self, prog: MyProgramV2) -> None:
        if self.cfg is None:
            return

        self.waveform = make_waveform(f"{self.name}_waveform", self.cfg["waveform"])

        # if this is the first time to init the pulse, init it
        if not self.has_registered(prog, self.pulse_name):
            self.init_pulse(prog, self.pulse_name)

    def init_pulse(self, prog: MyProgramV2, name: str) -> None:
        cfg = self.cfg
        assert cfg is not None

        ro_ch = cfg.get("ro_ch") if "mixer_freq" in cfg else None
        prog.declare_gen(
            cfg["ch"],
            nqz=cfg["nqz"],
            mixer_freq=cfg.get("mixer_freq"),
            mux_freqs=cfg.get("mux_freqs"),
            mux_gains=cfg.get("mux_gains"),
            mux_phases=cfg.get("mux_phases"),
            ro_ch=ro_ch,
        )

        self.waveform.create(prog, cfg["ch"])

        # derive pulse style
        waveform_cfg = cfg["waveform"]
        wav_style = waveform_cfg["style"]
        wav_kwargs = dict(freq=cfg["freq"], phase=cfg["phase"], gain=cfg["gain"])

        if wav_style == "const":
            wav_kwargs["style"] = "const"
            wav_kwargs["length"] = waveform_cfg["length"]
        else:
            wav_kwargs["envelope"] = self.waveform.name
            if wav_style == "flat_top":
                wav_kwargs["style"] = "flat_top"
                wav_kwargs["length"] = (
                    waveform_cfg["length"] - waveform_cfg["raise_waveform"]["length"]
                )
            else:
                wav_kwargs["style"] = "arb"

        if "mask" in cfg:
            wav_kwargs["mask"] = cfg["mask"]
        if "outsel" in cfg:
            wav_kwargs["outsel"] = cfg["outsel"]

        # add the pulse
        prog.add_pulse(cfg["ch"], name, ro_ch=cfg.get("ro_ch"), **wav_kwargs)

        # register the pulse
        self.register(prog, name)

    def total_length(self) -> float:
        if self.cfg is None:
            return 0.0
        return (
            self.cfg["pre_delay"]
            + self.cfg["waveform"]["length"]
            + self.cfg["post_delay"]
        )

    # -----------------------
    # TODO: better way to share pulse between modules?

    def register(self, prog: MyProgramV2, name: str) -> None:
        if not hasattr(prog, "_module_pulse_list"):
            prog._module_pulse_list = []
        prog._module_pulse_list.append(name)

    def has_registered(self, prog: MyProgramV2, name: str) -> bool:
        if not hasattr(prog, "_module_pulse_list"):
            return False
        return name in prog._module_pulse_list

    # -----------------------

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        cfg = self.cfg

        if cfg is None:
            return t

        pre_delay: float = cfg["pre_delay"]
        length: float = cfg["waveform"]["length"]
        post_delay: float = cfg["post_delay"]

        prog.pulse(cfg["ch"], self.pulse_name, t=t + pre_delay, tag=self.name)

        # block mode is True by default
        if cfg["block_mode"]:
            return t + pre_delay + length + post_delay
        else:
            return t  # no block, return the start time as the end time

    @classmethod
    def set_param(
        cls, pulse_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        if param_name == "on/off":
            pulse_cfg["gain"] = param_value * pulse_cfg["gain"]

            # if the pulse is const or flat_top, also shrink the waveform length
            if pulse_cfg["waveform"]["style"] in ["const", "flat_top"]:
                set_waveform_param(pulse_cfg["waveform"], param_name, param_value)
        elif param_name == "length":
            set_waveform_param(pulse_cfg["waveform"], param_name, param_value)
        elif param_name in ["gain", "freq", "phase"]:
            pulse_cfg[param_name] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
