from __future__ import annotations
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Any, NotRequired, Optional, Union, cast

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .util import round_timestamp
from .waveform import Waveform, WaveformCfg

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class PulseCfg(ModuleCfg, closed=True):
    waveform: WaveformCfg
    ch: int
    nqz: int
    freq: Union[float, QickParam]
    phase: Union[float, QickParam]
    gain: Union[float, QickParam]
    pre_delay: Union[float, QickParam]
    post_delay: Union[float, QickParam]

    mixer_freq: NotRequired[float]
    mux_freqs: NotRequired[list[float]]
    mux_gains: NotRequired[list[float]]
    mux_phases: NotRequired[list[float]]
    mask: NotRequired[list[int]]
    outsel: NotRequired[int]
    ro_ch: NotRequired[int]


class Pulse(Module, tag="pulse"):
    def __init__(
        self,
        name: str,
        cfg: Optional[PulseCfg],
        tag: Optional[str] = None,
        block_mode: bool = True,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

        self.tag = tag
        self.block_mode = block_mode

    def init(self, prog: MyProgramV2) -> None:
        if self.cfg is None:
            return

        self.waveform = Waveform(f"{self.name}_waveform", self.cfg["waveform"])
        self.pulse_id = prog.pulse_registry.calc_name(self.cfg)

        prog.pulse_registry.check_valid_mixer_freq(self.name, self.cfg)

        # if this is the first time to init the pulse, init it
        # if not, re-use the pulse
        if prog.pulse_registry.register(self.name, self.cfg):
            self.init_pulse(prog, self.pulse_id)

    def init_pulse(self, prog: MyProgramV2, pulse_id: str) -> None:
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
        wav_kwargs = dict[str, Any](
            freq=cfg["freq"], phase=cfg["phase"], gain=cfg["gain"]
        )

        if "mask" in cfg:
            wav_kwargs["mask"] = cfg["mask"]
        if "outsel" in cfg:
            wav_kwargs["outsel"] = cfg["outsel"]

        # add the pulse
        prog.add_pulse(
            cfg["ch"],
            pulse_id,
            ro_ch=cfg.get("ro_ch"),
            **wav_kwargs,
            **self.waveform.to_wav_kwargs(),
        )

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        if self.cfg is None:
            return 0.0
        return round_timestamp(
            prog,
            (
                round_timestamp(prog, self.cfg["pre_delay"])
                + round_timestamp(prog, self.waveform.length, gen_ch=self.cfg["ch"])
                + round_timestamp(prog, self.cfg["post_delay"])
            ),
        )

    # -----------------------

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cfg = self.cfg

        if cfg is None:
            return t

        prog.pulse(cfg["ch"], self.pulse_id, t=t + cfg["pre_delay"], tag=self.tag)

        if self.block_mode:  # default
            return t + self.total_length(prog)
        else:  # no block, return the start time as the end time
            return t

    @staticmethod
    def set_param(
        cfg: PulseCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseCfg:
        if param_name == "length":
            Waveform.set_param(cfg["waveform"], param_name, param_value)
        elif param_name in ["gain", "freq", "phase"]:
            cfg[param_name] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return cfg

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> PulseCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "pulse"
        cfg.setdefault("phase", 0.0)
        cfg.setdefault("pre_delay", 0.0)
        cfg.setdefault("post_delay", 0.0)

        if isinstance(cfg["waveform"], str):
            cfg["waveform"] = ml.get_waveform(cfg["waveform"])

        return cast(PulseCfg, cfg)
