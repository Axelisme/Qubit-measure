import warnings
from copy import deepcopy
from typing import List, Optional, Type, TypedDict, Union

from qick.asm_v2 import QickParam
from typing_extensions import NotRequired

from ..base import MyProgramV2
from .base import Module
from .util import round_timestamp
from .waveform import Waveform, WaveformCfg


class PulseCfg(TypedDict):
    waveform: WaveformCfg
    ch: int
    nqz: int
    freq: Union[float, QickParam]
    phase: Union[float, QickParam]
    gain: Union[float, QickParam]
    pre_delay: Union[float, QickParam]
    post_delay: Union[float, QickParam]
    block_mode: bool
    mixer_freq: NotRequired[float]
    mux_freqs: NotRequired[List[float]]
    mux_gains: NotRequired[List[float]]
    mux_phases: NotRequired[List[float]]
    ro_ch: NotRequired[int]


def check_block_mode(name: str, cfg: PulseCfg, want_block: bool) -> None:
    if cfg["block_mode"] != want_block:
        warnings.warn(
            f"{name} block_mode is {cfg['block_mode']}, this may not be what you want"
        )


class BasePulse(Module):
    def __init__(
        self, name: str, cfg: Optional[PulseCfg], pulse_name: Optional[str] = None
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

        self.pulse_name = name if pulse_name is None else pulse_name

    def init(self, prog: MyProgramV2) -> None:
        if self.cfg is None:
            return

        self.waveform = Waveform(f"{self.name}_waveform", self.cfg["waveform"])

        # if this is the first time to init the pulse, init it
        if not prog.pulse_registry.has(self.pulse_name):
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
        wav_kwargs = dict(freq=cfg["freq"], phase=cfg["phase"], gain=cfg["gain"])

        if "mask" in cfg:
            wav_kwargs["mask"] = cfg["mask"]
        if "outsel" in cfg:
            wav_kwargs["outsel"] = cfg["outsel"]

        # add the pulse
        prog.add_pulse(
            cfg["ch"],
            name,
            ro_ch=cfg.get("ro_ch"),
            **wav_kwargs,
            **self.waveform.to_wav_kwargs(),
        )

        # register the pulse
        prog.pulse_registry.register(name, cfg)

        if "mixer_freq" in cfg:
            prog.pulse_registry.check_valid_mixer_freq(name, cfg)

    def total_length(self) -> float:
        if self.cfg is None:
            return 0.0
        return (
            self.cfg["pre_delay"]
            + self.cfg["waveform"]["length"]
            + self.cfg["post_delay"]
        )

    # -----------------------

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cfg = self.cfg

        if cfg is None:
            return t

        prog.pulse(cfg["ch"], self.pulse_name, t=t + cfg["pre_delay"], tag=self.name)

        if cfg["block_mode"]:  # default
            return round_timestamp(prog, t + self.total_length(), gen_ch=cfg["ch"])
        else:
            return t  # no block, return the start time as the end time

    @staticmethod
    def set_param(
        pulse_cfg: PulseCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseCfg:
        if param_name == "on/off":
            pulse_cfg["gain"] = param_value * pulse_cfg["gain"]

            # if the pulse is const or flat_top, also shrink the waveform length
            if pulse_cfg["waveform"]["style"] in ["const", "flat_top"]:
                Waveform.set_param(pulse_cfg["waveform"], param_name, param_value)
        elif param_name == "length":
            Waveform.set_param(pulse_cfg["waveform"], param_name, param_value)
        elif param_name in ["gain", "freq", "phase"]:
            pulse_cfg[param_name] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return pulse_cfg


class PaddingPulse(Module):
    def __init__(
        self, name: str, cfg: PulseCfg, pulse_name: Optional[str] = None
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

        if cfg["waveform"]["style"] != "padding":
            raise ValueError("Padding pulse only supports const waveform")

    def init(self, prog: MyProgramV2) -> None:
        cfg = self.cfg
        wav_cfg = cfg["waveform"]

        pre_length: float = wav_cfg["pre_length"]  # type: ignore
        post_length: float = wav_cfg["post_length"]  # type: ignore
        mid_length = wav_cfg["length"] - pre_length - post_length

        # declare waveforms
        self.waveforms = [
            Waveform(
                f"{self.name}_waveform_{i}",
                {"style": "const", "length": length},  # type: ignore
            )
            for i, length in enumerate([pre_length, mid_length, post_length])
        ]

        # declare channel
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

        # derive pulse style
        wav_kwargs = dict(freq=cfg["freq"], phase=cfg["phase"])

        if "mask" in cfg:
            wav_kwargs["mask"] = cfg["mask"]
        if "outsel" in cfg:
            wav_kwargs["outsel"] = cfg["outsel"]

        # add pulses
        gains = [wav_cfg["pre_gain"], cfg["gain"], wav_cfg["post_gain"]]  # type: ignore
        for i, wav in enumerate(self.waveforms):
            wav.create(prog, cfg["ch"])
            prog.add_pulse(
                cfg["ch"],
                f"{self.name}_{i}",
                ro_ch=cfg.get("ro_ch"),
                gain=gains[i],
                **wav_kwargs,
                **wav.to_wav_kwargs(),
            )

    def total_length(self) -> float:
        return (
            self.cfg["pre_delay"]
            + self.cfg["waveform"]["length"]
            + self.cfg["post_delay"]
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cfg = self.cfg

        pre_delay = cfg["pre_delay"]
        post_delay = cfg["post_delay"]

        cur_t = prog.us2cycles(t + pre_delay)
        for i, wav in enumerate(self.waveforms):
            prog.pulse(cfg["ch"], f"{self.name}_{i}", t=prog.cycles2us(cur_t))
            cur_t += prog.us2cycles(wav.waveform_cfg["length"]) + 1
        cur_t += prog.us2cycles(post_delay)

        # block mode is True by default
        if cfg["block_mode"]:
            return prog.cycles2us(cur_t)
        else:
            return t  # no block, return the start time as the end time

    @staticmethod
    def set_param(
        pulse_cfg: PulseCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseCfg:
        if param_name == "on/off":
            pulse_cfg["gain"] = param_value * pulse_cfg["gain"]
            pulse_cfg["waveform"]["pre_length"] = (  # type: ignore
                param_value * (pulse_cfg["waveform"]["pre_length"] - 0.01) + 0.01  # type: ignore
            )
            pulse_cfg["waveform"]["post_length"] = (  # type: ignore
                param_value * (pulse_cfg["waveform"]["post_length"] - 0.01) + 0.01  # type: ignore
            )
            pulse_cfg["waveform"]["length"] = (
                param_value * (pulse_cfg["waveform"]["length"] - 0.03) + 0.03
            )
        elif param_name == "length":
            pulse_cfg["waveform"]["length"] = param_value
        elif param_name in ["freq", "phase"]:
            pulse_cfg[param_name] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return pulse_cfg


class Pulse(Module):
    @classmethod
    def get_pulse_cls(
        cls, cfg: Optional[PulseCfg]
    ) -> Union[Type[BasePulse], Type[PaddingPulse]]:
        if cfg is None or cfg["waveform"]["style"] != "padding":
            return BasePulse
        else:
            return PaddingPulse

    def __init__(
        self, name: str, cfg: Optional[PulseCfg], pulse_name: Optional[str] = None
    ) -> None:
        pulse_cls = self.get_pulse_cls(cfg)
        self.pulse = pulse_cls(name, cfg, pulse_name)  # type: ignore

    def init(self, prog: MyProgramV2) -> None:
        self.pulse.init(prog)

    def total_length(self) -> float:
        return self.pulse.total_length()

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.pulse.run(prog, t)

    @staticmethod
    def set_param(
        pulse_cfg: PulseCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseCfg:
        return Pulse.get_pulse_cls(pulse_cfg).set_param(
            pulse_cfg, param_name, param_value
        )

    @property
    def cfg(self) -> Optional[PulseCfg]:
        return self.pulse.cfg

    @property
    def name(self) -> str:
        return self.pulse.name
