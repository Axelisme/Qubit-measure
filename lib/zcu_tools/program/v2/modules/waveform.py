from abc import ABC, abstractmethod
from typing import Any, Dict

from qick.asm_v2 import QickParam

from ..base import MyProgramV2


class AbsWaveform(ABC):
    SUPPORT_STYLES = []

    def __init__(self, name: str, waveform_cfg: Dict[str, Any]) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

        style = self.waveform_cfg["style"]
        if style not in self.SUPPORT_STYLES:
            raise ValueError(f"Support style: {self.SUPPORT_STYLES}, got {style}")

    @abstractmethod
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        pass

    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )


class ConstWaveform(AbsWaveform):
    SUPPORT_STYLES = ["const"]

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg

        assert wav_cfg["style"] in self.SUPPORT_STYLES

    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        if param_name == "on/off":
            waveform_cfg["length"] = param_value * waveform_cfg["length"] + 0.01
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


class CosineWaveform(AbsWaveform):
    SUPPORT_STYLES = ["cosine"]

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg

        assert wav_cfg["style"] in self.SUPPORT_STYLES

        length: float = wav_cfg["length"]
        prog.add_cosine(ch, self.name, length=length, **kwargs)


class GaussWaveform(AbsWaveform):
    SUPPORT_STYLES = ["gauss"]

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg

        assert wav_cfg["style"] in self.SUPPORT_STYLES

        length: float = wav_cfg["length"]
        sigma: float = wav_cfg["sigma"]
        prog.add_gauss(ch, self.name, sigma=sigma, length=length, **kwargs)


class DragWaveform(AbsWaveform):
    SUPPORT_STYLES = ["drag"]

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg

        assert wav_cfg["style"] in self.SUPPORT_STYLES

        length: float = wav_cfg["length"]
        sigma: float = wav_cfg["sigma"]
        delta: float = wav_cfg["delta"]
        alpha: float = wav_cfg["alpha"]
        prog.add_DRAG(
            ch,
            self.name,
            sigma=sigma,
            length=length,
            delta=delta,
            alpha=alpha,
            **kwargs,
        )


class FlatTopWaveform(AbsWaveform):
    SUPPORT_STYLES = ["flat_top"]

    def __init__(self, name: str, waveform_cfg: Dict[str, Any]) -> None:
        super().__init__(name, waveform_cfg)

        raise_cfg = waveform_cfg["raise_waveform"]
        if raise_cfg["style"] == "flat_top":
            raise ValueError("Nested flat top pulses are not supported")

        if raise_cfg["style"] == "const":
            raise ValueError("Flat top with constant raise style is not supported")

        self.raise_waveform = make_waveform(name, raise_cfg)

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg

        assert wav_cfg["style"] in self.SUPPORT_STYLES

        kwargs.setdefault("even_length", True)

        self.raise_waveform.create(prog, ch, **kwargs)

    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        if param_name == "on/off":
            waveform_cfg["length"] = (
                param_value * waveform_cfg["length"]
                + waveform_cfg["raise_waveform"]["length"]
                + 0.01
            )
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


def make_waveform(name: str, waveform_cfg: Dict[str, Any]) -> AbsWaveform:
    style = waveform_cfg["style"]
    if style == "const":
        return ConstWaveform(name, waveform_cfg)
    elif style == "cosine":
        return CosineWaveform(name, waveform_cfg)
    elif style == "gauss":
        return GaussWaveform(name, waveform_cfg)
    elif style == "drag":
        return DragWaveform(name, waveform_cfg)
    elif style == "flat_top":
        return FlatTopWaveform(name, waveform_cfg)
    else:
        raise ValueError(f"Unknown waveform style: {style}")


def set_waveform_param(
    waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
) -> None:
    style = waveform_cfg["style"]
    if style == "const":
        return ConstWaveform.set_param(waveform_cfg, param_name, param_value)
    elif style == "cosine":
        return CosineWaveform.set_param(waveform_cfg, param_name, param_value)
    elif style == "gauss":
        return GaussWaveform.set_param(waveform_cfg, param_name, param_value)
    elif style == "drag":
        return DragWaveform.set_param(waveform_cfg, param_name, param_value)
    elif style == "flat_top":
        return FlatTopWaveform.set_param(waveform_cfg, param_name, param_value)
    else:
        raise ValueError(f"Unknown waveform style: {style}")
