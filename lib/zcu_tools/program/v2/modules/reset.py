from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

from ..base import MyProgramV2
from .base import Module, str2module
from .pulse import Pulse

if TYPE_CHECKING:
    from ..modules import ModuleLibrary


class AbsReset(Module):
    pass


def derive_reset_cfg(
    ml: ModuleLibrary, reset_cfg: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if reset_cfg is None:
        return None

    reset_cfg = str2module(ml, reset_cfg)

    reset_type = reset_cfg["type"]
    if reset_type == "none":
        return NoneReset.derive_cfg(ml, reset_cfg)
    elif reset_type == "pulse":
        return PulseReset.derive_cfg(ml, reset_cfg)
    elif reset_type == "two_pulse":
        return TwoPulseReset.derive_cfg(ml, reset_cfg)
    else:
        raise ValueError(f"Unknown reset type: {reset_type}")


def make_reset(name: str, reset_cfg: Optional[Dict[str, Any]]) -> AbsReset:
    if reset_cfg is None:
        return NoneReset()

    reset_type = reset_cfg["type"]

    if reset_type == "none":
        return NoneReset()
    elif reset_type == "pulse":
        return PulseReset(
            name,
            pulse_cfg=reset_cfg["pulse_cfg"],
            post_pulse_cfg=reset_cfg.get("post_pulse_cfg"),
        )
    elif reset_type == "two_pulse":
        return TwoPulseReset(
            name,
            pulse1_cfg=reset_cfg["pulse1_cfg"],
            pulse2_cfg=reset_cfg["pulse2_cfg"],
            post_pulse_cfg=reset_cfg.get("post_pulse_cfg"),
        )
    else:
        raise ValueError(f"Unknown reset type: {reset_type}")


class NoneReset(AbsReset):
    def init(self, prog: MyProgramV2) -> None:
        pass

    @classmethod
    def derive_cfg(
        cls, ml: ModuleLibrary, module_cfg: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if module_cfg is None:
            return None

        if module_cfg["type"] != "none":
            raise ValueError("NoneReset can only derive from 'none' type")

        return module_cfg

    def run(self, prog: MyProgramV2) -> None:
        pass


class PulseReset(AbsReset):
    def __init__(
        self,
        name: str,
        pulse_cfg: Dict[str, Any],
        post_pulse_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg)

        if post_pulse_cfg is not None:
            self.post_reset_pulse = Pulse(name=f"{name}_post_pulse", cfg=post_pulse_cfg)
        else:
            self.post_reset_pulse = None

    @classmethod
    def derive_cfg(
        cls, ml: ModuleLibrary, module_cfg: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        module_cfg = str2module(ml, module_cfg)

        if module_cfg["type"] != "pulse":
            raise ValueError("PulseReset can only derive from 'pulse' type")

        module_cfg["pulse_cfg"] = str2module(ml, module_cfg["pulse_cfg"])
        if "post_pulse_cfg" in module_cfg:
            module_cfg["post_pulse_cfg"] = str2module(ml, module_cfg["post_pulse_cfg"])

        return module_cfg

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse.init(prog)

        if self.post_reset_pulse is not None:
            self.post_reset_pulse.init(prog)

    def run(self, prog: MyProgramV2) -> None:
        self.reset_pulse.run(prog)

        if self.post_reset_pulse is not None:
            self.post_reset_pulse.run(prog)


class TwoPulseReset(AbsReset):
    def __init__(
        self,
        name: str,
        pulse1_cfg: Dict[str, Any],
        pulse2_cfg: Dict[str, Any],
        post_pulse_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name

        self.reset_pulse1 = Pulse(name=f"{name}_pulse1", cfg=pulse1_cfg)
        self.reset_pulse2 = Pulse(name=f"{name}_pulse2", cfg=pulse2_cfg)

        if post_pulse_cfg is not None:
            self.post_reset_pulse = Pulse(name=f"{name}_post_pulse", cfg=post_pulse_cfg)
        else:
            self.post_reset_pulse = None

    @classmethod
    def derive_cfg(
        cls, ml: ModuleLibrary, module_cfg: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        module_cfg = str2module(ml, module_cfg)

        if module_cfg["type"] != "two_pulse":
            raise ValueError("TwoPulseReset can only derive from 'two_pulse' type")

        module_cfg["pulse1_cfg"].setdefault("post_delay", None)

        module_cfg["pulse1_cfg"] = str2module(ml, module_cfg["pulse1_cfg"])
        module_cfg["pulse2_cfg"] = str2module(ml, module_cfg["pulse2_cfg"])
        if "post_pulse_cfg" in module_cfg:
            module_cfg["post_pulse_cfg"] = str2module(ml, module_cfg["post_pulse_cfg"])

        return module_cfg

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse1.init(prog)
        self.reset_pulse2.init(prog)

        if self.post_reset_pulse is not None:
            self.post_reset_pulse.init(prog)

    def run(self, prog: MyProgramV2) -> None:
        self.reset_pulse1.run(prog)
        self.reset_pulse2.run(prog)

        if self.post_reset_pulse is not None:
            self.post_reset_pulse.run(prog)
