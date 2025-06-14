from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from .module import Module
from .pulse import Pulse

if TYPE_CHECKING:
    from .program import MyProgramV2


class AbsReset(Module):
    pass


def make_reset(reset_cfg: Dict[str, Any]) -> AbsReset:
    name = reset_cfg["type"]

    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset(
            name,
            pulse_cfg=reset_cfg["pulse_cfg"],
            post_pulse_cfg=reset_cfg.get("post_pulse_cfg"),
        )
    elif name == "two_pulse":
        return TwoPulseReset(
            name,
            pulse1_cfg=reset_cfg["pulse1_cfg"],
            pulse2_cfg=reset_cfg["pulse2_cfg"],
            post_pulse_cfg=reset_cfg.get("post_pulse_cfg"),
        )
    else:
        raise ValueError(f"Unknown reset type: {name}")


class NoneReset(AbsReset):
    def init(self, _) -> None:
        pass

    def reset_qubit(self, _) -> None:
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
