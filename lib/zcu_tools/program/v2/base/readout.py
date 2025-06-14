from abc import ABC, abstractmethod
from typing import Any, Dict, List

from myqick.asm_v2 import AveragerProgramV2

from .pulse import force_no_post_delay, trigger_pulse


class AbsReadout(ABC):
    @abstractmethod
    def init(self, prog: AveragerProgramV2):
        pass

    @abstractmethod
    def readout_qubit(self, prog: AveragerProgramV2):
        pass


def make_readout(name: str) -> AbsReadout:
    if name == "base":
        return BaseReadout()
    elif name == "two_pulse":
        return TwoPulseReadout()
    else:
        raise ValueError(f"Unknown readout type: {name}")


class BaseReadout(AbsReadout):
    def init(self, prog: AveragerProgramV2):
        res_pulse: Dict[str, Any] = prog.res_pulse
        res_ch: int = res_pulse["ch"]
        ro_chs: List[int] = prog.adc["chs"]

        # TODO: support multiple readout channels
        assert len(ro_chs) == 1, "Only one readout channel is supported"
        ro_ch = ro_chs[0]

        # TODO: support post delay
        force_no_post_delay(res_pulse, "res_pulse")

        prog.declare_pulse(res_pulse, "res_pulse", ro_ch)

        prog.declare_readout(ch=ro_ch, length=prog.adc["ro_length"])
        prog.add_readoutconfig(
            ch=ro_ch, name="readout_adc", freq=res_pulse["freq"], gen_ch=res_ch
        )

    def readout_qubit(self, prog: AveragerProgramV2):
        ro_ch: int = prog.adc["chs"][0]

        prog.send_readoutconfig(ro_ch, "readout_adc", t=0)

        trigger_pulse(prog, prog.res_pulse, "res_pulse")

        prog.trigger([ro_ch], t=prog.adc["trig_offset"])


class TwoPulseReadout(AbsReadout):
    def init(self, prog: AveragerProgramV2):
        res_pulse: Dict[str, Any] = prog.res_pulse
        res_ch: int = res_pulse["ch"]
        ro_chs: List[int] = prog.adc["chs"]

        # TODO: support multiple readout channels
        assert len(ro_chs) == 1, "Only one readout channel is supported"
        ro_ch = ro_chs[0]

        # TODO: support post delay
        force_no_post_delay(res_pulse, "res_pulse")

        prog.declare_pulse(prog.pre_res_pulse, "pre_res_pulse")
        prog.declare_pulse(res_pulse, "res_pulse", ro_ch)

        prog.declare_readout(ch=ro_ch, length=prog.adc["ro_length"])
        prog.add_readoutconfig(
            ch=ro_ch, name="readout_adc", freq=res_pulse["freq"], gen_ch=res_ch
        )

    def readout_qubit(self, prog: AveragerProgramV2):
        ro_ch: int = prog.adc["chs"][0]

        prog.send_readoutconfig(ro_ch, "readout_adc", t=0)

        trigger_pulse(prog, prog.pre_res_pulse, "pre_res_pulse")
        trigger_pulse(prog, prog.res_pulse, "res_pulse")

        prog.trigger([ro_ch], t=prog.adc["trig_offset"])
