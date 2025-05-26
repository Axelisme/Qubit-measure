from abc import ABC, abstractmethod
from typing import Any, Dict, List


from myqick.asm_v2 import AveragerProgramV2


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
    else:
        raise ValueError(f"Unknown readout type: {name}")


class BaseReadout(AbsReadout):
    def init(self, prog: AveragerProgramV2):
        res_pulse: Dict[str, Any] = prog.res_pulse
        res_ch: int = res_pulse["ch"]
        ro_chs: List[int] = prog.adc["chs"]

        assert len(ro_chs) == 1, "Only one readout channel is supported"
        ro_ch = ro_chs[0]

        prog.declare_pulse(res_pulse, "res_pulse", ro_chs[0])

        prog.declare_readout(ch=ro_ch, length=prog.adc["ro_length"])
        prog.add_readoutconfig(
            ch=ro_ch, name="readout_adc", freq=res_pulse["freq"], gen_ch=res_ch
        )

    def readout_qubit(self, prog: AveragerProgramV2):
        ro_ch = prog.adc["chs"][0]

        prog.send_readoutconfig(ro_ch, "readout_adc", t=0)
        prog.pulse(prog.res_pulse["ch"], "res_pulse", t="auto")  # pyright: ignore[reportArgumentType]
        prog.delay_auto(t=prog.adc["trig_offset"], gens=False, tag="trig_offset")
        prog.trigger([ro_ch], t=None)
