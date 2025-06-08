from myqick.asm_v2 import QickParam

from .base import MyProgramV2, trigger_dual_pulse


class ACStarkProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.stark_res_pulse, "stark_res_pulse")
        self.declare_pulse(self.stark_qub_pulse, "stark_qub_pulse")
        super()._initialize(cfg)

        res_len = self.stark_res_pulse["length"]
        qub_len = self.stark_qub_pulse["length"]
        if isinstance(res_len, QickParam) or isinstance(qub_len, QickParam):
            raise ValueError("Sweep length in stark probe is not supported")

        if qub_len > res_len:
            raise ValueError(
                "Qubit pulse length must be shorter than reset pulse length"
            )

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse with stark probe
        res_len = self.stark_res_pulse["length"]
        qub_len = self.stark_qub_pulse["length"]
        t2 = (res_len - qub_len) / 2
        trigger_dual_pulse(
            self,
            self.stark_res_pulse,
            self.stark_qub_pulse,
            "stark_res_pulse",
            "stark_qub_pulse",
            t2=t2,
        )

        # measure
        self.readoutM.readout_qubit(self)
