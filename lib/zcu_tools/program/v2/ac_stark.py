from .base import MyProgramV2, trigger_pulse


class ACStarkProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.stark_res_pulse, "stark_res_pulse")
        self.declare_pulse(self.stark_qub_pulse, "stark_qub_pulse")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse with stark probe
        trigger_pulse(self, self.stark_res_pulse, "stark_res_pulse")
        trigger_pulse(self, self.stark_qub_pulse, "stark_qub_pulse")

        # measure
        self.readoutM.readout_qubit(self)
