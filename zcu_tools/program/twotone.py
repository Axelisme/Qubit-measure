from qick import AveragerProgram, RAveragerProgram

from .base import BaseTwoToneProgram


class TwoToneProgram(AveragerProgram, BaseTwoToneProgram):
    def initialize(self):
        return BaseTwoToneProgram.initialize(self)

    def body(self):
        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=self.qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()


class RGainTwoToneProgram(RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        BaseTwoToneProgram.parse_cfg(self)

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]

    def setup_gain_reg(self):
        # setup gain register
        ch = self.res_cfg["qub_ch"]
        self.q_rp = self.ch_page(ch)
        self.q_gain = self.sreg(ch, "gain")
        self.regwi(self.q_rp, self.q_gain, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_gain_reg()

        self.synci(200)

    def body(self):
        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=self.qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()

    def update(self):
        self.mathi(self.q_rp, self.q_gain, self.q_gain, "+", self.cfg["step"])
