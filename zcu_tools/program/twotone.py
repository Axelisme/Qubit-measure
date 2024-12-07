from qick import AveragerProgram, RAveragerProgram

from .base import BaseTwoToneProgram


class TwoToneProgram(AveragerProgram, BaseTwoToneProgram):
    def initialize(self):
        return BaseTwoToneProgram.initialize(self)

    def body(self):
        BaseTwoToneProgram.body(self)


class RGainTwoToneProgram(RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        BaseTwoToneProgram.parse_cfg(self)

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]

    def setup_gain_reg(self):
        # setup gain register
        qub_ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.q_gain = self.sreg(qub_ch, "gain")
        self.regwi(self.q_rp, self.q_gain, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_gain_reg()

        self.synci(200)

    def body(self):
        BaseTwoToneProgram.body(self)

    def update(self):
        self.mathi(self.q_rp, self.q_gain, self.q_gain, "+", self.cfg["step"])


class RFreqTwoToneProgram(RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        BaseTwoToneProgram.parse_cfg(self)

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]

    def setup_freq_reg(self):
        # setup freq register
        qub_ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.q_freq = self.sreg(qub_ch, "freq")
        self.regwi(self.q_rp, self.q_freq, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_freq_reg()

        self.synci(200)

    def body(self):
        BaseTwoToneProgram.body(self)

    def update(self):
        self.mathi(self.q_rp, self.q_freq, self.q_freq, "+", self.cfg["step"])
