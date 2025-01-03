from qick import RAveragerProgram

from .base import BaseTwoToneProgram


class AmpRabiProgram(RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        super().parse_cfg()

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]
        
        self.qub_pulse['gain'] = self.cfg['start']

    def setup_gain_reg(self):
        qub_ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.r_gain = self.sreg(qub_ch, "gain")
        self.regwi(self.q_rp, self.r_gain, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_readout()
        self.setup_qubit()
        self.setup_gain_reg()

        self.synci(200)

    def body(self):
        BaseTwoToneProgram.body(self)

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.cfg["step"])
