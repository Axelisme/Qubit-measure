from qick import RAveragerProgram

from .base import BaseEFProgram, set_pulse


class EFAmpRabiProgram(RAveragerProgram, BaseEFProgram):
    def parse_cfg(self):
        super().parse_cfg()

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]

        self.ef_pulse["gain"] = self.cfg["start"]

    def setup_gain(self):
        qub_ch = self.qub_cfg["qub_ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.r_gain = self.sreg(qub_ch, "gain")
        self.r_ef_gain = 3
        self.regwi(self.q_rp, self.r_ef_gain, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()
        self.setup_gain()

        self.synci(200)

    def pulse_ef(self):
        qub_ch = self.qub_cfg["qub_ch"]

        set_pulse(self, self.ef_pulse, qub_ch, waveform=self.ef_wavform)
        self.mathi(self.q_rp, self.r_gain, self.r_ef_gain, "+", 0)  # overwrite ef gain
        self.pulse(ch=qub_ch)

    def update(self):
        # update ef gain
        self.mathi(self.q_rp, self.r_ef_gain, self.r_ef_gain, "+", self.cfg["step"])
