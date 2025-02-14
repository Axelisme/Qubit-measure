from .twotone import RGainTwoToneProgram


class SingleShotProgram(RGainTwoToneProgram):
    def __init__(self, soccfg, cfg):
        cfg["start"] = 0
        cfg["step"] = 1
        cfg["expts"] = 2 * cfg["shots"]
        super().__init__(soccfg, cfg)

    def declare_gain_reg(self):
        ch = self.qub_pulse["ch"]  # type: ignore
        self.q_rp = self.ch_page(ch)
        self.q_gain = self.sreg(ch, "gain")
        self.q_gain2 = self.sreg(ch, "gain2")
        self.q_gain_t = 3
        self.q_gain_0 = 4
        self.mathi(self.q_rp, self.q_gain_t, self.q_gain, "+", 0)
        self.regwi(self.q_rp, self.q_gain_0, 0)

    def update(self):
        # swap r_gain_t and r_gain_0
        self.bitw(self.q_rp, self.q_gain_t, self.q_gain_t, "^", self.q_gain_0)
        self.bitw(self.q_rp, self.q_gain_0, self.q_gain_0, "^", self.q_gain_t)

    def acquire(self, *args, **kwargs):
        ro_length = self.us2cycles(self.adc["ro_length"], ro_ch=self.adc["chs"][0])  # type: ignore
        _, avgi, avgq = super().acquire(*args, **kwargs)
        avgi = avgi[0][0].reshape(-1, 2).T / ro_length
        avgq = avgq[0][0].reshape(-1, 2).T / ro_length
        return avgi, avgq
