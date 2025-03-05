from .twotone import RGainTwoToneProgram


class SingleShotProgram(RGainTwoToneProgram):
    def __init__(self, soccfg, cfg):
        cfg["sweep"] = dict(start=0, step=1, expts=2 * cfg["reps"])
        cfg["soft_avgs"] = 1  # force soft avgs to 1
        cfg["reps"] = 1  # force reps to 1
        super().__init__(soccfg, cfg)

    def declare_gain_reg(self):
        ch = self.qub_pulse["ch"]
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
        _, avgi, avgq = super().acquire(*args, **kwargs)
        assert isinstance(avgi, list), "make pyright happy"
        avgi = avgi[0][0].reshape(-1, 2).T  # (2, reps)
        avgq = avgq[0][0].reshape(-1, 2).T  # (2, reps)
        return avgi, avgq  # (2, reps)
