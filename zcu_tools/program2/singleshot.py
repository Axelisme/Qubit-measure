from .base import MyRAveragerProgram, SYNC_TIME, declare_pulse
from .twotone import twotone_body


class SingleShotProgram(MyRAveragerProgram):
    def parse_cfg(self):
        cfg = self.cfg

        # overwrite the number of experiments and repetitions
        if "start" in cfg:
            print(f"Warning: Find start is {cfg['start']}")
        if "expts" in cfg:
            print(f"Warning: Find expts is {cfg['expts']}")
        if "reps" in cfg:
            print(f"Warning: Find reps is {cfg['reps']}")
        cfg["start"] = 0
        cfg["step"] = self.qub_pulse["gain"]
        cfg["expts"] = 2
        cfg["reps"] = cfg["shots"]

    def declare_gain_reg(self):
        ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(ch)
        self.r_gain = self.sreg(ch, "gain")
        self.r_gain2 = self.sreg(ch, "gain2")
        self.r_gain_t = 3
        self.regwi(self.q_rp, self.r_gain_t, 0)

    def initialize(self):
        self.parse_cfg()
        self.resetM.init(self)
        self.readoutM.init(self)
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_gain_reg()

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self, self.set_gain_reg)

    def set_gain_reg(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain_t, "+", 0)
        self.mathi(self.q_rp, self.r_gain2, self.r_gain_t, ">>", 1)  # divide by 2

    def update(self):
        self.mathi(self.q_rp, self.r_gain_t, self.r_gain_t, "+", self.cfg["step"])

    def acquire_orig(self, *args, **kwargs):
        return super().acquire(*args, **kwargs)

    def acquire(self, *args, **kwargs):
        super().acquire(*args, **kwargs)
        return self.collect_shots()

    def collect_shots(self):
        cfg = self.cfg
        adc_cfg = self.adc_cfg
        ro_length = self.us2cycles(adc_cfg["ro_length"], ro_ch=adc_cfg["chs"][0])
        expts, reps = cfg["expts"], cfg["reps"]
        i0 = self.di_buf[0].reshape((expts, reps)) / ro_length
        q0 = self.dq_buf[0].reshape((expts, reps)) / ro_length
        return i0, q0
