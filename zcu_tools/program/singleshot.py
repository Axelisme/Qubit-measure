from qick import RAveragerProgram

from .base import BaseTwoToneProgram


class SingleShotProgram(RAveragerProgram, BaseTwoToneProgram):
    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()

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

        # set intial gain to 0
        self.pi_gain = self.qub_pulse["gain"]
        self.qub_pulse["gain"] = 0
        self.setup_qubit()

        # find the gain register
        qub_ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.r_gain = self.sreg(qub_ch, "gain")

        self.synci(200)

    def body(self):
        BaseTwoToneProgram.body(self)

    def update(self):
        # update the gain to pi pulse
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.pi_gain)

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
