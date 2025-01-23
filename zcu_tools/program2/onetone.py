from .base import MyAveragerProgram, MyRAveragerProgram, SYNC_TIME


def onetone_body(prog: MyAveragerProgram, before_readout=None):
    # reset
    prog.resetM.reset_qubit(prog)

    # readout
    prog.readoutM.readout_qubit(prog, before_readout)


class OneToneProgram(MyAveragerProgram):
    def initialize(self):
        self.resetM.init(self)
        self.readoutM.init(self)

        self.synci(SYNC_TIME)

    def body(self):
        onetone_body(self)


class RGainOnetoneProgram(MyRAveragerProgram):
    def declare_gain_reg(self):
        # setup gain register
        ch = self.res_pulse["ch"]
        self.r_rp = self.ch_page(ch)
        self.r_gain = self.sreg(ch, "gain")
        self.r_gain2 = self.sreg(ch, "gain2")
        self.r_gain_t = 3
        self.mathi(self.r_rp, self.r_gain_t, self.r_gain, "+", 0)
        self.r_step = self.cfg["step"]

    def initialize(self):
        self.res_pulse["gain"] = self.cfg["start"]

        self.resetM.init(self)
        self.readoutM.init(self)
        self.declare_gain_reg()

        self.synci(SYNC_TIME)

    def body(self):
        onetone_body(self, self.set_gain_reg)

    def set_gain_reg(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain_t, "+", 0)
        self.mathi(self.r_rp, self.r_gain2, self.r_gain_t, ">>", 1)  # divide by 2

    def update(self):
        self.mathi(self.r_rp, self.r_gain_t, self.r_gain_t, "+", self.r_step)


class RFreqOnetoneProgram(MyRAveragerProgram):
    def declare_freq_reg(self):
        res_ch = self.res_pulse["ch"]
        ro_ch = self.adc["chs"][0]

        self.r_rp = self.ch_page(res_ch)
        self.r_freq = self.sreg(res_ch, "freq")
        self.r_freq_t = 3
        self.mathi(self.r_rp, self.r_freq_t, self.r_freq, "+", 0)
        self.r_step = self.freq2reg_adc(self.cfg["step"], gen_ch=res_ch, ro_ch=ro_ch)

        self.ro_rp = self.ch_page_ro(ro_ch)
        self.ro_freq = self.sreg_ro(ro_ch, "adc_freq")
        self.ro_freq_t = 4
        self.mathi(self.ro_rp, self.ro_freq_t, self.ro_freq, "+", 0)
        self.ro_step = self.freq2reg_adc(self.cfg["step"], gen_ch=res_ch, ro_ch=ro_ch)

    def initialize(self):
        assert len(self.adc["chs"]) == 1, "Only one adc channel is supported"

        self.res_pulse["freq"] = self.sweep_cfg["start"]

        self.resetM.init(self)
        self.readoutM.init(self)
        self.declare_freq_reg()

        self.synci(SYNC_TIME)

    def body(self):
        onetone_body(self, self.set_freq_reg)

    def set_freq_reg(self):
        self.mathi(self.r_rp, self.r_freq, self.r_freq_t, "+", 0)
        self.mathi(self.ro_rp, self.ro_freq, self.ro_freq_t, "+", 0)

    def update(self):
        self.mathi(self.r_rp, self.r_freq_t, self.r_freq_t, "+", self.r_step)
        self.mathi(self.ro_rp, self.ro_freq_t, self.ro_freq_t, "+", self.ro_step)
