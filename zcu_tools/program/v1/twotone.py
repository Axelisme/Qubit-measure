from qick.averager_program import QickSweep, merge_sweeps

from .base import (
    SYNC_TIME,
    MyAveragerProgram,
    MyNDAveragerProgram,
    MyRAveragerProgram,
    declare_pulse,
    set_pulse,
)

PULSE_DELAY = 0.01  # us


def twotone_body(prog, before_pulse=None):
    # reset
    prog.resetM.reset_qubit(prog)

    # qubit pulse
    ch = prog.qub_pulse["ch"]
    if prog.ch_count[ch] > 1:
        set_pulse(prog, prog.qub_pulse, waveform="qub_pulse")
    if before_pulse is not None:
        before_pulse()
    prog.pulse(ch=ch)
    prog.sync_all(prog.us2cycles(PULSE_DELAY))

    # measure
    prog.readoutM.readout_qubit(prog)


class TwoToneProgram(MyAveragerProgram):
    def initialize(self):
        super().initialize()
        declare_pulse(self, self.qub_pulse, "qub_pulse")

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self)


class RGainTwoToneProgram(MyRAveragerProgram):
    def declare_gain_reg(self):
        ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(ch)
        self.q_gain = self.sreg(ch, "gain")
        self.q_gain2 = self.sreg(ch, "gain2")
        self.q_gain_t = 3
        self.mathi(self.q_rp, self.q_gain_t, self.q_gain, "+", 0)
        self.q_step = self.cfg["step"]

    def initialize(self):
        self.qub_pulse["gain"] = self.cfg["start"]

        super().initialize()
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_gain_reg()

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self, self.set_gain_reg)

    def set_gain_reg(self):
        self.mathi(self.q_rp, self.q_gain, self.q_gain_t, "+", 0)
        self.mathi(self.q_rp, self.q_gain2, self.q_gain_t, ">>", 1)  # divide by 2

    def update(self):
        self.mathi(self.q_rp, self.q_gain_t, self.q_gain_t, "+", self.q_step)


class RFreqTwoToneProgram(MyRAveragerProgram):
    def declare_freq_reg(self):
        ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(ch)
        self.q_freq = self.sreg(ch, "freq")
        self.q_freq_t = 3
        self.mathi(self.q_rp, self.q_freq_t, self.q_freq, "+", 0)
        self.q_step = self.freq2reg(self.cfg["step"], gen_ch=ch)

    def initialize(self):
        self.qub_pulse["freq"] = self.cfg["start"]

        super().initialize()
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_freq_reg()

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self, self.set_freq_reg)

    def set_freq_reg(self):
        self.mathi(self.q_rp, self.q_freq, self.q_freq_t, "+", 0)

    def update(self):
        self.mathi(self.q_rp, self.q_freq_t, self.q_freq_t, "+", self.q_step)


class RFreqTwoToneProgramWithRedReset(MyRAveragerProgram):
    def declare_freq_reg(self):
        qub_ch = self.qub_pulse["ch"]
        reset_ch = self.reset_pulse["ch"]

        self.q_rp = self.ch_page(qub_ch)
        self.q_freq = self.sreg(qub_ch, "freq")
        self.q_freq_t = 3
        self.mathi(self.q_rp, self.q_freq_t, self.q_freq, "+", 0)
        self.q_step = self.freq2reg(self.cfg["step"], gen_ch=qub_ch)

        self.s_rp = self.ch_page(reset_ch)
        self.s_freq = self.sreg(reset_ch, "freq")
        self.s_freq_t = 4
        self.mathi(self.s_rp, self.s_freq_t, self.s_freq, "+", 0)
        self.s_step = self.freq2reg(self.cfg["step"], gen_ch=reset_ch)

    def initialize(self):
        self.qub_pulse["freq"] = self.cfg["start"]
        self.reset_pulse["freq"] = self.cfg["r_f"] - self.cfg["start"]

        super().initialize()
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_freq_reg()

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self, self.set_freq_reg)

    def set_freq_reg(self):
        self.mathi(self.q_rp, self.q_freq, self.q_freq_t, "+", 0)
        self.mathi(self.s_rp, self.s_freq, self.s_freq_t, "+", 0)

    def update(self):
        self.mathi(self.q_rp, self.q_freq_t, self.q_freq_t, "+", self.q_step)
        self.mathi(self.s_rp, self.s_freq_t, self.s_freq_t, "-", self.s_step)


class PowerDepProgram(MyNDAveragerProgram):
    def add_freq_sweep(self):
        sweep_cfg = self.sweep_cfg["freq"]
        r_freq = self.get_gen_reg(self.qub_pulse["ch"], "freq")
        self.add_sweep(
            QickSweep(
                self, r_freq, sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"]
            )
        )

    def add_gain_sweep(self):
        sweep_cfg = self.sweep_cfg["gain"]
        r_gain = self.get_gen_reg(self.qub_pulse["ch"], "gain")
        r_gain2 = self.get_gen_reg(self.qub_pulse["ch"], "gain2")
        self.add_sweep(
            merge_sweeps(
                [
                    QickSweep(
                        self,
                        r_gain,
                        sweep_cfg["start"],
                        sweep_cfg["stop"],
                        sweep_cfg["expts"],
                    ),
                    QickSweep(
                        self,
                        r_gain2,
                        sweep_cfg["start"] // 2,
                        sweep_cfg["stop"] // 2,
                        sweep_cfg["expts"],
                    ),
                ]
            )
        )

    def initialize(self):
        self.qub_pulse["freq"] = self.sweep_cfg["freq"]["start"]
        self.qub_pulse["gain"] = self.sweep_cfg["gain"]["start"]

        super().initialize()
        if self.ch_count[self.qub_pulse["ch"]] > 1:
            raise ValueError(
                "Only one pulse per channel is supported in PowerDepProgram"
            )
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.add_freq_sweep()
        self.add_gain_sweep()

        self.synci(SYNC_TIME)

    def body(self):
        twotone_body(self)
