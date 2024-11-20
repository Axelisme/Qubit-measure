from .base import BaseTimeProgram, create_pulse


class T1Program(BaseTimeProgram):
    def initialize(self):
        super().initialize()

        create_pulse(self, self.cfg["qubit"]["qub_ch"], self.cfg["qub_pulse"])

        self.synci(200)

    def body(self):
        res_cfg = self.res_cfg
        qub_cfg = self.qub_cfg

        self.flux_ctrl.trigger()

        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))
        self.sync(self.q_rp, self.r_wait)

        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(self.cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(self.cfg["relax_delay"]),
        )
