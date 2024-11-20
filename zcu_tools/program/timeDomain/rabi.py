from .base import BaseTimeProgram, create_pulse


class AmpRabiProgram(BaseTimeProgram):
    def initialize(self):
        super().initialize()

        qub_pulse_cfg = self.cfg["qub_pulse"]

        # overwrite qubit pulse gain
        qub_pulse_cfg["gain"] = self.cfg["start"]

        create_pulse(self, self.qub_cfg["qub_ch"], qub_pulse_cfg)

        self.synci(200)

    def body(self):
        cfg = self.cfg
        res_cfg = self.res_cfg
        qub_cfg = self.qub_cfg

        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.02))

        # measure
        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )
