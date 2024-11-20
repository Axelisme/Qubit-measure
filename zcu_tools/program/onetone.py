from .base import BaseOneToneProgram


class OnetoneProgram(BaseOneToneProgram):
    def initialize(self):
        super().initialize()

        self.synci(200)

    def body(self):
        cfg = self.cfg
        res_cfg = self.res_cfg

        self.flux_ctrl.trigger()

        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )
