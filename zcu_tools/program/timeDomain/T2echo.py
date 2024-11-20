from ..pulse import create_waveform, set_pulse
from .base import BaseTimeProgram


class T2EchoProgram(BaseTimeProgram):
    def initialize(self):
        super().initialize()

        self.pi_cfg = self.cfg["qub_pulse"]["pi"]
        self.pi2_cfg = self.cfg["qub_pulse"]["pi2"]

        # set the pulse registers for resonator and qubit
        qub_ch = self.cfg["qubit"]["qub_ch"]
        self.pi_wavform = create_waveform(self, qub_ch, self.pi_cfg)
        self.pi2_wavform = create_waveform(self, qub_ch, self.pi2_cfg)

        self.synci(200)

    def body(self):
        cfg = self.cfg
        res_cfg = self.res_cfg
        qub_cfg = self.qub_cfg
        pi_cfg = self.pi_cfg
        pi2_cfg = self.pi2_cfg

        self.flux_ctrl.trigger()

        # pi/2 - pi - pi/2 sequence
        set_pulse(self, qub_cfg["qub_ch"], pi2_cfg, waveform=self.pi2_wavform)
        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, qub_cfg["qub_ch"], pi_cfg, waveform=self.pi_wavform)
        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, qub_cfg["qub_ch"], pi2_cfg, waveform=self.pi2_wavform)
        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )
