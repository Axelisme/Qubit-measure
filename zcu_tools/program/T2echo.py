import qick as qk  # type: ignore

from .flux import make_fluxControl
from .util import create_pulse, create_waveform, set_pulse


class T2EchoProgram(qk.RAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_cfg = cfg["resonator"]
        qub_cfg = cfg["qubit"]
        res_pulse_cfg = cfg["res_pulse"]
        qub_pulse_cfg = cfg["qub_pulse"]
        pi_cfg = qub_pulse_cfg["pi"]
        pi2_cfg = qub_pulse_cfg["pi2"]

        self.res_cfg = res_cfg
        self.qub_cfg = qub_cfg
        self.pi_cfg = pi_cfg
        self.pi2_cfg = pi2_cfg

        sweep_cfg = cfg["sweep"]
        cfg["start"] = self.us2cycles(sweep_cfg["start"])
        cfg["step"] = self.us2cycles(sweep_cfg["step"])
        cfg["expts"] = sweep_cfg["expts"]

        res_ch = res_cfg["res_ch"]
        qub_ch = qub_cfg["qub_ch"]

        # set the initial parameters
        self.q_rp = self.ch_page(qub_ch)
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, cfg["start"])

        # declare the resonator channel and readout channels
        self.declare_gen(ch=res_ch, nqz=res_cfg["nqz"])
        self.declare_gen(ch=qub_ch, nqz=qub_cfg["nqz"])
        for ro_ch in res_cfg["ro_chs"]:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(cfg["readout_length"]),
                freq=res_pulse_cfg["freq"],
                gen_ch=res_ch,
            )

        # prepare the flux control
        flux_cfg = cfg["flux"]
        self.flux_ctrl = make_fluxControl(self, flux_cfg["method"], flux_cfg)
        self.flux_ctrl.set_flux(flux=flux_cfg["value"])

        # set the pulse registers for resonator and qubit
        create_pulse(self, res_ch, res_pulse_cfg, for_readout=True)
        self.pi_wavform = create_waveform(self, qub_ch, pi_cfg)
        self.pi2_wavform = create_waveform(self, qub_ch, pi2_cfg)

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

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, "+", self.cfg["step"])
