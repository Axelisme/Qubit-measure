from copy import deepcopy

import qick as qk  # type: ignore

from .flux import make_fluxControl
from .util import create_pulse


class AmplitudeRabiProgram(qk.RAveragerProgram):
    def initialize(self):
        self.cfg = deepcopy(self.cfg)  # prevent in-place modification
        cfg = self.cfg
        res_cfg = cfg["resonator"]
        qub_cfg = cfg["qubit"]
        res_pulse_cfg = cfg["res_pulse"]
        qub_pulse_cfg = cfg["qub_pulse"]

        self.res_cfg = res_cfg
        self.qub_cfg = qub_cfg

        sweep_cfg = cfg["sweep"]
        cfg["start"] = sweep_cfg["start"]
        cfg["step"] = sweep_cfg["step"]
        cfg["expts"] = sweep_cfg["expts"]

        # overwrite qubit pulse gain
        qub_pulse_cfg["gain"] = cfg["start"]

        res_ch = res_cfg["res_ch"]
        qub_ch = qub_cfg["qub_ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.r_gain = self.sreg(qub_ch, "gain")

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
        create_pulse(self, qub_ch, qub_pulse_cfg)

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

    def update(self):
        # update gain of the Gaussian pi pulse
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.cfg["step"])
