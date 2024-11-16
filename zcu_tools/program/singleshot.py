from copy import deepcopy

import qick as qk  # type: ignore
from zcu_tools.configuration import parse_res_pulse, parse_qub_pulse

from .flux import make_fluxControl
from .util import create_pulse


class SingleShotProgram(qk.RAveragerProgram):
    def initialize(self):
        self.cfg = deepcopy(self.cfg)  # prevent in-place modification
        cfg = self.cfg
        glb_cfg: dict = cfg["global"]
        res_cfg = glb_cfg["res_cfgs"][cfg["resonator"]]
        qub_cfg = glb_cfg["qub_cfgs"][cfg["qubit"]]
        res_pulse_cfg = parse_res_pulse(cfg)
        qub_pulse_cfg = parse_qub_pulse(cfg)

        self.glb_cfg = glb_cfg
        self.res_cfg = res_cfg
        self.qub_cfg = qub_cfg
        self.res_pulse_cfg = res_pulse_cfg
        self.qub_pulse_cfg = qub_pulse_cfg

        res_ch = res_cfg["res_ch"]
        qub_ch = qub_cfg["qub_ch"]

        # overwrite the number of experiments and repetitions
        if "expts" in cfg or "reps" in cfg:
            if "expts" in cfg:
                print(f"Find expts is {cfg['expts']}")
            if "reps" in cfg:
                print(f"Find reps is {cfg['reps']}")
            print("Warning: expts and reps are overwritten by 2 and shots")
        cfg["start"] = 0
        cfg["step"] = qub_pulse_cfg["gain"]
        cfg["expts"] = 2
        cfg["reps"] = cfg["shots"]

        # set intial gain to 0
        self.pi_gain = cfg["step"]
        qub_pulse_cfg["gain"] = 0

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
        flux_cfgs = glb_cfg["flux_cfgs"]
        self.flux_ctrl = make_fluxControl(self, cfg["flux"]["method"], flux_cfgs)
        self.flux_ctrl.set_flux(flux=cfg["flux"]["value"])

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
        self.sync_all(self.us2cycles(0.05))

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
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.pi_gain)

    def acquire(self, soc, progress=False):
        super().acquire(soc, progress=progress)
        return self.collect_shots()

    def collect_shots(self):
        cfg = self.cfg
        expts, reps = cfg["expts"], cfg["reps"]
        readout_length = cfg["readout_length"]
        i0 = self.di_buf[0].reshape((expts, reps)) / readout_length
        q0 = self.dq_buf[0].reshape((expts, reps)) / readout_length
        return i0, q0
