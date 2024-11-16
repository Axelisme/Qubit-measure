import qick as qk  # type: ignore
from zcu_tools.configuration import parse_res_pulse

from .flux import make_fluxControl
from .util import create_pulse


class OnetoneProgram(qk.AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        glb_cfg: dict = cfg["global"]
        res_cfg = glb_cfg["res_cfgs"][cfg["resonator"]]
        res_pulse_cfg = parse_res_pulse(cfg)

        self.gbl_cfg = glb_cfg
        self.res_cfg = res_cfg

        # declare the resonator channel and readout channels
        res_ch = res_cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=res_cfg["nqz"])

        # prepare the flux control
        flux_cfgs = glb_cfg["flux_cfgs"]
        self.flux_ctrl = make_fluxControl(self, cfg["flux"]["method"], flux_cfgs)
        self.flux_ctrl.set_flux(flux=cfg["flux"]["value"])

        for ro_ch in res_cfg["ro_chs"]:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(cfg["readout_length"]),
                freq=res_pulse_cfg["freq"],
                gen_ch=res_ch,
            )

        # set the pulse registers for resonator
        create_pulse(
            self,
            ch=res_ch,
            pulse_cfg=res_pulse_cfg,
            for_readout=True,
        )

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
