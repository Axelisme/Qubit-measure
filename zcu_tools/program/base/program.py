from qick.asm_v1 import AcquireProgram

from .flux import make_fluxControl
from .pulse import create_waveform, is_single_pulse, set_pulse


class BaseOneToneProgram(AcquireProgram):
    def parse_cfg(self):
        assert hasattr(self, "cfg"), "cfg is not defined"
        assert isinstance(self.cfg, dict), "cfg is not a dict"

        self.flux_dev = self.cfg["flux_dev"]
        self.res_cfg = self.cfg["resonator"]
        self.res_pulse = self.cfg["res_pulse"]
        self.adc_cfg = self.cfg["adc"]

    def setup_readout(self):
        assert is_single_pulse(self.res_pulse), "Currently only support one pulse cfg"

        # declare the resonator channel and readout channels
        res_ch = self.res_pulse["ch"]
        ro_chs = self.adc_cfg["chs"]
        self.declare_gen(ch=res_ch, nqz=self.res_pulse["nqz"])
        for ro_ch in ro_chs:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(self.adc_cfg["ro_length"], ro_ch=ro_ch),
                freq=self.res_pulse["freq"],
                gen_ch=res_ch,
            )
        # create the resonator pulse
        self.res_wavform = create_waveform(self, self.res_pulse)
        set_pulse(self, self.res_pulse, ro_ch=ro_chs[0], waveform=self.res_wavform)

    def setup_flux(self):
        self.flux_ctrl = make_fluxControl(self, self.flux_dev)
        self.flux_ctrl.set_flux(self.cfg.get("flux"))

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()

        self.synci(200)

    def measure_pulse(self):
        cfg = self.cfg
        res_pulse = self.res_pulse
        adc_cfg = self.adc_cfg

        self.measure(
            pulse_ch=res_pulse["ch"],
            adcs=adc_cfg["chs"],
            adc_trig_offset=self.us2cycles(adc_cfg["trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )

    def body(self):
        self.flux_ctrl.trigger()

        self.measure_pulse()


class BaseTwoToneProgram(BaseOneToneProgram):
    def parse_cfg(self):
        super().parse_cfg()
        self.qub_cfg = self.cfg["qubit"]
        self.qub_pulse = self.cfg["qub_pulse"]

    def setup_qubit(self):
        qub_ch = self.qub_pulse["ch"]
        self.declare_gen(ch=qub_ch, nqz=self.qub_pulse["nqz"])
        self.qub_wavform = create_waveform(self, self.qub_pulse)
        if isinstance(self.qub_wavform, str):
            set_pulse(self, self.qub_pulse, waveform=self.qub_wavform)

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()

        self.synci(200)

    def body(self):
        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=self.qub_pulse["ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()
