from qick.asm_v1 import AcquireProgram

from .pulse import create_waveform, set_pulse


class BaseOneToneProgram(AcquireProgram):
    def parse_cfg(self):
        assert isinstance(self.cfg, dict), "cfg is not a dict"

        self.dac_cfg: dict = self.cfg["dac"]
        self.adc_cfg: dict = self.cfg["adc"]

        self.res_pulse = self.dac_cfg.get("res_pulse")

    def setup_readout(self):
        # declare the resonator generator
        res_ch = self.res_pulse["ch"]
        self.declare_gen(ch=res_ch, nqz=self.res_pulse["nqz"])

        # declare the readout channels
        adc_cfg = self.adc_cfg
        ro_chs = adc_cfg["chs"]
        for ro_ch in ro_chs:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(adc_cfg["ro_length"], ro_ch=ro_ch),
                freq=self.res_pulse["freq"],
                gen_ch=res_ch,
            )

        # create the resonator pulse
        create_waveform(self, "res_pulse", self.res_pulse)
        set_pulse(self, self.res_pulse, ro_chs[0], "res_pulse")

    def initialize(self):
        self.parse_cfg()
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
        self.measure_pulse()


class BaseTwoToneProgram(BaseOneToneProgram):
    def parse_cfg(self):
        BaseOneToneProgram.parse_cfg(self)

        self.qub_pulse = self.dac_cfg.get("qub_pulse")

    def setup_qubit(self):
        qub_pulse = self.qub_pulse
        self.declare_gen(qub_pulse["ch"], nqz=qub_pulse["nqz"])
        create_waveform(self, "qub_pulse", qub_pulse)
        set_pulse(self, qub_pulse, waveform="qub_pulse")

    def initialize(self):
        self.parse_cfg()
        self.setup_readout()
        self.setup_qubit()

        self.synci(200)

    def body(self):
        # qubit pulse
        self.pulse(ch=self.qub_pulse["ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()
