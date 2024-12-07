import yaml

from .tools import deepupdate, numpy2number


class DefaultCfg:
    dac_cfgs = {}
    adc_cfgs = {}
    exp_default = {}

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)

        cls.dac_cfgs: dict = cfg.get("dac_cfgs", {})
        cls.adc_cfgs: dict = cfg.get("adc_cfgs", {})
        cls.exp_default: dict = cfg.get("exp_default", {})

        assert isinstance(cls.dac_cfgs, dict), "dac_cfgs should be a dict"
        assert isinstance(cls.adc_cfgs, dict), "adc_cfgs should be a dict"
        assert isinstance(cls.exp_default, dict), "exp_default should be a dict"

    @classmethod
    def dump(cls, filepath=None):
        if filepath is None:
            import time

            filepath = f"cfg_{time.strftime('%Y%m%d_%H%M%S')}.yaml"

        if not filepath.endswith(".yaml"):
            filepath += ".yaml"

        dump_cfg = numpy2number(cls.dict())
        with open(filepath, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def set_dac(cls, behavior="force", **dac_cfgs):
        if "pulses" in dac_cfgs:
            print(
                "Warning: Use set_dac to set pulse is not recommended, use set_pulse instead."
            )
        deepupdate(cls.dac_cfgs, dac_cfgs, behavior=behavior)

    @classmethod
    def get_dac(cls, name: str):
        return cls.dac_cfgs.get(name)

    @classmethod
    def set_adc(cls, behavior="force", **adc_cfgs):
        deepupdate(cls.adc_cfgs, adc_cfgs, behavior=behavior)

    @classmethod
    def get_adc(cls, name: str):
        return cls.adc_cfgs.get(name)

    @classmethod
    def set_pulse(cls, behavior="force", **pulse_cfgs):
        dac_pulses = cls.dac_cfgs.setdefault("pulses", {})
        deepupdate(dac_pulses, pulse_cfgs, behavior=behavior)

    @classmethod
    def get_pulse(cls, name: str) -> dict:
        return cls.dac_cfgs.get("pulses", {}).get(name, {})

    @classmethod
    def set_default(cls, **kwargs):
        deepupdate(cls.exp_default, kwargs, behavior="force")

    @classmethod
    def dict(cls):
        return {
            "dac_cfgs": cls.dac_cfgs,
            "adc_cfgs": cls.adc_cfgs,
            "exp_default": cls.exp_default,
        }
