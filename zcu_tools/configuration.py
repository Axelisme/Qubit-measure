from .tools import deepupdate, numpy2number
from typing import Optional, Literal


class DefaultCfg:
    dac_cfgs = {}
    adc_cfgs = {}
    dev_cfgs = {}

    @classmethod
    def load(cls, filepath: str):
        import yaml

        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)

        cls.dac_cfgs: dict = cfg.get("dac_cfgs", {})
        cls.adc_cfgs: dict = cfg.get("adc_cfgs", {})
        cls.dev_cfgs: dict = cfg.get("dev_cfgs", {})

        assert isinstance(cls.dac_cfgs, dict), "dac_cfgs should be a dict"
        assert isinstance(cls.adc_cfgs, dict), "adc_cfgs should be a dict"
        assert isinstance(cls.dev_cfgs, dict), "dev_cfgs should be a dict"

    @classmethod
    def dump(cls, filepath: Optional[str] = None):
        import yaml

        if filepath is None:
            import time

            filepath = f"cfg_{time.strftime('%Y%m%d_%H%M%S')}.yaml"

        if not filepath.endswith(".yaml"):
            filepath += ".yaml"

        dump_cfg = numpy2number(cls.dict())
        with open(filepath, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def set_dac(
        cls, behavior: Literal["error", "force", "ignore"] = "force", **dac_cfgs
    ):
        if "pulses" in dac_cfgs:
            print(
                "Warning: Use set_dac to set pulse is not recommended, use set_pulse instead."
            )
        deepupdate(cls.dac_cfgs, dac_cfgs, behavior=behavior)

    @classmethod
    def get_dac(cls, name: str):
        return cls.dac_cfgs.get(name)

    @classmethod
    def set_adc(
        cls, behavior: Literal["error", "force", "ignore"] = "force", **adc_cfgs
    ):
        deepupdate(cls.adc_cfgs, adc_cfgs, behavior=behavior)

    @classmethod
    def get_adc(cls, name: str):
        return cls.adc_cfgs.get(name)

    @classmethod
    def set_pulse(cls, **pulse_cfgs):
        dac_pulses = cls.dac_cfgs.setdefault("pulses", {})
        # directly overwrite the pulse
        for k, v in pulse_cfgs.items():
            dac_pulses[k] = v

    @classmethod
    def get_pulse(cls, name: str) -> dict:
        return cls.dac_cfgs.get("pulses", {}).get(name, {})

    @classmethod
    def clear_pulses(cls):
        cls.dac_cfgs.pop("pulses", None)

    @classmethod
    def set_dev(
        cls, behavior: Literal["error", "force", "ignore"] = "force", **dev_cfgs
    ):
        deepupdate(cls.dev_cfgs, dev_cfgs, behavior=behavior)

    @classmethod
    def get_dev(cls, name: str):
        return cls.dev_cfgs.get(name)

    @classmethod
    def dict(cls):
        return {
            "dac_cfgs": cls.dac_cfgs,
            "adc_cfgs": cls.adc_cfgs,
            "dev_cfgs": cls.dev_cfgs,
        }
