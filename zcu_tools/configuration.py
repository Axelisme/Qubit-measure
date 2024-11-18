import time
from copy import deepcopy

import yaml

from .tools import deepupdate


def make_cfg(exp_cfg: dict, **kwargs):
    exp_cfg = deepcopy(exp_cfg)
    deepupdate(exp_cfg, kwargs, overwrite=True)

    replace_by_default(exp_cfg)

    return exp_cfg


def replace_by_default(exp_cfg):
    assert DefaultCfg.is_init_global(), "Configuration is not initialized."

    # replace resonator config
    if "resonator" in exp_cfg:
        if isinstance(exp_cfg["resonator"], str):
            # convert string to dict
            exp_cfg["resonator"] = DefaultCfg.res_cfgs[exp_cfg["resonator"]]

        res_cfg = exp_cfg["resonator"]

        if isinstance(exp_cfg["res_pulse"], str):
            # string like "pulse1"
            exp_cfg["res_pulse"] = res_cfg["pulses"][exp_cfg["res_pulse"]]
        elif isinstance(exp_cfg["res_pulse"], list):
            # list of tuples like [("pi", "pulse1"), ("pi2", "pulse2")]
            pulses = {}
            for name, pulse in exp_cfg["res_pulse"]:
                if isinstance(pulse, str):
                    pulse = res_cfg["pulses"][pulse]
                pulses[name] = pulse
            exp_cfg["res_pulse"] = pulses

    # replace qubit config
    if "qubit" in exp_cfg:
        if isinstance(exp_cfg["qubit"], str):
            # convert string to dict
            exp_cfg["qubit"] = DefaultCfg.qub_cfgs[exp_cfg["qubit"]]

        qub_cfg = exp_cfg["qubit"]

        if isinstance(exp_cfg["qub_pulse"], str):
            # string like "pulse1"
            exp_cfg["qub_pulse"] = qub_cfg["pulses"][exp_cfg["qub_pulse"]]
        elif isinstance(exp_cfg["qub_pulse"], list):
            # list of tuples like [("pi", "pulse1"), ("pi2", "pulse2")]
            pulses = {}
            for name, pulse in exp_cfg["qub_pulse"]:
                if isinstance(pulse, str):
                    pulse = qub_cfg["pulses"][pulse]
                pulses[name] = pulse
            exp_cfg["qub_pulse"] = pulses

    # replace flux config
    if "flux" in exp_cfg:
        # add spefic flux config to the experiment
        deepupdate(exp_cfg["flux"], DefaultCfg.flux_cfgs[exp_cfg["flux"]["method"]])

    res_cfg = exp_cfg["resonator"]

    # default experiment parameters
    exp_cfg.setdefault("relax_delay", 0.0)  # relax delay
    if "readout_length" not in exp_cfg:  # readout length
        res_pulse = exp_cfg["res_pulse"]
        if "length" in res_pulse:
            exp_cfg["readout_length"] = res_pulse["length"]
        else:
            raise ValueError("Cannot determine readout length.")
    if "adc_trig_offset" not in exp_cfg:  # adc_trig_offset
        if "adc_trig_offset" in res_cfg:
            exp_cfg["adc_trig_offset"] = res_cfg["adc_trig_offset"]
        else:
            raise ValueError("Cannot determine adc_trig_offset.")


class DefaultCfg:
    res_cfgs = None
    qub_cfgs = None
    flux_cfgs = None

    @classmethod
    def init_global(cls, res_cfgs: dict, qub_cfgs: dict, flux_cfgs: dict):
        assert not cls.is_init_global(), "Configuration is already initialized."
        assert isinstance(res_cfgs, dict), f"res_cfgs should be dict, got {res_cfgs}"
        assert isinstance(qub_cfgs, dict), f"qub_cfgs should be dict, got {qub_cfgs}"
        assert isinstance(flux_cfgs, dict), f"flux_cfgs should be dict, got {flux_cfgs}"

        cls.res_cfgs = res_cfgs
        cls.qub_cfgs = qub_cfgs
        cls.flux_cfgs = flux_cfgs

    @classmethod
    def is_init_global(cls):
        return cls.res_cfgs is not None

    @classmethod
    def load(cls, filepath):
        assert not DefaultCfg.is_init_global(), "Configuration is already initialized."

        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)

        cls.res_cfgs = cfg["res_cfgs"]
        cls.qub_cfgs = cfg["qub_cfgs"]
        cls.flux_cfgs = cfg["flux_cfgs"]

    @classmethod
    def dump(cls, filepath=None):
        if filepath is None:
            filepath = f"cfg_{time.strftime('%Y%m%d_%H%M%S')}.yaml"

        # type cast all numpy types to python types
        def numpy2number(obj):
            if hasattr(obj, "tolist"):
                obj = obj.tolist()
            if isinstance(obj, dict):
                return {k: numpy2number(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [numpy2number(v) for v in obj]
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        dump_cfg = numpy2number(cls.dict())
        with open(filepath, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def set_res(cls, resonator, overwrite=False, **cfg):
        deepupdate(cls.res_cfgs[resonator], cfg, overwrite=overwrite)

    @classmethod
    def set_qub(cls, qubit, overwrite=False, **cfg):
        deepupdate(cls.qub_cfgs[qubit], cfg, overwrite=overwrite)

    @classmethod
    def set_res_pulse(cls, resonator: str, overwrite=False, **pulse_cfgs):
        res_cfg = cls.res_cfgs[resonator]
        res_cfg.setdefault("pulses", {})
        deepupdate(res_cfg["pulses"], pulse_cfgs, overwrite=overwrite)

    @classmethod
    def set_qub_pulse(cls, qubit, overwrite=False, **pulse_cfgs):
        qub_cfg = cls.qub_cfgs[qubit]
        qub_cfg.setdefault("pulses", {})
        deepupdate(qub_cfg["pulses"], pulse_cfgs, overwrite=overwrite)

    @classmethod
    def dict(cls):
        return {
            "res_cfgs": cls.res_cfgs,
            "qub_cfgs": cls.qub_cfgs,
            "flux_cfgs": cls.flux_cfgs,
        }
