import time
from copy import deepcopy

import yaml

from .tools import deepupdate, numpy2number


def make_cfg(exp_cfg: dict, **kwargs):
    exp_cfg = deepcopy(exp_cfg)
    deepupdate(exp_cfg, kwargs, behavior="force")

    auto_derive(exp_cfg)

    return exp_cfg


def auto_derive_pulse(pulse_cfg: dict, pulses: dict) -> dict:
    def auto_derive_waveform(pulse_cfg: dict):
        # style and length are required
        style = pulse_cfg["style"]
        length = pulse_cfg["length"]
        if style == "flat_top":
            raise_cfg = pulse_cfg.setdefault("raise_pulse", {})
            # default raise style is cosine
            raise_cfg.setdefault("style", "cosine")
            # default raise pulse is 10% of the total length
            raise_cfg.setdefault("length", 0.1 * length)

            # derive raise pulse parameters
            auto_derive_waveform(raise_cfg)
        elif style == "gauss":
            pulse_cfg.setdefault("sigma", length / 4)

    # derive pulse config
    if isinstance(pulse_cfg, str):
        # string like "pulse1"
        pulse_cfg: dict = deepcopy(pulses[pulse_cfg])
        auto_derive_waveform(pulse_cfg)
    elif isinstance(pulse_cfg, list):
        # list of tuples like [("pi", "pulse1"), ("pi2", "pulse2")]
        pulse_cfg = dict(pulse_cfg)
        for name, pulse in pulse_cfg.items():
            if isinstance(pulse, str):
                pulse_cfg[name] = deepcopy(pulses[pulse])
            auto_derive_waveform(pulse_cfg[name])
    elif isinstance(pulse_cfg, dict):
        auto_derive_waveform(pulse_cfg)

    return pulse_cfg


def auto_derive_res(exp_cfg: dict):
    res_cfgs = deepcopy(DefaultCfg.res_cfgs)

    # derive resonator config
    if isinstance(exp_cfg["resonator"], str):
        exp_cfg["resonator"] = res_cfgs[exp_cfg["resonator"]]
    res_cfg: dict = exp_cfg["resonator"]

    # derive resonator pulse config
    pulse_cfgs: dict = res_cfg.get("pulses", {})
    exp_cfg["res_pulse"] = auto_derive_pulse(exp_cfg["res_pulse"], pulse_cfgs)

    # remove pulses from resonator config for clarity
    res_cfg.pop("pulses", None)


def auto_derive_qub(exp_cfg: dict):
    qub_cfgs = deepcopy(DefaultCfg.qub_cfgs)

    # derive qubit config
    if isinstance(exp_cfg["qubit"], str):
        exp_cfg["qubit"] = qub_cfgs[exp_cfg["qubit"]]
    qub_cfg: dict = exp_cfg["qubit"]

    # derive qubit pulse config
    pulse_cfgs: dict = qub_cfg.get("pulses", {})
    exp_cfg["qub_pulse"] = auto_derive_pulse(exp_cfg["qub_pulse"], pulse_cfgs)

    # remove pulses from qubit config for clarity
    qub_cfg.pop("pulses", None)


def auto_derive_flux(exp_cfg: dict):
    # assign None if not provided
    exp_cfg.setdefault("flux", 0.0)

    if not isinstance(exp_cfg["flux"], dict):
        exp_cfg["flux"] = {"value": exp_cfg["flux"]}

    flux_cfgs = deepcopy(DefaultCfg.flux_cfgs)

    # if not method provided, use default method
    if "method" not in exp_cfg["flux"]:
        exp_cfg["flux"]["method"] = flux_cfgs["default_method"]

    # derive flux config
    method = exp_cfg["flux"]["method"]
    if isinstance(method, str):
        # like "yokogawa"
        deepupdate(exp_cfg["flux"], {method: flux_cfgs[method]}, behavior="ignore")
    elif isinstance(method, list):
        # like ["yokogawa", "zcu216"]
        deepupdate(
            exp_cfg["flux"], {m: flux_cfgs[m] for m in method}, behavior="ignore"
        )


def auto_derive_exp(exp_cfg: dict):
    res_cfg = exp_cfg["resonator"]
    res_pulse = exp_cfg["res_pulse"]

    # default experiment parameters

    # relax delay
    exp_cfg.setdefault("relax_delay", 0.0)

    # readout length
    # weird factor 1.3 to make the readout pulse same length as the pulse
    exp_cfg.setdefault("readout_length", res_pulse["length"] / 1.3)

    # adc_trig_offset
    if "adc_trig_offset" not in exp_cfg and "adc_trig_offset" in res_cfg:
        exp_cfg["adc_trig_offset"] = res_cfg["adc_trig_offset"]


def auto_derive(exp_cfg):
    assert DefaultCfg.is_init_global(), "Configuration is not initialized."

    auto_derive_res(exp_cfg)
    if "qubit" in exp_cfg:
        auto_derive_qub(exp_cfg)
    auto_derive_flux(exp_cfg)

    auto_derive_exp(exp_cfg)


class DefaultCfg:
    res_cfgs = None
    qub_cfgs = None
    flux_cfgs = None

    @classmethod
    def init_global(
        cls, res_cfgs: dict, qub_cfgs: dict, flux_cfgs: dict, overwrite=False
    ):
        if not overwrite:
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
    def load(cls, filepath, overwrite=False):
        if not overwrite:
            assert (
                not DefaultCfg.is_init_global()
            ), "Configuration is already initialized."

        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)

        cls.res_cfgs = cfg["res_cfgs"]
        cls.qub_cfgs = cfg["qub_cfgs"]
        cls.flux_cfgs = cfg["flux_cfgs"]

    @classmethod
    def dump(cls, filepath=None):
        if filepath is None:
            filepath = f"cfg_{time.strftime('%Y%m%d_%H%M%S')}.yaml"

        if not filepath.endswith(".yaml"):
            filepath += ".yaml"

        dump_cfg = numpy2number(cls.dict())
        with open(filepath, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def set_res(cls, resonator, behavior="force", **cfg):
        deepupdate(cls.res_cfgs[resonator], cfg, behavior=behavior)

    @classmethod
    def set_qub(cls, qubit, behavior="force", **cfg):
        deepupdate(cls.qub_cfgs[qubit], cfg, behavior=behavior)

    @classmethod
    def set_res_pulse(cls, resonator: str, behavior="force", **pulse_cfgs):
        res_cfg = cls.res_cfgs[resonator]
        res_cfg.setdefault("pulses", {})
        deepupdate(res_cfg["pulses"], pulse_cfgs, behavior=behavior)

    @classmethod
    def get_res_pulse(cls, resonator: str, pulse_name: str):
        res_cfg = cls.res_cfgs[resonator]
        return res_cfg["pulses"][pulse_name]

    @classmethod
    def set_qub_pulse(cls, qubit, behavior="force", **pulse_cfgs):
        qub_cfg = cls.qub_cfgs[qubit]
        qub_cfg.setdefault("pulses", {})
        deepupdate(qub_cfg["pulses"], pulse_cfgs, behavior=behavior)

    @classmethod
    def get_qub_pulse(cls, qubit, pulse_name):
        qub_cfg = cls.qub_cfgs[qubit]
        return qub_cfg["pulses"][pulse_name]

    @classmethod
    def dict(cls):
        return {
            "res_cfgs": cls.res_cfgs,
            "qub_cfgs": cls.qub_cfgs,
            "flux_cfgs": cls.flux_cfgs,
        }
