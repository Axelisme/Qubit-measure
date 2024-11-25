from copy import deepcopy

from .configuration import DefaultCfg
from .tools import deepupdate


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
    if "qub_pulse" in exp_cfg:
        exp_cfg["qub_pulse"] = auto_derive_pulse(exp_cfg["qub_pulse"], pulse_cfgs)
    if "ge_pulse" in exp_cfg:
        exp_cfg["ge_pulse"] = auto_derive_pulse(exp_cfg["ge_pulse"], pulse_cfgs)
    if "ef_pulse" in exp_cfg:
        exp_cfg["ef_pulse"] = auto_derive_pulse(exp_cfg["ef_pulse"], pulse_cfgs)

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
