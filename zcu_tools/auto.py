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
            # the minimum length is 15 ns
            raise_cfg.setdefault("length", 0.1 * max(length, 0.15))

            # derive raise pulse parameters
            auto_derive_waveform(raise_cfg)
        else:
            pulse_cfg.setdefault("phase", 0)

            if style == "gauss":
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

    # replace resonator name with resonator config
    if isinstance(exp_cfg["resonator"], str):
        exp_cfg["resonator"] = res_cfgs[exp_cfg["resonator"]]

    res_cfg: dict = exp_cfg["resonator"]

    # replace pulses with pulse config
    pulse_cfgs: dict = res_cfg.get("pulses", {})
    exp_cfg["res_pulse"] = auto_derive_pulse(exp_cfg["res_pulse"], pulse_cfgs)

    # remove pulses from resonator config for clarity
    res_cfg.pop("pulses", None)


def auto_derive_qub(exp_cfg: dict):
    qub_cfgs = deepcopy(DefaultCfg.qub_cfgs)

    # if not provided
    if "qubit" not in exp_cfg:
        return

    # replace qubit name with qubit config
    if isinstance(exp_cfg["qubit"], str):
        qub_name = exp_cfg["qubit"]
        exp_cfg["qubit"] = qub_cfgs[qub_name]
        exp_cfg["qubit"]["name"] = qub_name
    qub_cfg: dict = exp_cfg["qubit"]

    # replace pulses with pulse config
    pulse_cfgs: dict = qub_cfg.get("pulses", {})
    # for single qubit experiment
    if "qub_pulse" in exp_cfg:
        exp_cfg["qub_pulse"] = auto_derive_pulse(exp_cfg["qub_pulse"], pulse_cfgs)
    # for ef experiment
    if "ef_pulse" in exp_cfg:
        exp_cfg["ef_pulse"] = auto_derive_pulse(exp_cfg["ef_pulse"], pulse_cfgs)
    if "ge_pulse" in exp_cfg:
        exp_cfg["ge_pulse"] = auto_derive_pulse(exp_cfg["ge_pulse"], pulse_cfgs)

    # remove pulses from qubit config for clarity
    qub_cfg.pop("pulses", None)


def auto_derive_flux(exp_cfg: dict):
    flux_cfgs: dict = deepcopy(DefaultCfg.flux_cfgs)

    # if not provided
    if "flux_dev" not in exp_cfg:
        exp_cfg["flux_dev"] = "none"

    # replace flux_dev with flux config
    if isinstance(exp_cfg["flux_dev"], str):
        method = exp_cfg["flux_dev"]
        exp_cfg["flux_dev"] = flux_cfgs.get(method, {})
        exp_cfg["flux_dev"]["name"] = method
    method = exp_cfg["flux_dev"]["name"]

    if method == "none" or "flux" not in exp_cfg:
        return

    if exp_cfg["flux"] == "sw_spot":
        assert "qubit" in exp_cfg, "No qubit provided for sw_spot."
        sw_spot = DefaultCfg.get_sw_spot(exp_cfg["qubit"])
        assert method in sw_spot, f"No sw_spot for {method}"
        exp_cfg["flux"] = sw_spot[method]


def auto_derive_exp(exp_cfg: dict):
    res_cfg = exp_cfg["resonator"]
    res_pulse = exp_cfg["res_pulse"]

    # default experiment parameters
    # 0.0 by default
    exp_cfg.setdefault("relax_delay", 0.0)

    # readout length
    if "readout_length" not in exp_cfg:
        # factor 1.3 to make the readout length same length as the readout pulse
        assert "length" in res_pulse, "Cannot auto derive readout_length."
        exp_cfg["readout_length"] = res_pulse["length"] / 1.3

    # adc_trig_offset
    if "adc_trig_offset" not in exp_cfg:
        # use the adc_trig_offset from resonator config
        assert "adc_trig_offset" in res_cfg, "Cannot auto derive adc_trig_offset."
        exp_cfg["adc_trig_offset"] = res_cfg["adc_trig_offset"]


def fill_default_str(exp_cfg: dict):
    for key, value in DefaultCfg.exp_default.items():
        exp_cfg.setdefault(key, value)


def auto_derive(exp_cfg):
    assert DefaultCfg.is_init_global(), "Configuration is not initialized."

    # add some user specified parameters
    fill_default_str(exp_cfg)

    # derive other parameters
    auto_derive_res(exp_cfg)
    auto_derive_qub(exp_cfg)
    auto_derive_flux(exp_cfg)

    auto_derive_exp(exp_cfg)