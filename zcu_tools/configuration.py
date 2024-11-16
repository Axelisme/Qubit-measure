import time
from collections.abc import Mapping
from copy import deepcopy

import yaml

from .tools import deepupdate


def make_cfg(exp_cfg: dict, **kwargs):
    assert DefaultCfg.is_init_global(), "Configuration is not initialized."

    exp_cfg = deepcopy(exp_cfg)
    deepupdate(exp_cfg, kwargs)

    cfg = {"global": DefaultCfg.dict(), **exp_cfg}

    auto_fill_default(cfg)

    return cfg


def auto_fill_default(cfg):
    res_pulse = parse_res_pulse(cfg)

    if "readout_length" not in cfg:
        cfg["readout_length"] = res_pulse["length"]

    if "relax_delay" not in cfg:
        cfg["relax_delay"] = 0.0


def parse_cfg(value: str | dict, fallback: dict) -> dict:
    """Parse the configuration with one or list of fallback dict."""
    if isinstance(value, Mapping):
        return value

    if value in fallback:
        return fallback[value]
    raise ValueError(
        f"Cannot find key {value} in fallback, which has keys: {list(fallback.keys())}"
    )


def parse_res_pulse(cfg: dict, name: str = "res_pulse") -> dict:
    resonator = cfg["resonator"]
    return parse_cfg(cfg[name], cfg["global"]["res_cfgs"][resonator].get("pulses", {}))


def parse_qub_pulse(cfg: dict, name: str = "qub_pulse") -> dict:
    qubit = cfg["qubit"]
    return parse_cfg(cfg[name], cfg["global"]["qub_cfgs"][qubit].get("pulses", {}))


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
    def load_global(cls, filepath):
        assert not DefaultCfg.is_init_global(), "Configuration is already initialized."

        with open(filepath, "r") as f:
            glb_cfg = yaml.safe_load(f)

        cls.res_cfgs = glb_cfg["res_cfgs"]
        cls.qub_cfgs = glb_cfg["qub_cfgs"]
        cls.flux_cfgs = glb_cfg["flux_cfgs"]

    @classmethod
    def dump_global(cls, filepath=None):
        if filepath is None:
            filepath = f"cfg_{time.strftime('%Y%m%d_%H%M%S')}.yaml"

        # type cast all numpy types to python types
        def recast(obj):
            if hasattr(obj, "tolist"):
                obj = obj.tolist()
            if isinstance(obj, dict):
                return {k: recast(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [recast(v) for v in obj]
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        dump_cfg = recast(cls.dict())
        with open(filepath, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def set_res_cfg(cls, resonator, **cfg):
        cls.res_cfgs[resonator].update(cfg)

    @classmethod
    def set_qubit_cfg(cls, qubit, **cfg):
        cls.qub_cfgs[qubit].update(cfg)

    @classmethod
    def add_res_pulse(cls, resonator: str, overwrite=False, **pulse_cfgs):
        res_cfg = cls.res_cfgs[resonator]
        res_cfg.setdefault("pulses", {})
        if not overwrite:
            for key in pulse_cfgs:
                if key in res_cfg["pulses"]:
                    raise KeyError(
                        f"Key {key} already exists in resonator {resonator} pulses."
                    )
        res_cfg["pulses"].update(pulse_cfgs)

    @classmethod
    def add_qub_pulse(cls, qubit, overwrite=False, **pulse_cfgs):
        qub_cfg = cls.qub_cfgs[qubit]
        qub_cfg.setdefault("pulses", {})
        if not overwrite:
            for key in pulse_cfgs:
                if key in qub_cfg["pulses"]:
                    raise KeyError(f"Key {key} already exists in qubit {qubit} pulses.")
        qub_cfg["pulses"].update(pulse_cfgs)

    @classmethod
    def dict(cls):
        return {
            "res_cfgs": cls.res_cfgs,
            "qub_cfgs": cls.qub_cfgs,
            "flux_cfgs": cls.flux_cfgs,
        }
