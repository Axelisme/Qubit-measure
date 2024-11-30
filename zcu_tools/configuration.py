import time
from typing import Union

import yaml

from .tools import deepupdate, numpy2number


class DefaultCfg:
    res_cfgs = None
    qub_cfgs = None
    flux_cfgs = None
    exp_default = {}

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
    def get_res_pulse(cls, resonator: str, pulse_name: str) -> dict:
        res_cfg = cls.res_cfgs[resonator]
        return res_cfg["pulses"][pulse_name]

    @classmethod
    def set_qub_pulse(cls, qubit: str, behavior="force", **pulse_cfgs):
        qub_cfg = cls.qub_cfgs[qubit]
        qub_cfg.setdefault("pulses", {})
        deepupdate(qub_cfg["pulses"], pulse_cfgs, behavior=behavior)

    @classmethod
    def get_qub_pulse(cls, qubit: str, pulse_name: str) -> dict:
        qub_cfg = cls.qub_cfgs[qubit]
        return qub_cfg["pulses"][pulse_name]

    @classmethod
    def set_labeled_flux(cls, qubit: Union[str, dict], method: str, **lbd_flux):
        if isinstance(qubit, str):
            qubit = cls.qub_cfgs.setdefault(qubit, {})
        labeled_flux = qubit.setdefault("labeled_flux", {})
        labeled_flux.set_default(method, {}).update(lbd_flux)

    @classmethod
    def get_labeled_flux(cls, qubit: Union[str, dict], method: str) -> dict:
        if isinstance(qubit, str):
            qubit = cls.qub_cfgs.get(qubit, {})
        return qubit["labeled_flux"].get(method, {})

    @classmethod
    def set_default(cls, **kwargs):
        deepupdate(cls.exp_default, kwargs, behavior="force")

    @classmethod
    def dict(cls):
        return {
            "res_cfgs": cls.res_cfgs,
            "qub_cfgs": cls.qub_cfgs,
            "flux_cfgs": cls.flux_cfgs,
        }
