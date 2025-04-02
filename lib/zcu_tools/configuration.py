from typing import Literal, Optional

from .tools import deepupdate, numpy2number


class DefaultCfg:
    dac_cfgs = {}
    adc_cfgs = {}
    dev_cfgs = {}

    @classmethod
    def load(cls, filepath: str):
        """
        Load configuration from a YAML file.

        Args:
            filepath (str): Path to the YAML configuration file.

        Raises:
            AssertionError: If the loaded configuration is not a dictionary.
        """
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
        """
        Dump the current configuration to a YAML file.

        Args:
            filepath (Optional[str]): Path to save the YAML configuration file. If None, a timestamped file will be created.
        """
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
        """
        Update DAC configurations.

        Args:
            behavior (Literal["error", "force", "ignore"]): Behavior when updating configurations. Defaults to "force".
            **dac_cfgs: Key-value pairs of DAC configurations to update.

        Warnings:
            If "pulses" in dac_cfgs, a warning is printed recommending the use of set_pulse instead.
        """
        if "pulses" in dac_cfgs:
            print(
                "Warning: Use set_dac to set pulse is not recommended, use set_pulse instead."
            )
        deepupdate(cls.dac_cfgs, dac_cfgs, behavior=behavior)

    @classmethod
    def get_dac(cls, name: str):
        """
        Retrieve a specific DAC configuration by name.

        Args:
            name (str): Name of the DAC configuration to retrieve.

        Returns:
            dict: The DAC configuration if found, otherwise None.
        """
        return cls.dac_cfgs.get(name)

    @classmethod
    def set_adc(
        cls, behavior: Literal["error", "force", "ignore"] = "force", **adc_cfgs
    ):
        """
        Update ADC configurations.

        Args:
            behavior (Literal["error", "force", "ignore"]): Behavior when updating configurations. Defaults to "force".
            **adc_cfgs: Key-value pairs of ADC configurations to update.
        """
        deepupdate(cls.adc_cfgs, adc_cfgs, behavior=behavior)

    @classmethod
    def get_adc(cls, name: str):
        """
        Retrieve a specific ADC configuration by name.

        Args:
            name (str): Name of the ADC configuration to retrieve.

        Returns:
            dict: The ADC configuration if found, otherwise None.
        """
        return cls.adc_cfgs.get(name)

    @classmethod
    def set_pulse(cls, **pulse_cfgs):
        """
        Set or update pulse configurations.

        Args:
            **pulse_cfgs: Key-value pairs of pulse configurations to set or update.
        """
        dac_pulses = cls.dac_cfgs.setdefault("pulses", {})
        # directly overwrite the pulse
        for k, v in pulse_cfgs.items():
            dac_pulses[k] = v

    @classmethod
    def get_pulse(cls, name: str, raise_err=True) -> dict:
        """
        Retrieve a specific pulse configuration by name.

        Args:
            name (str): Name of the pulse configuration to retrieve.

        Returns:
            dict: The pulse configuration if found, otherwise an empty dictionary.
        """
        pulses_cfg = cls.dac_cfgs.get("pulses", {})
        if name not in pulses_cfg and raise_err:
            raise KeyError(f"Pulse {name} not found in DAC configurations.")
        return pulses_cfg.get(name, {})

    @classmethod
    def clear_pulses(cls):
        """
        Clear all pulse configurations.
        """
        cls.dac_cfgs.pop("pulses", None)

    @classmethod
    def set_dev(
        cls, behavior: Literal["error", "force", "ignore"] = "force", **dev_cfgs
    ):
        """
        Update device configurations.

        Args:
            behavior (Literal["error", "force", "ignore"]): Behavior when updating configurations. Defaults to "force".
            **dev_cfgs: Key-value pairs of device configurations to update.
        """
        deepupdate(cls.dev_cfgs, dev_cfgs, behavior=behavior)

    @classmethod
    def get_dev(cls, name: str):
        """
        Retrieve a specific device configuration by name.

        Args:
            name (str): Name of the device configuration to retrieve.

        Returns:
            dict: The device configuration if found, otherwise None.
        """
        return cls.dev_cfgs.get(name)

    @classmethod
    def dict(cls):
        """
        Get the current configuration as a dictionary.

        Returns:
            dict: A dictionary containing all DAC, ADC, and device configurations.
        """
        return {
            "dac_cfgs": cls.dac_cfgs,
            "adc_cfgs": cls.adc_cfgs,
            "dev_cfgs": cls.dev_cfgs,
        }
