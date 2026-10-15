"""Typed in-process dependencies for waveform and materialization contracts."""

from pathlib import Path

import numpy as np
from zcu_tools.cfg_model import CfgInputValue, ModuleSpec, WaveformSpec
from zcu_tools.core.waveform_format import WaveformSamples


class AssetInfo:
    """Current asset reads without repository globals or shared mutable state."""

    def __init__(self, data_key: str = "shape") -> None:
        """Serve one configured asset key, initially with duration 2.5 microseconds."""
        self._data_key = data_key
        self.current_duration = 2.5

    def duration(self, data_key: str) -> float:
        """Return the configured asset duration; other keys fail explicitly."""
        if data_key != self._data_key:
            raise ValueError(f"Missing asset: {data_key}")
        return self.current_duration

    def path_for(self, data_key: str) -> Path:
        """Reject native conversion: these in-memory samples have no file."""
        raise ValueError("In-memory assets have no playback path")

    def samples(self, data_key: str) -> WaveformSamples:
        """Read isolated I data and a current reference axis; Q is zero."""
        return WaveformSamples(
            time=np.array([0.0, self.duration(data_key)]),
            idata=np.array([0.0, 1.0]),
            qdata=None,
        )


class Resolver:
    """Fixed Spec snapshots for public factory tests; each instance owns its maps."""

    def __init__(self) -> None:
        self.modules: dict[str, ModuleSpec] = {}
        self.waveforms: dict[str, WaveformSpec] = {}

    def module_spec(self, path: str) -> ModuleSpec:
        """Read an isolated module Spec; absent keys raise KeyError."""
        if path not in self.modules:
            raise KeyError(f"Missing module key: {path}")
        return self.modules[path].model_copy(deep=True)

    def waveform_spec(self, path: str) -> WaveformSpec:
        """Read an isolated waveform Spec; absent keys raise KeyError."""
        if path not in self.waveforms:
            raise KeyError(f"Missing waveform key: {path}")
        return self.waveforms[path].model_copy(deep=True)


def mux_raw(count: int = 2) -> dict[str, CfgInputValue]:
    """Return numeric mux input with deliberately descending frequency/channel order."""
    return {
        "type": "readout/mux",
        "gen_ch": 9,
        "nqz": 2,
        "mixer_freq": 6000.0,
        "length": 2.0,
        "ro_length": 1.5,
        "tones": [
            {"freq": 6020.0 - i * 40.0, "gain": -0.1, "phase": 450.0, "ro_ch": 7 - i}
            for i in range(count)
        ],
    }
