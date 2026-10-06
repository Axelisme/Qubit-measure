"""Typed in-process dependencies for waveform and materialization contracts."""

import numpy as np
from zcu_tools.cfg_model import ModuleSpec, WaveformSamples, WaveformSpec


class AssetInfo:
    """Current asset reads without repository globals or shared mutable state."""

    def __init__(self) -> None:
        self.current_duration = 2.5

    def duration(self, data_key: str) -> float:
        """Return the current duration of shape; other keys fail explicitly."""
        if data_key != "shape":
            raise ValueError(f"Missing asset: {data_key}")
        return self.current_duration

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
        """Read a module Spec; absent or wrong-partition keys raise ValueError."""
        if path not in self.modules:
            raise ValueError(f"Missing module key: {path}")
        return self.modules[path].model_copy(deep=True)

    def waveform_spec(self, path: str) -> WaveformSpec:
        """Read a waveform Spec; absent or wrong-partition keys raise ValueError."""
        if path not in self.waveforms:
            raise ValueError(f"Missing waveform key: {path}")
        return self.waveforms[path].model_copy(deep=True)
