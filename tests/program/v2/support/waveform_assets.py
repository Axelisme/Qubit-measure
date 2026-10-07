"""Case-local waveform assets shared by runtime playback and simulation tests."""

from dataclasses import dataclass

from zcu_tools.cfg_model import WaveformSamples


@dataclass(frozen=True)
class InMemoryWaveformAsset:
    """Serve one case-local asset, copying arrays on every read.

    data_key is its concrete asset key. data holds validated, normalized I/Q
    samples on a strictly increasing reference time axis in microseconds.
    Other keys raise ValueError. No repository scope or global state is used.
    """

    data_key: str
    data: WaveformSamples

    def duration(self, data_key: str) -> float:
        """Return the reference duration in microseconds for this key."""
        return float(self.samples(data_key).time[-1])

    def samples(self, data_key: str) -> WaveformSamples:
        """Read isolated samples; reject keys outside this case's scope."""
        if data_key != self.data_key:
            raise ValueError(f"Missing asset: {data_key}")
        return WaveformSamples(
            time=self.data.time.copy(),
            idata=self.data.idata.copy(),
            qdata=self.data.qdata.copy() if self.data.qdata is not None else None,
        )
