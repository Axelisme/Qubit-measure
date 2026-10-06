"""Shared configuration schemas and injected dependencies, without I/O."""

from .base import ConfigBase
from .dependencies import WaveformAssetInfo, WaveformSamples

__all__ = ["ConfigBase", "WaveformAssetInfo", "WaveformSamples"]
