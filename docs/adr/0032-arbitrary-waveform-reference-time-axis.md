---
status: accepted
---

# Arbitrary waveform data uses a reference time axis

Arbitrary waveform playback data carries a reference time axis and is not stretched or compressed to match a `ModuleLibrary` arbitrary waveform length. The program layer samples the stored data on the requested playback window; if the requested length is shorter than the stored time-axis end, the trailing data is truncated and a warning is logged. This preserves the existing interpolation semantics while making accidental truncation visible.
