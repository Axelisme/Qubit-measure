---
status: accepted
---

# Arbitrary waveform data uses a reference time axis

Arbitrary waveform playback data carries a reference time axis and is not stretched or compressed by `ModuleLibrary` configuration. A `style:"arb"` waveform entry stores only the asset `data` key; it does not store a separate length field. The playback length is the asset duration (`time[-1]`) computed from `ArbWaveformDatabase.inspect(data)`, and the program layer samples the stored data over that full asset duration. To shorten, extend, or otherwise retime an arbitrary waveform, change the asset arrays or formula recipe rather than overriding length in the waveform config.
