> 狀態：已退役（2026-09-27）。跨模組決策由 [ADR-0063](../0063-persistence-ownership.md) 接替；局部契約見 [waveform assets](../../../lib/zcu_tools/resources/waveform_assets.md) 與 [program](../../../lib/zcu_tools/program/v2/README.md)。以下保留歷史正文。

---
status: accepted
---

# Arbitrary waveform data uses a reference time axis

Arbitrary waveform playback data carries a reference time axis and is not stretched or compressed by `ModuleLibrary` configuration. A `style:"arb"` waveform entry stores only the asset `data` key; it does not store a separate length field. The playback length is the asset duration (`time[-1]`) computed from `ArbWaveformDatabase.inspect(data)`, and the program layer samples the stored data over that full asset duration. To shorten, extend, or otherwise retime an arbitrary waveform, change the asset arrays or formula recipe rather than overriding length in the waveform config.
