---
status: accepted
---

# ArbWaveformDatabase owns shared arbitrary waveform asset operations

Core arbitrary waveform asset operations live in `meta_tool.ArbWaveformDatabase`, not only in a measure-gui-local service, so notebooks and GUI use the same validation, formula rendering, single-file `.npz` persistence, rename/delete, and on-demand inspection summary semantics. `list()` is a cheap directory index operation that does not open every `.npz` and returns sorted data keys; `list_entries()` is the cheap directory index variant that also returns filesystem facts such as `mtime` and `file_size`. `inspect(data_key)`, `load(data_key)`, and preview generation open a selected asset and compute duration, sample count, Q presence, peak magnitude (`peak_abs`), and recipe summary on demand. GUI services/controllers adapt this shared repository for user interaction and remote/MCP policy, but they do not duplicate the asset repository rules or persist derived metadata.
