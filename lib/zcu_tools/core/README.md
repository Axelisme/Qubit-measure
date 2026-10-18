# zcu_tools.core — shared stateless contracts

**Last updated:** 2026-10-12，共用 numpy numeric scalar 投影

`core` 提供 resources 與 program 共用的無狀態契約。Repo 內只依賴 utils，不持有 GUI、資源 scope 或硬體。

- `complex_codec` 擁有唯一 `{"__complex__": [re, im]}` scalar 格式，供 GUI 與 resources 共用。Schema 決定有限值限制，expression 決定數值語意。Resources view 將 scalar tag 讀為 native complex，手動 scalar 寫入編回 tag。Snapshot、document_value、document_fields 與 ParameterContext 等 non-view 路徑保留 raw tag。
- `element.ElementIdentity` 是 captured actual、estimate 與 writeback candidate 共用的原始 list index／同次 name。它只作報告標識，不解析 alias、participants 或寫入目的地。
- `numeric_scalar.normalize_numpy_scalar` 將單一 numpy floating／integer scalar 投影為 Python float／int，供 AnalysisDocument 與 AcceptedWrite 共用。它不遞迴；finite 與 strict JSON 驗證留在各邊界。
- `simulation` defines immutable numeric acquire/round requests, snapshots, pulse events, readout windows and joint ADC signals. Templates store per-point event/readout arrays without expanding repetitions; `materialize_point` expands one diagnostic timeline. Integrated round statistics defer raw shot generation, sharing state counts across ADCs and retaining round totals within nearest-even rounding error. `SamplingIdentity` owns the common local PCG64 seed/stream format. Times are us, frequencies MHz, phase degrees and gain normalized. It has no driver, Module or GUI dependency. Ownership and shapes are documented in [program simulation](../program/v2/simulation/README.md).
- `waveform_format` 擁有 NPZ samples、opaque JSON recipe、codec 與單一格式錯誤。它不讀寫檔案，也不解讀公式。
- Resources 負責 formula 語意、資產命名及檔案生命週期。Program 與 sim 播放 path 指定的 samples。
- `cfg_model` 與頂層 `format_version.py` 尚未搬入 core。搬遷排在 waveform 接線後。
