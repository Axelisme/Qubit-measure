# zcu_tools.core — shared stateless contracts

**Last updated:** 2026-10-09，complex view 與 raw capture 分界

`core` 提供 resources 與 program 共用的無狀態契約。Repo 內只依賴 utils，不持有 GUI、資源 scope 或硬體。

- `complex_codec` 擁有唯一 `{"__complex__": [re, im]}` scalar 格式，供 GUI 與 resources 共用。Schema 決定有限值限制，expression 決定數值語意。Resources view 將 scalar tag 讀為 native complex，手動 scalar 寫入編回 tag。Snapshot、document_value、document_fields 與 ParameterContext 等 non-view 路徑保留 raw tag。
- `waveform_format` 擁有 NPZ samples、opaque JSON recipe、codec 與單一格式錯誤。它不讀寫檔案，也不解讀公式。
- Resources 負責 formula 語意、資產命名及檔案生命週期。Program 與 sim 播放 path 指定的 samples。
- `cfg_model` 與頂層 `format_version.py` 尚未搬入 core。搬遷排在 waveform 接線後。
