# waveform_format — samples and NPZ codec

**Last updated:** 2026-10-21

**讀者：** framework-dev。波形 samples、NPZ codec 與格式錯誤契約。

此模組只接收值或 bytes。`WaveformSamples`、`WaveformFile`、`validate_samples`、`decode_npz` 與 `encode_npz` 是公開接縫。

- Reference time 使用 µs，從零嚴格遞增。I/Q 是有限的一維 float64，數量 2–1,000,001，magnitude 不超過 1。無 Q 或全零 Q 用 None 表示，NPZ 仍寫 qdata。
- NPZ 必須含 time/idata/qdata，可含 scalar recipe_json，不接受其他 keys。Codec 使用 allow_pickle=False。
- Recipe 只驗 JSON 語法，保留原文。Formula 語意由 resources 驗證，program/sim 只播放 samples。
- `WaveformFormatError` 以 reason 與 typed data 描述格式錯誤。Resources 在 repository 接縫轉譯成既有 ArbWaveformError，保留原因。
- 每次驗證或 decode 產生本次讀取的 arrays。不凍結 bytes，不管理路徑或 asset version。
