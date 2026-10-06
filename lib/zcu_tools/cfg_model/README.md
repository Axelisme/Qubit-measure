# cfg_model — 共享配置核心

**Last updated:** 2026-10-06，package 與 waveform asset 依賴

`cfg_model` 提供不含執行行為或 I/O 的配置模型與具名依賴契約。
Caller 從 package root import 公開名稱。

- `ConfigBase` 提供 Pydantic 欄位驗證、巢狀更新與序列化。非 module consumer 的行為保持不變。
- `WaveformSamples` 表示一次隔離的 asset 讀取。它攜帶 reference time axis 與 normalized I/Q samples。
- `WaveformAssetInfo` 提供固定 scope 的 duration 與 samples 讀取。Provider 擁有 archive 驗證與 I/O，consumer 不重做這些工作。

核心只依賴外部函式庫與 `utils`，不 import `program`、`resources` 或 GUI。
Asset provider 固定目錄，不凍結 bytes。每次讀取都反映該目錄內的當下內容。
