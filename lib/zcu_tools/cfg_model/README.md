# cfg_model — 共享配置核心

**Last updated:** 2026-10-06，Spec／Cfg 與注入式 materialization

`cfg_model` 提供不含執行行為或 I/O 的配置模型與具名依賴契約。
Caller 從 package root import 公開名稱。

- `ConfigBase` 提供 Pydantic 欄位驗證、巢狀更新與序列化。非 module consumer 的行為保持不變。
- `WaveformSamples` 表示一次隔離的 asset 讀取。它攜帶 reference time axis 與 normalized I/Q samples。
- `ModuleSpec`／`WaveformSpec` 表示持久候選，保留具體 key 的 child 引用。兩個 `spec_from_raw` 入口驗證持久值，不查 library 或 assets，不將 runtime 資訊轉成字串。
- `ModuleCfg`／`WaveformCfg` 表示已展開的執行參數。它們只提供欄位驗證與 `set_param`，不建立 program runtime 物件。
- 兩個 cfg factory 透過 `CfgValidationContext` 注入 `ModuleResolver` 與 `WaveformAssetInfo`。單次 materialization 共用引用鏈，錯誤保留 root、chain 與期望 family；既有 cfg child 也重新驗證。
- Scalar leaf 由 Spec／Cfg 共用。純 Spec 驗證拒絕 composite runtime cfg、QICK sweep、complex、array，以及已綁定資產的 arb leaf。
- `WaveformAssetInfo` 提供固定 scope 的 duration 與 samples 讀取。Provider 擁有 archive 驗證與 I/O，consumer 不重做這些工作。Arb binding 不序列化，cfg deepcopy 保留同一 owner，不凍結 bytes。

核心只依賴外部函式庫與 `utils`，不 import `program`、`resources` 或 GUI。
Asset provider 固定目錄，不凍結 bytes。每次讀取都反映該目錄內的當下內容。
