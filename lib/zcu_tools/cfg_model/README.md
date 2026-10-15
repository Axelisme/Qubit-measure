# cfg_model — 共享配置核心

**Last updated:** 2026-10-15，FieldMeta 分類名稱與前端政策的責任邊界

`cfg_model` 提供不含執行行為或 I/O 的配置模型與具名依賴契約。
Caller 從 package root import 公開名稱。

- `ConfigBase` 提供 Pydantic 欄位驗證、巢狀更新與序列化。非 module consumer 的行為保持不變。
- `FieldMeta` 在 core 與持久 Spec 欄位承載 label、unit、decimals、description 與 group，不改驗證或序列化的值。group 是 MCP schema 可用的前端中立分類名稱 hint，因此由 core 保存。前端用 `get_field_meta` 讀取，並決定實際的 widget、排版與折疊行為。`Expr`、`Range`、`AliasName`、`Hidden`、`Locked`、`Slot` 與 `Template` 是前端中立的起始值宣告。它們不求值、不讀來源。上層 Form 將它們翻成輸入與 cfg；`TemplateDef` 的持久 Spec 目錄由呼叫者注入。
- `WaveformSamples` 由 core.waveform_format 擁有。它攜帶一次 asset 讀取的 reference time axis 與 normalized I/Q samples，不從 cfg_model re-export。
- `ModuleSpec`／`WaveformSpec` 表示持久候選，保留具體 key 的 child 引用。兩個 `spec_from_raw` 入口驗證持久值，不查 library 或 assets，不將 runtime 資訊轉成字串。
- `ModuleCfg`／`WaveformCfg` 表示已展開的執行參數。它們只提供欄位驗證與 `set_param`，不建立 program runtime 物件。
- `CfgInputMap`／`CfgInputValue` 是 factory 的未驗證唯讀輸入，接受 recursive mapping／sequence 與 `ConfigBase` token。具體 factory 仍按自己的 schema 驗證，不因 token 已是 model 就信任它。
- 兩個 cfg factory 透過 `CfgValidationContext` 注入 `ModuleResolver` 與 `WaveformAssetInfo`。單次 materialization 共用引用鏈，錯誤保留 root、chain 與期望 family；既有 cfg child 也重新驗證。
- Scalar leaf 由 Spec／Cfg 共用。純 Spec 驗證拒絕 composite runtime cfg、QICK sweep、complex、array，以及已綁定資產的 arb leaf。
- `WaveformAssetInfo` 提供固定 scope 的 duration、samples 與 path_for。Path 查詢只驗 key 與檔存在，不讀 bytes。Provider 擁有 archive 驗證與 I/O，consumer 不重做這些工作。Arb binding 不序列化，cfg deepcopy 保留同一 owner，不凍結 bytes。

本 package 只依賴外部函式庫、utils 與 core.waveform_format，不 import `program`、`resources` 或 GUI。
Asset provider 固定目錄，不凍結 bytes。每次讀取都反映該目錄內的當下內容。
