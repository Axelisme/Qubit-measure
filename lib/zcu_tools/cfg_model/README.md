# cfg_model — 共享配置核心

**Last updated:** 2026-10-20，Hidden literal 填入值與 Template／Slot seed

`cfg_model` 提供不含執行行為或 I/O 的配置模型與具名依賴契約。
Caller 從 package root import 公開名稱。

`PlaceholderRegistry` 由 session 組裝建立，初始為空。`Template.bindings` 提供任意具名替換，覆蓋 registry defaults。
`TemplateDef` 建構只驗語法，展開時才要求 binding。Registry 不解析 alias、不讀寫檔案，也不保存 process 全域狀態。

- `ConfigBase` 提供 Pydantic 欄位驗證、巢狀更新與序列化。非 module consumer 的行為保持不變。
- `FieldMeta` 在 core 與持久 Spec 欄位承載 label、unit、decimals、description、group、readonly 與 element_label，不改 cfg 驗證或序列化的值。group 是 MCP schema 可用的前端中立分類名稱 hint。前端用 `get_field_meta` 讀取，並決定實際的 widget、排版與折疊行為。`readonly=True` 由上層 Form core 強制執行，同值寫入通過，不同值拒絕；element_label 提供 picker label。
- `Expr`、`Range`、`LogPoints`、`AliasName`、`Hidden`、`Slot` 與 `Template` 是前端中立的起始值宣告。它們不求值、不讀來源。`Hidden(value)` 攜帶 core 單位的 literal，包含明填 None；Form 隱藏欄位並保留 core 驗證。`Hidden()` 保留普通 core 欄位的 default 還原語意。`Template.set`／`Slot.set` 接受 dotted Spec 路徑的 `Hidden(value)`，不宣告值的來源或更動 Run。`LogPoints.include_zero` 預設 True，False 請求正值 Log／List；分支與產生規則由 form_adapter 擁有。固定欄位使用初值與 readonly，不提供 `Locked` 或 Slot／Template 的 lock 參數。FormNode 的 list 按位置提供 model 元素或 scalar list 的初值，各位置的唯讀值可以不同。`TemplateDef` 的持久 Spec 目錄由呼叫者注入，Form 保留各範本的非 lock 初值與 input projection，不以單值 Literal 實作 lock。還原保留唯讀欄位的存檔值，不重算宣告初值。Core cfg 自有的單值 Literal 仍保留其固定值語意。
- Pulse 與 Pulse readout 依模型宣告結構排列主要欄位，順序為 Waveform、Gen ch、Freq、Gain。Form 不另設前端排序表。Pulse 的持久欄位仍為 `ch`，Pulse readout 仍為 `gen_ch`；儲存形狀與驗證不因排版改變。Mux readout 沒有 waveform，不新增該欄位。
- `Range.start`／`stop` 與 `Range.between` 的端點接受 `None`。它表示 Form 中可見空白且必填的端點，不是 0／1 或可執行預設。即使對應 input key 缺席，生成 model 仍要求操作員填入數字才能提交。另一端點與 `expts` 不受影響。Literal 與 `Expr` 端點沿用既有語意。
- `WaveformSamples` 由 core.waveform_format 擁有。它攜帶一次 asset 讀取的 reference time axis 與 normalized I/Q samples，不從 cfg_model re-export。
- `ModuleSpec`／`WaveformSpec` 表示持久候選，保留具體 key 的 child 引用。兩個 `spec_from_raw` 入口驗證持久值，不查 library 或 assets，不將 runtime 資訊轉成字串。
- `spec_types` 集中持久 Spec union 的展開。`concrete_spec_types` 供 TemplateDef 檢查宣告中的具體型別；`is_module_spec_type`／`is_waveform_spec_type` 供 library consumer 判定 family，接受已宣告型別及其 subclass。
- `ModuleCfg`／`WaveformCfg` 表示已展開的執行參數。它們只提供欄位驗證與 `set_param`，不建立 program runtime 物件。
- `readout/pulse` 獨立持有平鋪的 generator／ADC 欄位與 waveform。唯一 `freq` 同時供 generator／ADC 使用，waveform 仍可引用 key。Readout 不引用 Pulse，也沒有 DirectReadout；Reset 的 Pulse 引用保持不變。Pulse cfg 不保存 `ro_ch`，ADC matching 由 program runtime 提供。
- `readout/mux` 的持久 Spec 與數字 cfg 共用驗證。Module 明填 generator、Nyquist zone 與 mixer；有序的 1 至 8 筆 tones 各自保存頻率、gain、phase 與 logical ADC channel。每筆 tone 的選填 label 預設空字串，隨 Spec、cfg 與 library 保存，只作顯示且可重複。Tone 序號定義資料軸，program 不解讀 label。時間使用共用 const duration 與 ADC window，不含 waveform、逐 tone 引用或 sweep。整個 module 可透過固定 snapshot 的 key materialize；`set_param` 拒絕更新，caller 必須重建數字 cfg 與 program。
- `CfgInputMap`／`CfgInputValue` 是 factory 的未驗證唯讀輸入，接受 recursive mapping／sequence 與 `ConfigBase` token。具體 factory 仍按自己的 schema 驗證，不因 token 已是 model 就信任它。
- 兩個 cfg factory 透過 `CfgValidationContext` 注入 `ModuleResolver` 與 `WaveformAssetInfo`。單次 materialization 共用引用鏈，錯誤保留 root、chain 與期望 family；既有 cfg child 也重新驗證。
- Scalar leaf 由 Spec／Cfg 共用。純 Spec 驗證拒絕 composite runtime cfg、QICK sweep、complex、array，以及已綁定資產的 arb leaf。
- `WaveformAssetInfo` 提供固定 scope 的 duration、samples 與 path_for。Path 查詢只驗 key 與檔存在，不讀 bytes。Provider 擁有 archive 驗證與 I/O，consumer 不重做這些工作。Arb binding 不序列化，cfg deepcopy 保留同一 owner，不凍結 bytes。

本 package 只依賴外部函式庫、utils 與 core.waveform_format，不 import `program`、`resources` 或 GUI。
Asset provider 固定目錄，不凍結 bytes。每次讀取都反映該目錄內的當下內容。

Readout 的跨 owner 契約見 [ADR-0074](../../../docs/adr/0074-flat-readout-module.md)。
