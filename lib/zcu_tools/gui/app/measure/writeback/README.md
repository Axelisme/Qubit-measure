# `gui.app.measure.writeback`

**Last updated:** 2026-10-30

**讀者：** framework-dev。寫回提案、review widget 與逐項 Apply 契約。

本模組擁有 parameter、module 與 record 的 proposal、draft、service、control 與 Qt review widget。
`items.py` 定義 adapter 的可變 proposals；`contracts.py` 定義 owner 的 edit、preview 與 submission records。
`service.py` 的 `WritebackDraft` 擁有每份 draft 的 entries、Forms、固定來源、summaries、applied state 與操作。
`WritebackService` 只建立 draft、注入既有 context 寫入能力，並協調跨 draft 的 `accept_drafts`。
Control 保留 tab／pane admission、事件與 detached receipt；UI 和 remote 繼續使用 control。

## Complete module proposals

`calibrated_module_writeback(parameters, tested_module, target=...,
description=..., field_updates=...)` builds one complete `ModuleWriteback`.
The caller supplies a read-only Run/Load ParameterContext, the native tested
module cfg, and exact dotted Spec paths with new values.
The builder resolves the captured slot, copies `LibrarySnapshot.module_spec()`,
and materializes `get_module()` through the existing native converter.
It compares every other field and reports differing dotted paths.
Equivalent inline values or different child keys with equal materialized values
are accepted. The proposal preserves the library's original child keys.

Only the exact updated paths are excluded from comparison. Updating
`tones.1.classifier` still checks that tone's frequency and ADC.
`resources.parameter_fields.set_model_value` writes existing indexed elements;
Spec validation remains with cfg_model. The builder has no classifier policy.
It does not read the live target, write files or change Apply semantics.

Primary and Post remain complete replacements. One batch accepts Primary before
Post. Separate Apply calls use the last replacement, so applying center-only
Primary after Post removes radius and matrix. Library stores no provenance.

## Dialog lifetime

The ledger owns a LifetimeScope and retires all open scalar/module editors when
its host calls `detach()`. ExpTabWidget registers that cleanup for Analysis and
Post-Analysis. Each dialog closes its own scope before completion. Module-form
subscriptions and editor callbacks are removed before the tab's Qt children are
deleted. Owner retirement discards pending input and does not commit Apply-as.
Normal Done retains its existing name-commit policy. No destroyed closure owns
form cleanup.

## Initial selection and notes

Adapter 可省略不合適的候選，或以 proposal 的 `selected=False` 保留候選但不預選。
`selected` 預設為 True，`note` 預設為 None。Adapter 自行判斷實驗品質，writeback 不比較品質閾值。
Draft 複製 proposal 的初選與 note。GUI 在目的地下面顯示 note，RPC preview 也回傳同一段文字。
人工可改勾選並 Apply。Edit、retarget 與 Apply 保留 note，不重新判斷品質；人工勾選不表示品質合格。

## Record 列

Record 的目標欄顯示 `<元件名> · record`。未 Apply 時保留共用的 `*` 標記，成功後移除。
Current 欄顯示 `entry record`。Proposed 欄與 tooltip 逐行列出全部 `key=value`，保留字串與巢狀 JSON 的完整內容。
每個 key 與值的前綴可見。長值可能被欄寬裁切，tooltip 與 Copy 仍保留完整內容。
Widget 將每個欄位當作一般 JSON 呈現，不識別裝置或單位。
Copy 將完整 JSON 放入剪貼簿。Record 沒有 Edit 動作，也不開啟編輯對話框。

`RecordWriteback(target_name, description, record=...)` 提供 adapter 組裝的 JSON object。
框架不解析欄位、不補座標，也不因缺裝置資料而省略 proposal。
Append owner 加入 `component`、`run_id`、`point`、`at` 與 `source` provenance 信封。
`component` 由 table 的具體 component binding 提供，producer 同名欄位不能覆寫它。
格式由 adapter 間的 policy 決定，見 [lab authoring 的 Record policy](../../../../../../zcu_lab/v2/measure-authoring.md#record-policy)。
記錄放在 `records/components/<target>.jsonl`，事件放在 `records/ledger.jsonl`。
Wire kind、session prefix 與 written key 分別為 `record`、`record-<n>` 與 `records`，不讀舊名稱或路徑。

## Apply

`WritebackDraft` 從自己的固定 Run／Load capture 解析選定提案與 module Form，
再交給 context owner。來源只提供提案求值與 alias／slot 路徑，不決定寫入 point。
Preview 與 Apply 取當下作用中的文件組，global 寫 entry 層，point 寫目前 point。
Preview 的位置只含 entry_name 與 point，global 的 point 為 null。
未開 entry 仍可預覽，位置兩欄與 current 值皆為 null，但 Apply 以 no_context 拒絕。
缺少歷史來源不阻擋 concrete parameter／record 提案。
Record 的 point 跟隨提交位置，global 樣本保存 null。
全部提案通過輸入驗證後，共用 executor 先追加一筆 accepted
ledger event，再依序提交 point 參數組、library 組與逐筆 Record，組內保留提案順序。
Parameter 與 library 各用一次單檔交易，組內任一項無效整組不寫，Record 各列追加樣本。
第一個失敗會停止後組，保留 accepted event 與已提交組，並標記已成功的 draft 列。
Ledger append 失敗時不提交任何列。Apply 不提供跨列 rollback 或自動 retry。

資源層對完整 serialization tree 驗證一次 YAML 表示。
Library 只保存 Spec，不保存 per-leaf stderr 或 provenance，也不驗來源事件或附件。
DocumentStore 每次讀取磁碟 bytes，未變內容沿用驗證快取。
每次 Apply 仍驗證新 draft 的完整模型，並保留 accepted 稽核事件。
各組提交到固定目的地，錯誤分類與 applied 狀態由已提交項目決定。
Apply 同步執行。

## Captured device evidence

`create_draft(device_snapshot=...)` 接受固定裝置設定或讀取 detached Run 證據的 callback。
建立 draft 不呼叫 callback，`draft.device_snapshot` 才讀取並回傳隔離設定。
歷史設定格式錯誤時，這次讀取保留 experiment owner 的欄位錯誤，不改成 None 或空 mapping。
Post draft 延續 Primary 的 reader，沒有裝置需求的 Analyze／Post 不先驗證這份證據。

## Captured acceptance

`accept_drafts` 在同一次 owner 呼叫核對全部 selected drafts 的 IDs，再依輸入群組順序寫入。
值與目的地取 owner 當下的 draft，不比對 preview、不還原捕捉時的 proposal。
Writeback control 固定 Primary 在 Post 前，並在提交前檢查 permission、idle 與 active draft。
Unknown／duplicate IDs 在第一筆寫入前拒絕。Draft 編輯或替換本身不拒絕接受，見 ADR-0075。
提交中的失敗回 confirmed IDs、逐項 actual values 成功前綴及診斷，不 retry／rollback。
Actual values 由 resources/context 的實際寫入 owner 逐項捕捉，整組成功才回傳。GUI 以 committed index 對回供應 items，再關聯 ID、pane 與固定目的地；不用 path 或 applied 數量配對。
不在整批後 readback，後項覆寫與讀取故障不改先前已確認的 receipt。
Draft 的 `apply` 與 `writeback_write` 保留原有 raise 與 edit/write 契約。
Service 先用 draft 的 `validate_selection` 檢查 owner、closed 與 IDs，不解析值或固定目的地。
再用 draft 的 `accept` 取得逐 draft application result；`applied_ids` 與 `values` 是已確認前綴，
`written` 則包含全部準備好的路徑，不能當成已提交清單。

## Submission receipts

Control 的 `begin_writeback(tab_id)` 配置一次性 `submission_id`，不驗證 draft、不寫入，也不授權提交。
`write_writeback_for_pane` 與 `accept_writeback` 可帶該 ID，仍在 owner thread 同步執行。
同 ID 只准提交一次，未知 ID 或不同 tab 的 ID 會拒絕。
`get_writeback_receipt` 回 detached 的 pending／completed／failed 狀態與成功前綴。
每列保留 pane、item ID、kind、實際值、concrete target 與提交時的 entry_name／point。
Destination 只含 entry_name 與 point，不含 kind。提交成功的 entry_name 必為字串。
Pending 可表示只 reserve 或仍在執行，不能推論尚未寫入。
Receipt 留在 GUI 生命週期的記憶體。後續 draft 編輯或目的地變更不改舊 receipt。
逾時只結束 caller 的等待，不取消同步寫入，也不觸發 retry 或 rollback。
