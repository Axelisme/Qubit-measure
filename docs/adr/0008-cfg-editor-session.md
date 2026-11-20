---
status: accepted
---

# 0008 — Headless Form 與 writeback draft

## 脈絡

Qt 與 agent 都要編輯 tab cfg、Library entry 與分析產生的 writeback proposal。可編輯狀態不能依賴 dialog 是否開啟，也不能讓 agent 修改一份 GUI 無法顯示的副本。

共用 Form 核心與固定來源見 [[0065]]，Measure 的 Form 定義見 [[0073]]。本篇記錄狀態的擁有者與 writeback 提交邊界。

## 決策

`gui.form.FormState` 保存可不完整的 input。`gui.widgets.form.FormEditor` 只呈現 caller-owned state 與固定 FormEnv，不擁有提交或保存政策。Detach 不提交文字、不重設 state，也不銷毀宿主的 draft。

- Measure tab 由 `TabFormResource` 持有 state、env 與 lifetime。Open／Reset 捕捉來源，Qt、remote 與 Run 使用同一資源。Run 只接受目前有效的 result，不重讀全域來源。
- Library 與 Inspect 由 session 的 `LibraryEntryForm` 持有 state、固定 DocumentSetCapture 與 arb 名稱。開啟既有 entry 時，生成器直接投影持久 Spec。Apply 取有效 Spec，交 `apply_library_change`，不經 runtime cfg 或 caller lowering。關閉時丟棄 form 即可，沒有 editor session registry、LRU 或 editor_id。
- 每個 `WritebackDraft` 擁有自己的 detached entries、固定來源、module Forms、summaries 與 applied state，以及 preview、edit、write、Apply 和 teardown。每個 module item 配一個 `ModuleEntryForm`，固定建立時的來源。Qt dialog 與 remote 都經 writeback control admission 讀寫同一 draft，dialog 是否存在不影響 draft。
- `WritebackService` 建立 draft 並注入既有 `ContextControlPort` 寫入能力，只保留 `accept_drafts` 的跨 draft 預驗證與成功前綴協調。`WritebackControlFacet` 繼續負責 tab／pane admission、事件與 detached receipt，不另設 per-draft service 介面。

Form 的狀態可以 Invalid。Preview 只供呈現，不是 Apply 許可。Library Apply 與 module writeback 提交都要求有效 result。文字解析錯誤若仍留在 view，宿主也要拒絕 Apply，不能提交 state 中較早的有效值。關閉 editor 丟棄未解析文字，不寫 Library。

## Writeback proposal 與提交

Adapter 可提供 `ParameterWriteback`、`ModuleWriteback`、`RecordWriteback` 與 `ArbAssetWriteback`。`ModuleWriteback` 直接提供持久 `ModuleSpec`，不提供舊編輯樹或 editor identity。Child library key 保持 key。`ArbAssetWriteback` 帶資產 key 與樣本，不是 library waveform Spec 的提案。Draft 可改它的勾選狀態，不能改 key 或樣本。Module Form 的 arb 選項是建立 draft 時的資產 key，加上同份 draft 的資產提案 key，不論有沒有勾選。

2026-11-19，使用者說：「然后发布 arbitrary 资产这是可以的，但是我其实觉得 recipe 内想要写档案也是没有问题的。」Writeback 可供 GUI 與 MCP 共用，recipe 寫檔只有 MCP 途徑。使用者比較兩案後，對維持 writeback 答：「可以」。比較與發布理由見 [討論紀錄](../discussions/2026-11-20-arb-asset-writeback.md)。

`gui.app.measure.writeback.calibrated_module_writeback` 擁有中立的完整 module proposal 建構。
Caller 提供固定 DocumentSetCapture、Run 的 native module cfg、目標與精確欄位路徑及新值。
Builder 以 `LibrarySnapshot.module_spec()` 的原 Spec 為底。
它以 `get_module()` 與 program 既有 native converter 取得可比較 cfg，只忽略指定更新路徑。
其他數值或結構不同時拒絕，列出 dotted 欄位。
相同 materialized 值可接受不同 key 或 inline 來源，proposal 保留 library 原 child keys。
Framework 不解讀 classifier，也不增加 Apply patch 或來源 capture。
依賴沿用 C8 分層與 C9 無循環 contract。

`resources.parameter_fields.set_model_value` 修改 detached Spec。
它可經過既有 list 的 canonical 非負十進位 index，只寫既有元素，不擴 tuple／ndarray 或建立容器。
Setter 與 projection reader 共用 index 驗證，cfg_model 在產生 proposal 前完整驗證 Spec。

2026-10-29 裁決：「中立 builder 放 gui.app.measure.writeback：caller 指定要更新的精確欄位路徑與新值，builder 以 LibrarySnapshot.module_spec() 為底、用 get_module() 的 materialized cfg 比較其餘欄位，保留 library key，回傳完整 ModuleSpec replacement；framework 不寫 classifier 特例，不改 apply／patch／capture。」

Analysis 與 Post-Analysis 各保存自己的 opaque draft。Remote 以 `(tab_id, subtab_id)` 定址 pane，再以 item session id 選項目。這個 id 只屬於 draft，不是 editor handle。`tab.writeback_set` 的 module edits 使用 Form 路徑陣列，回同一項目的 schema、input、values 與 errors。結構錯誤拒絕該 Form batch，普通 invalid 值留在 draft，Apply 與 `tab.writeback_write` 拒絕提交。

Selected Apply 先準備所有選中項目的值與目的地。任一 module Form Invalid 時，不 append accepted、不寫 target。每個選中 module 的 arb key 必須已存在於作用中 entry，或有同份 draft 的已選資產提案。缺少資產時在 append accepted 前拒絕該次 draft 的 Apply，錯誤指出 module 與 key。只檢查 module Spec 直接引用的 key 與沿作用中 library 可解析的 child 引用。無法解析的 library child 維持既有行為，這個邊界是 2026-11-20 的 agent 代裁，使用者同日覆核時回覆「確認」。GUI、MCP 與 recipe 用同一項檢查。Primary 與 Post 依序各自提交，後面的 draft 不能替前面的 draft 提供尚未發布的 key。

準備成功後，Context write owner 交 executor 先 append accepted，再依序提交：arb 資產一組、parameters 一組、library 一組、records 逐筆，組內順序保留。資產不覆寫。首錯停止，保留 accepted evidence 與已確認提交，不 rollback，也不保證跨檔交易，見 [[0063]]。

`tab.writeback_write` 依序調整 draft；後項失敗可保留前項 draft 修改，這不等於已寫入 target。

Draft 固定來源的 run_id 與參數環境，來源可以沒有持久 target。後來切 point、改 alias 或切換 pane 不改歷史求值內容。Preview 顯示當下作用中的位置，Apply 重新取得作用中文件組，不取歷史來源。GLOBAL 的目的地是 global 文件組，POINT 則是選定 point 文件組。未開 entry 仍可預覽合法 draft，current 為 None；非空 Apply 以 no_context 拒絕。空歷史環境不構成拒絕理由。

Preview 的 destination_context 與 receipt destination 只含 entry_name、point，不含 kind。Global 的 point 為 null；未開 entry 的 preview 兩欄都為 null，已提交 receipt 的 entry_name 則必為字串。Opaque actual_path 與來源 capture 不變，預覽不是寫入許可。

2026-10-29 使用者問：「關於代裁的23，真的需要知道entry/point的狀態嗎？不能只是聲明讀取或寫入當前激活的三個檔案嗎？」目前規則依此只呈現作用中位置，並允許 global 寫回，取代 2026-10-28 只接受 GUI 所選 point 的限制，見 [[0063]]。

### Parameter upsert

ParameterWriteback 與 ParameterUpdate 可帶 `create_kind: str | None = None`，由 adapter 指定元件不存在時使用的 registered kind。None 不建立元件，維持原本的缺元件錯誤。Apply 在同一次 parameter edit 中建立缺少的元件，再寫提案值與 metadata。建立時只提供提案欄位，其餘使用 registered model defaults。既有元件只更新指定欄位，不改 kind 或其他欄位。任一筆失敗整組不寫。Kind 與欄位仍由 lab 定義，framework 不寫死 resonator，遵循 [[0063]]。

`EditView.add_component(name, kind=..., **fields)` 與 set、set_meta 共用同一個 draft，不另開 edit。Views owner 內的 `add_component_to_draft` 仍只收文件 draft。EditView 的 source 由 owning view 傳入，只用於新增元件時的錯誤來源。

`onetone/range_find` 對每個有名字的 dip 提出 `<name>.freq` MHz，指定 create_kind="resonator"。不要求 capture，也不依 capture 內名稱篩列。Unassigned 不提案，identity_diagnostic 仍提示配對不確定。Preview 現值缺失時維持缺值顯示。GUI Apply、MCP apply_writeback 與 recipe accept 使用同一個 parameter item ID，不新增列種類、入口或按鈕。

例如 R1、R2 的 dip 在 6012.3 與 6123.4 MHz，作用中 point 只有 R1，接受兩列後覆寫 R1.freq 並建立只指定 freq 的 resonator R2。名稱拼錯時會建立多餘元件，這是已接受的取捨，Apply 前可取消勾選。此政策只由 range_find 啟用，onetone/freq 與 mux adapters 維持 create_kind=None；其他實驗需要建立元件時須由其 adapter 明確指定。

2026-11-16，使用者裁定「沒有就建立，有就複寫」，並對最終做法回答「可以，照這樣定」。使用者隨後要求 commit_parameter_updates 保持中立，反對為此擴大 add_component_to_draft 的參數型別，改由 EditView 提供 add_component。比較過的選項與被否決的理由見 [討論紀錄](../discussions/2026-11-16-range-find-writeback-upsert.md)。

Host 的 busy、writable、result、context 與 [[0070]] holder admission 仍各自生效。可預期的 `LibraryEntryInvalid` 保留 dialog、row、selection 與 draft，供使用者修正；非預期錯誤向外傳遞。

## 取捨與相鄰責任

GUI 與 remote 共用 headless state，所以 agent 的修改可由相應 editor 呈現，不必先開 dialog。各宿主仍有不同的提交用途，不能把 tab Run、Library Apply 與 writeback selected Apply 當成同一種交易。

Context 寫入權威見 [[0067]]，保存與 accepted evidence 見 [[0063]]，wire 與錯誤投影見 [[0068]]。舊 CfgDraft、CfgFormWidget、CfgEditorService 與 editor.* RPC 已退休，不提供相容入口。
