---
status: accepted
---

# ADR-0008 — Headless Form 與 writeback draft

## 脈絡

Qt 與 agent 都要編輯 tab cfg、Library entry 與分析產生的 writeback proposal。可編輯狀態不能依賴 dialog 是否開啟，也不能讓 agent 修改一份 GUI 無法顯示的副本。

共用 Form 核心與固定來源見 [[0065]]，Measure 的 Form 定義見 [[0073]]。本篇記錄狀態的擁有者與 writeback 提交邊界。

## 決策

`gui.form.FormState` 保存可不完整的 input。`gui.widgets.form.FormEditor` 只呈現 caller-owned state 與固定 FormEnv，不擁有提交或保存政策。Detach 不提交文字、不重設 state，也不銷毀宿主的 draft。

- Measure tab 由 `TabFormResource` 持有 state、env 與 lifetime。Open／Reset 捕捉來源，Qt、remote 與 Run 使用同一資源。Run 只接受目前有效的 result，不重讀全域來源。
- Library 與 Inspect 由 session 的 `LibraryEntryForm` 持有 state、固定 ParameterContext 與 arb 名稱。開啟既有 entry 時，生成器直接投影持久 Spec。Apply 取有效 Spec，交 `apply_library_change`，不經 runtime cfg 或 caller lowering。關閉時丟棄 form 即可，沒有 editor session registry、LRU 或 editor_id。
- Writeback 由 `WritebackService` 持有 opaque draft。每個 module item 配一個 `ModuleEntryForm`，固定建立時的來源。Qt dialog 與 remote 都讀寫這個 state，dialog 是否存在不影響 draft。

Form 的狀態可以 Invalid。Preview 只供呈現，不是 Apply 許可。Library Apply 與 module writeback 提交都要求有效 result。文字解析錯誤若仍留在 view，宿主也要拒絕 Apply，不能提交 state 中較早的有效值。關閉 editor 丟棄未解析文字，不寫 Library。

## Writeback proposal 與提交

Adapter 的 `ModuleWriteback` 直接提供持久 `ModuleSpec`，不提供舊編輯樹或 editor identity。Child library key 保持 key。沒有現存 producer 的 waveform writeback 不保留；parameter 與 sample proposals 沿各自既有契約。

Analysis 與 Post-Analysis 各保存自己的 opaque draft。Remote 以 `(tab_id, subtab_id)` 定址 pane，再以 item session id 選項目。這個 id 只屬於 draft，不是 editor handle。`tab.writeback_set` 的 module edits 使用 Form 路徑陣列，回同一項目的 schema、input、values 與 errors。結構錯誤拒絕該 Form batch，普通 invalid 值留在 draft，Apply 與 `tab.writeback_write` 拒絕提交。

Selected Apply 先準備所有選中項目的值與目的地。任一 module form Invalid 時，不 append accepted、不寫 target。準備成功後，既有 Context write owner 先 append accepted，再依序呼叫 setters。Setter 首錯停止，保留 accepted evidence 與成功前綴，不 rollback，也不保證跨檔 crash transaction。`tab.writeback_write` 依序調整 draft；後項失敗可保留前項 draft 修改，這不等於已寫入 target。

Draft 固定來源的 run_id 與 SavedPoint。後來切 point、改 alias 或切換 pane 都不改目的地。Host 的 busy、writable、result、context 與 [[0070]] holder admission 仍各自生效。可預期的 `LibraryEntryInvalid` 保留 dialog、row、selection 與 draft，供使用者修正；非預期錯誤向外傳遞。

## 取捨與相鄰責任

GUI 與 remote 共用 headless state，所以 agent 的修改可由相應 editor 呈現，不必先開 dialog。各宿主仍有不同的提交用途，不能把 tab Run、Library Apply 與 writeback selected Apply 當成同一種交易。

Context 寫入權威見 [[0067]]，保存與 accepted evidence 見 [[0063]]，wire 與錯誤投影見 [[0068]]。舊 CfgDraft、CfgFormWidget、CfgEditorService 與 editor.* RPC 已退休，不提供相容入口。
