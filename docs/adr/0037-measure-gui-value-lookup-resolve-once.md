---
status: accepted
---

# ADR-0037 — Session value lookup 與 resolve-once 輸入

**狀態：** accepted。

## 脈絡

Adapter defaults 和 GUI／agent 輸入有時需要 cached device values、session 路徑或 predictor metadata。穩定的參數讀取使用固定 `ParameterContext` 與 typed helpers。Session value lookup 只處理少量跨來源的 scalar 查詢，不取代參數容器或 alias 表。

ContextService 的寫入權威見 [[0067]]；Form、expression 與固定來源見 [[0065]]，表單初值與範本見 [[0073]]。

## 決策

`ValueLookup` 提供列舉與 typed scalar 讀取。`ValueRegistry` 提供註冊，由 session composition 和 source owners 使用。`SessionEnv.values` 只公開 read-only lookup，adapter 不取得 mutable registry 或 app service。

Provider 依 owner 成組替換或移除。Context、predictor 或 device 更新後，source binder 更新對應 providers，避免留下舊來源。Provider 同步、只讀且不做硬體 I/O。Device provider 只讀 `SessionState.devices[*].info` 的 cache。Lookup 區分 missing key、known-but-unavailable、provider failure、重複註冊和型別錯誤。

Cached device values 供 defaults 和 resolve-once 輸入使用，不是 Run 的硬體 snapshot。GUI Run 框架在取得 hardware lease 後、呼叫 adapter.run 前讀取所有已連線 device；讀取失敗不以 lookup cache 補值。Snapshot 的保存見 [[0063]]。

內建來源投影 SessionEnv 的 cached metadata、predictor metadata 與具名 device cache。ContextService 只發布 entry 的 result／database paths 與 active_label。元件名稱使用 `ParameterContext` 的固定 alias capture。Device keys 包括 `device.<name>.name` 和 `device.<name>.value`。名稱中的 device 是具體裝置名稱；lookup 不推論哪個裝置代表 flux。

## 參數與 expression

參數讀取使用 `ParameterContext` 的 point／library 固定投影。`$qubit.freq` 是 alias 路徑，`Q1.freq` 是具體元件路徑。`$qubit.module.x180.waveform.length` 先沿 point 的 module slot 找 library，再讀 module leaf。Alias 的 authored links 只沿這份 capture 求值，不讀目前 session。

裸 `$alias` 回傳固定表解析後的名稱字串，例如 `$qubit` 得到 `Q1`。Alias 容器可保存元件或 device 名稱，不在 capture 或 preview 預查名稱可用性。實驗室表單以元件為焦點，裝置初值讀取 `$qubit.flux_source` 等元件欄位，不要求 device alias，見 [[0073]]。缺列或未綁定時指名失敗，不代用同名元件、舊解析值或靜態名稱。名稱不接受數值算術或字串插值。

Form input 以顯式 `{expr: text}` 保存 expression，普通字串保持 literal。只有文字欄位可用 Expression，下拉與勾選欄位沒有這個模式。求值沿宿主的固定 FormEnv，不讀後來的全域來源。Measure Run 使用目前有效的 Form result；Library Apply 將結果數字寫進持久 Spec，不保存 expression。

Form defaults 的 expression 在建立 input 時求成 direct 值。使用者輸入的 expression 保留原文，重算仍使用同一 capture。開啟與 Reset 的捕捉時機見 [[0065]]、[[0073]]。

## Resolve-once 輸入

`ValueRef` 與 `resolve_value_ref` 是內部 typed scalar 讀取入口。呼叫時讀 registered value，回普通值，不保存 lazy reference。Target type、declared type 與 provider type 不相容時拒絕，不以字串猜型別。

Remote 的 `value.list`／`value.read` 保留唯讀 lookup。舊 editor 的 `value_ref` tagged wire、eval tag 與 `@{}` 文字補全已退休。這不移除內部 ValueRef，也不改變已捕捉的 Form／Run 來源。

## 寫入與限制

ContextService 接收已解析的普通值，透過參數容器或 Library owner 提交。Inspect Parameters 編輯 concrete point／setup 路徑，Modules 提交持久 Spec，都不把 ValueRef 當成持久來源。

數值 expression 使用 `gui.session.expression` 的受限 simpleeval 引擎。它沿固定 ParameterContext 解析參數，限制 AST、函式與數值大小，不使用 Python `eval` 或 SymPy。Lookup 不提供通用複合 source-expression parser，也不應取代穩定的 typed dependencies。

## 後果

Owner-scoped provider lifecycle 避免舊來源殘留。唯讀 lookup、內部立即讀值、Form expression 與 Run acceptance 各有自己的用途。新增 lookup key 前，先確認它確實是跨來源需求。
