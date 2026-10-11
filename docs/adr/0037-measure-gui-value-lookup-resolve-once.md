# ADR-0037 — Session value lookup 與 resolve-once 輸入

**狀態：** accepted。

## 脈絡

Adapter defaults 和 GUI／agent 輸入有時需要 cached device values、session 路徑或 predictor metadata。穩定的參數讀取使用固定 `ParameterContext` 與 typed helpers。Session value lookup 只處理少量跨來源的 scalar 查詢，不取代參數容器或 alias 表。

ContextService 的寫入權威見 [[0067]]；cfg resource、expression 與獨立 draft 的分界見 [[0065]]；context-free definition 和 seed 見 [[0012]]。

## 決策

`ValueLookup` 提供列舉與 typed scalar 讀取。`ValueRegistry` 提供註冊，由 session composition 和 source owners 使用。`SessionEnv.values` 只公開 read-only lookup，adapter 不取得 mutable registry 或 app service。

Provider 依 owner 成組替換或移除。Context、predictor 或 device 更新後，source binder 更新對應 providers，避免留下舊來源。Provider 同步、只讀且不做硬體 I/O。Device provider 只讀 `SessionState.devices[*].info` 的 cache。Lookup 區分 missing key、known-but-unavailable、provider failure、重複註冊和型別錯誤。

Cached device values 供 defaults 和 resolve-once 輸入使用，不是 Run 的硬體 snapshot。GUI Run 框架在取得 hardware lease 後、呼叫 adapter.run 前讀取所有已連線 device；讀取失敗不以 lookup cache 補值。Snapshot 的保存見 [[0063]]。

內建來源投影 SessionEnv 的 cached metadata、predictor metadata 與具名 device cache。ContextService 發布 entry 的 result／database paths 與 active_label；`context.chip_name`、`context.qub_name` 和 `context.res_name` 讀取各自 cached 欄位，不從 alias 表推導名稱。Device keys 包括 `device.<name>.name` 和 `device.<name>.value`。名稱中的 device 是具體裝置名稱；lookup 不推論哪個裝置代表 flux。

## 參數與 expression

參數讀取使用 `ParameterContext` 的 point／library 固定投影。`$qubit.freq` 是 alias 路徑，`Q1.freq` 是具體元件路徑。`$qubit.module.x180.waveform.length` 先沿 point 的 module slot 找 library，再讀 module leaf。Alias 的 authored links 只沿這份 capture 求值，不讀目前 session。

裸 `$alias` 回傳固定表解析後的名稱字串，例如 `$flux_dev` 得到 `yoko1`。元件與 device 共用這張表，不在 capture 或 preview 預查名稱可用性。缺列或未綁定時指名失敗，不代用同名元件、舊解析值或靜態名稱。名稱不接受數值算術或字串插值。

`EvalValue` 保存 expression 與解析結果。Measure tab 的 cfg resource 在 publication 時計算結果，Run 接受當前 Valid publication 的已解析 values。獨立 draft 的 legacy lowering 可重新求值並檢查型別，失敗就拒絕，不是 tab Run 的入口。

Cfg expression 的 `$alias` 與 `$alias.path` 是動態 alias 引用。`@{key}` 才是寫入時捕捉一次固定 source view 的數值，轉成括號 literal。捕捉失敗拒絕整批，不保留待未來求值的 token。精確來源與解析邊界由 [[0065]] 和 cfg owner 定義。

## Resolve-once 輸入

`ValueRef` 表示「現在讀取 registered value，存入普通值」，不保存 lazy reference。Agent 使用明確 tagged object，例如：

```json
{"__kind":"value_ref","key":"device.flux.value","type":"float"}
```

這裡的 `flux` 是具體 registered device name，不是 alias。普通 wire 字串不自動解讀為 ValueRef。

GUI 的 `value_source_input.py` 在 `@{prefix` 提供分段 completion。`@{` 列頂層，`device.` 列下一層。Tab／Backtab 接受候選；若候選都是 namespace，controller 補上句點並展開下一層，完整 leaf 才補上 `}`。輸入 `@{full.key} ` 的尾端空格後，controller 讀取一次並用格式化數值替換 token。這是文字輸入便利功能，與 cfg resource 的 expression capture 分開。

Definition 的 typed `value_source(...)` Seed 和 role-default `Source(...)` 在 fresh cfg instantiate 時使用 `SessionEnv.values`，然後存 direct value。Restore 或使用者編輯不重跑 seed。

## 寫入與限制

ContextService 接收已解析的普通值，透過參數容器或 library owner 提交。Inspect 編輯 concrete point／setup 路徑，不把 ValueRef 當成持久來源。GUI token helper 只替換輸入文字，不自行寫文件。

數值 expression 使用 `gui.session.expression` 的受限 simpleeval 引擎。它沿固定 ParameterContext 解析參數，限制 AST、函式與數值大小，不使用 Python `eval` 或 SymPy。Lookup 不提供通用複合 source-expression parser，也不應取代穩定的 typed dependencies。

## 後果

GUI 和 agent 共用 source lookup，但各自提供文字 token 與 tagged-object 輸入。Owner-scoped provider lifecycle 避免舊來源殘留。參數、alias、cached value source 和 Run acceptance 各有明確 owner；新增 lookup key 前，先確認它確實是跨來源需求。
