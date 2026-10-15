# gui/form — Qt-free Pydantic input editing

**Last updated:** 2026-10-14，統一 options schema 與 fresh defaults

`static_defaults(model)` 接受普通 Pydantic BaseModel 或持久 Spec，產生 detached input JSON。必填欄位與 None 預設保持缺席，讓 before validator 能從其他欄位推導值。Default factory 每次呼叫重新執行。FormModel defaults 與普通 model 的分支切換共用這個入口。

FormModel 的實例只承載普通值。類別上的 defaults、choices、options 與 input_schema 提供輸入契約。FormInput 保存未填、null、expression 與 complex tags；FormState 擁有 draft/base、batch write、branch switch、結果、差異與 compatible restore，不擁有 Engine revision 或提交 policy。

Expression 只用於文字欄位：int、float、complex 與 str value leaves 只對顯式 `{"expr": str}` 求值。以下拉或勾選呈現的 leaf 沒有 expression：Literal、Enum、bool、宣告 choices 的 leaf，以及 library key（`KeyRef.name`，包括 `ModuleRef.name`）。這些 leaf 的 input schema 沒有 Expr 分支；寫入的 envelope 留在 input，result 在該欄位報錯且不求值（choices leaf 由 core 報「只接受列出的值」，其餘由 Pydantic 報型別錯誤）；compatible restore 丟棄它們的 envelope（choices 隨 env 變，restore 不看 choices）。同時含文字成員與選項成員的 union（例如 `float | Literal["auto"]`）沒有對應的控制項，input schema、求值與 compatible restore 遇到時 raise ValueError，訊息帶欄位路徑。判斷規則只在 core 一處，schema、求值、defaults 與 restore 都呼叫它；editor 只看 schema 有沒有 Expr 分支。普通字串，包括 `$` 開頭的內容，仍是 literal。Core 使用共用 evaluate_scalar_expr，再遵守 Pydantic 約束；missing／unbound alias 指名失敗，不以預設值代替。Input schema 與 compatible restore 保留 expression 原文，GUI 與 MCP 共用這份輸入契約。

Defaults hook 可以用 `{"expr": ...}` 宣告 scalar 初始值的來源，例如 `$flux_dev` 或 point path。`default_input` 與分支切換的 defaults 由 core 用同一個 env 當場求值，input 只存算好的 direct 值；來源之後改變，要重新建立 default input 才反映。Expression 算不出來，或值讓 result 在該欄位報錯時，欄位留空：required 欄位只顯示 `Field required`，有 Pydantic 靜態預設的欄位驗證時用那個預設。來源本身不可用的錯誤照常拋出。使用者或 MCP 寫入的 expression 保留原文。

`Option` 宣告 label、部分 JSON input value、可選 BaseModel fields 與 commands。`options(env)` 是宣告 hook，consumer 讀 schema 的 `x-form-options` 有效清單。Core 自動產生 Literal、Enum、choices 與 tagged union 選項。最深 local 宣告優先，包含表示無選項的空清單，非重疊 parent 宣告保留。Bool 與一般 nullable model 照舊用 checkbox 與 Enabled。Fields 投影成內嵌子 schema，不顯示 value 已決定的 tag。比對只看 value 的每個葉，JSON 型別與值相同即符合，第一項匹配優先。無匹配時 consumer 顯示目前值加 `(missing)`，不寫 state。

Union 位置的 partial object 是選分支加寫入內容，與 tag 葉切換共用新分支 defaults。不繼承舊值，defaults 只收到固定 env。非換分支的 leaf write 與 compatible restore 規則不變。明確指令用 `OptionCommand(name, label, run)` 宣告，schema 只列 name 與 label。`FormState.run_command(path, name, env)` 只找目前 active branch 第一個匹配選項的指令。Run callable 讀同一份固定 env，回傳該欄位的部分 input，core 用一次 write 提交。未知或不適用指令、來源與結構失敗不寫入，不建立第二份 state。`defaults(env)` 只依型別與固定來源產生新 input，沒有 previous 參數或隱含 ref 複製。

每個 intrinsic tag 都有一個普通 Option。多個 tag 指向同一 model 時，各選項共用 branch fields，保留 tag 的 JSON 型別與宣告順序。Core 從 annotation 讀取 tag，不從動態 choices 或 consumer 標記推導身分。Model 欄位可達的每個 branch 都是一個具體 model，同一 tag key 的具名 union 展開成各成員。Choices 仍限制當次可接受的 leaf 值，不改目前 input 的 branch。

`FormModel.default_input`、`FormModel.input_schema` 與 `FormInput` 在入口拒絕未標 tag 的 `str | Model` 欄位。ValueError 指出欄位路徑與不支援的形狀。檢查走過 nullable、alias 與所有 model branches，包含未啟用的分支；它不讀 captured sources。Key-or-inline input 必須使用明確的 tagged ref／inline union，不能靠 JSON 字串或物件型別選分支。持久 Spec 的 `str | Spec` 不改，生成器在 input 邊界投影成 tagged wrappers。

Module 欄位使用 `ModuleRef` 與 `ModuleInline` 衍生 class 的 tagged union，由 `kind` 選分支。Ref 的 `name` 是 literal key，inline 的 `module` 是支援的 input model。原生 `ModuleInput.module` 是持久 `ModuleSpec`，內含未標 tag unions，不能直接放進手寫 FormModel。Caller 使用 `gui.form_adapter` 的生成欄位與選項。生成器宣告 None、library keys、Inline kinds 與 `to_inline` 指令，core 不替 consumer 列 library 清單。選項的 fields 不含 value 已固定的 tag，ref 選項不帶 fields。

Editor 與 MCP 讀同一份 `x-form-options`，照 label 列選項，一次寫入 value，只展開選中項目的 fields。它們不推論 union、ref 或 library 的呈現。普通 nullable model 仍用 Enabled，bool 仍用 checkbox。明確複製只經 `to_inline` 指令，使用 host 的 guarded command callback；一般選 Inline 只建立 fresh defaults。

Scalar、整個 sequence 與整個 typed mapping 都是 leaf。普通 nested model 只編輯子欄位，一般可留空 model 區段仍用 Enabled 控制 object／null 啟停。Choices 使用 model-field leaf paths，驗證與 schema 共用同一份當次選項。Parent 與 child 宣告同一 leaf 時，最深 nested FormModel 的 local choices 優先，包含空選項。非重疊的 parent 選項保持有效，重用同型 nested model 的欄位各自隔離。

Compatible restore 以原文欄位集合與型別選擇未標 tag 的 model-only union。恰好一個 member 相容才保留 subtree，合法 expression envelope 依欄位能力參與比對，不求值。零個或多個 member 相容就丟棄，不猜第一個 branch。Tagged branch 的 selector 與 defaults 契約不變。還原後 captured source 檢查失敗的 leaf（例如 library 沒有的 ref 名稱，或沒有 captured library 時的任何 ref）保留目前 input，目前沒有就留空，路徑和其他 dropped paths 一起回傳。不是由還原 leaf 造成的 source 失敗仍然 raise；一般 write、default_input 與 result 的 source 語意不變。

`KeyRef` 是 key-only 的 tagged ref 基底，欄位為 `kind="ref"` 與 literal `name: str`。Schema 不提供 Expr，result 不求值 Expr envelope，compatible restore 丟棄 envelope。它不讀 library，也不限制 key 清單。`ModuleRef` 繼承它，保留 module-only captured library source 檢查。生成器的 nested waveform／module ref 使用 `KeyRef`；來源政策不依 input 的位置決定。

FormEnv 借用 host 捕捉的 ParameterContext、library 與 detached `BaseDeviceInfo` snapshots（含 custom driver info）。共用工具只讀裝置名稱；需要特定 driver 欄位的表單自己用 `isinstance` 收窄。沒有 parameter capture 時仍可編輯 ordinary input 與純數學 expression；參數引用失敗會落在消費欄位的 error path。工具不寫來源、不 refresh 或 poll。ModuleRef/ModuleInline 使用公開 library Spec 讀取；to_cfg 的 materialization 與 range 展開歸具體表單。

FormState 的 subscriber 在 input 安裝後同步執行。個別 subscriber 故障記錄 traceback，不回滾 input，也不阻止後續 subscriber 或 host publication。Host 的 mutation guard 涵蓋安裝、editor subscriber 與 resource publication，同一通知期間可讀不可重入 mutation。

FormState.preview 提供隔離的 ordinary leaf values 與完整 errors，供 view 在 model-invalid draft 中顯示其他欄位的新求值結果。失敗的 expression 或含失敗子值的 whole container 不提供值。成功時包含 Pydantic defaults 與 coercions，不把它們寫回 input。Preview values 不代表可提交；host 仍檢查 errors 與 view 的 local parse errors。

WriteRejected 是 host 寫入入口拒絕一批輸入時 raise 的具名例外，message 由 host 寫給使用者看。Host 在寫入任何內容前 raise，FormState 維持原狀；model 會拒絕的值仍寫進 state，由 result 回報。Core 自己不 raise 它。Editor 收到時保留該欄輸入並顯示 message，其他例外照常拋出。

Display 只承載欄位呈現資訊。順序依欄位宣告，段落依 nested model；collapsed 只用於子 model 欄位。`group` 把欄位歸到同一 model 層級內的具名畫面群組（例如 `"Advanced"`），空字串在建構時報錯；schema 只在宣告的 property 節點標 `x-form-group`，不改驗證、defaults、choices 或 input。`decimals` 是 float 欄位顯示的小數位數，只在宣告時進 schema，非 float 欄位宣告就報錯；它不改 input。`hidden=True` 的欄位不畫，值與預設留在 input、照常驗證，schema 仍列出它並標 `hidden`。巢狀 model 欄位沒有 label 時，schema title 是欄位名（底線換空白、只有首字母大寫，`ro_cfg` 是 `Ro cfg`），不是 Pydantic 帶進來的類別名。Model 不經 union selector 到達時，它身為 tagged union branch 的單值 tag 是固定的結構 tag，schema 標 `hidden`；branch 身分來自本表單與 ModuleInput 可達的 tagged union，含 branch 的子類別（inline readout 的 `ro_cfg.type`；`LinRange.kind` 即使沒有 union 也保持結構 tag）。Union 位置的 tag 仍是 selector。Editor、host 與 MCP 共用 core 行為，但不反向成為 core dependency。

掃描輸入只提供 `LinRange`，由表單 owner 在轉 cfg 時展開 start／stop／n。不提供 explicit-list 分支。

跨模組契約見 docs/adr/draft/pydantic-form-tool.md。這是共用表單工具，不改 CfgSchema。Core cfg 推導與 module materialization 由上層 `gui.form_adapter` 組合，不放進本 package。
